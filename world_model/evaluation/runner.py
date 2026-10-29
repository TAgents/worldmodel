"""Causal evaluation and reproducible archives for the first sprint."""

import hashlib
import math
import platform
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path
from typing import Literal

from world_model.core import ModelArtifact
from world_model.core.serialization import canonical_json
from world_model.core.types import Contract, NonNegativeInt
from world_model.inference import INFERENCE_VERSION
from world_model.synthetic.generator import Dataset, GENERATOR_VERSION, generate_dataset

from .client import PredictorProcess
from .metrics import BOOTSTRAP_RESAMPLES, BOOTSTRAP_SEED, losses, paired_interval, summarize
from .protocol import METHODS, PredictionBatch, StepRequest, readings_hash

Split = Literal["development", "reserved"]
SPLIT_SEEDS = {"development": tuple(range(20)), "reserved": tuple(range(10000, 10100))}


class PredictionRecord(Contract):
    episode_index: NonNegativeInt
    batch: PredictionBatch


@dataclass(frozen=True)
class EvaluationResult:
    report: dict
    predictions: tuple[PredictionRecord, ...]


def evaluate_dataset(
    artifact: ModelArtifact,
    dataset: Dataset,
    *,
    source_id: str = "cheap",
    target_state: str = "degraded",
    split: Split = "development",
    bootstrap_resamples: int = BOOTSTRAP_RESAMPLES,
) -> EvaluationResult:
    """Issue and freeze each batch before consulting its next-state label."""
    model = artifact.spec
    if split not in SPLIT_SEEDS:
        raise ValueError("unknown evaluation split")
    seeds = tuple(episode.seed for episode in dataset.episodes)
    if len(seeds) < 2 or not set(seeds).issubset(SPLIT_SEEDS[split]):
        raise ValueError("evaluation needs at least two episodes belonging to its declared split")
    if type(bootstrap_resamples) is not int or bootstrap_resamples < 2:
        raise ValueError("bootstrap needs at least two resamples")
    if split == "reserved" and (
        seeds != SPLIT_SEEDS["reserved"] or dataset.horizon != 200
        or bootstrap_resamples != BOOTSTRAP_RESAMPLES
        or source_id != "cheap" or target_state != "degraded"
    ):
        raise ValueError("reserved evaluation requires the complete prespecified protocol")
    if dataset.model_hash != artifact.content_hash:
        raise ValueError("dataset and inference must use the same model artifact")
    if set(dataset.source_ids) != set(model.observation_models):
        raise ValueError("dataset sources must match the model")
    if target_state not in model.states or source_id not in model.observation_models:
        raise ValueError("target state and acquisition source must exist in the model")
    if any(set(episode.states) - set(model.states) for episode in dataset.episodes):
        raise ValueError("dataset contains an unknown state label")
    limits = model.acquisition_limits
    cost = dataset.horizon * model.observation_models[source_id].read_cost_credits
    if dataset.horizon > limits.episode_origin_ticks or dataset.horizon > limits.max_reads_per_episode:
        raise ValueError("fixed acquisition schedule exceeds the episode/read limit")
    if cost > limits.episode_budget_credits:
        raise ValueError("fixed acquisition schedule exceeds the information budget")

    forecasts = {name: [] for name in METHODS}
    outcomes = []
    episode_scores = {name: [] for name in METHODS}
    records = []
    with PredictorProcess(artifact) as predictor:
        for episode_index, episode in enumerate(dataset.episodes):
            predictor.reset()
            selected = [reading for reading in episode.readings if reading.source_id == source_id]
            history = []
            scores = {name: [] for name in METHODS}
            for tick, reading in enumerate(selected):
                history.append(reading)
                batch = predictor.step(StepRequest(tick=tick, readings=(reading,)))
                if batch.acquired_reading_count != len(history) or batch.acquired_readings_hash != readings_hash(history):
                    raise RuntimeError("predictor evidence does not match the acquired prefix")
                records.append(PredictionRecord(episode_index=episode_index, batch=batch))
                # The immutable forecast has returned from the separate process.
                # Only now may the evaluator consult its target label.
                outcome = episode.states[batch.target_tick] == target_state
                outcomes.append(outcome)
                for name in METHODS:
                    probability = batch.predictions[name][target_state]
                    forecasts[name].append(probability)
                    scores[name].append(losses(probability, outcome))
            for name in METHODS:
                episode_scores[name].append(tuple(
                    math.fsum(score[index] for score in scores[name]) / dataset.horizon
                    for index in range(2)
                ))

    comparisons = {}
    for baseline in METHODS[1:]:
        intervals = {}
        for index, metric in enumerate(("brier", "log_loss")):
            pairs = list(zip(episode_scores["filter"], episode_scores[baseline], strict=True))
            differences = [left[index] - right[index] for left, right in pairs]
            intervals[metric] = (
                paired_interval(differences, resamples=bootstrap_resamples)
                if all(math.isfinite(value) for value in differences) else None
            )
        comparisons[baseline] = intervals
    comparison_met = all(comparison["brier"]["upper_95"] < 0 for comparison in comparisons.values())
    report = {
        "schema_version": 1,
        "protocol": "first-sprint-v1",
        "split": split,
        "model_hash": artifact.content_hash,
        "dataset_sha256": hashlib.sha256(dataset.to_bytes()).hexdigest(),
        "generator_version": GENERATOR_VERSION,
        "inference_version": INFERENCE_VERSION,
        "episode_count": len(dataset.episodes),
        "origin_ticks_per_episode": dataset.horizon,
        "forecasts_per_method": len(outcomes),
        "target_state": target_state,
        "acquisition": {
            "source_id": source_id, "schedule": "one read per origin tick",
            "reads_per_episode": dataset.horizon, "credits_per_episode": cost,
            "budget_credits_per_episode": limits.episode_budget_credits,
        },
        "methods": {name: summarize(forecasts[name], outcomes) for name in METHODS},
        "filter_minus_baseline": comparisons,
        "bootstrap": {
            "unit": "paired whole episodes", "resamples": bootstrap_resamples,
            "seed": BOOTSTRAP_SEED, "interval": "percentile 95%, linear quantiles",
        },
        "brier_comparison_condition_met": comparison_met,
        "reserved_predictive_gate_passed": comparison_met if split == "reserved" else None,
        "log_loss_convention": "No clipping; mean_log_loss is null when infinite_log_loss_count > 0. Nonfinite log-loss differences have no interval.",
    }
    return EvaluationResult(report=report, predictions=tuple(records))


def archive_result(
    destination: str | Path, artifact: ModelArtifact, dataset: Dataset, result: EvaluationResult
) -> Path:
    """Preserve bytes, code, scores, and hashes; never overwrite an existing run."""
    destination = Path(destination)
    if dataset.model_hash != artifact.content_hash or result.report["model_hash"] != artifact.content_hash:
        raise ValueError("archive model and results do not match")
    if result.report["dataset_sha256"] != hashlib.sha256(dataset.to_bytes()).hexdigest():
        raise ValueError("archive dataset and results do not match")
    files = {
        "model.json": artifact.content,
        "dataset.json": dataset.to_bytes(),
        "predictions.jsonl": b"".join(canonical_json(record.model_dump(mode="json")) for record in result.predictions),
        "report.json": canonical_json(result.report),
    }
    if hasattr(result, "actions"):
        files["actions.jsonl"] = b"".join(canonical_json(record.model_dump(mode="json")) for record in result.actions)
    package_root = Path(__file__).resolve().parents[1]
    for path in sorted(package_root.rglob("*.py")):
        files[f"source/world_model/{path.relative_to(package_root).as_posix()}"] = path.read_bytes()
    manifest = {
        "schema_version": 1,
        "generator_version": GENERATOR_VERSION,
        "inference_version": INFERENCE_VERSION,
        "runtime": {
            "python": platform.python_version(),
            "pydantic": version("pydantic"), "pydantic_core": version("pydantic_core"),
        },
        "files": {name: hashlib.sha256(content).hexdigest() for name, content in files.items()},
    }
    # The manifest is the completion marker. An interrupted write cannot look
    # like a complete archive, and a retry must choose a new destination.
    destination.mkdir(parents=True, exist_ok=False)
    for name, content in files.items():
        path = destination / name
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as stream:
            stream.write(content)
    with (destination / "manifest.json").open("xb") as stream:
        stream.write(canonical_json(manifest))
    return destination


def run_benchmark(
    artifact: ModelArtifact, *, destination: str | Path, split: Split = "development",
    source_id: str = "cheap", target_state: str = "degraded",
) -> EvaluationResult:
    if Path(destination).exists():
        raise FileExistsError(f"archive already exists: {destination}")
    if split not in SPLIT_SEEDS:
        raise ValueError("unknown evaluation split")
    dataset = generate_dataset(artifact, seeds=SPLIT_SEEDS[split])
    result = evaluate_dataset(artifact, dataset, source_id=source_id, target_state=target_state, split=split)
    archive_result(destination, artifact, dataset, result)
    return result

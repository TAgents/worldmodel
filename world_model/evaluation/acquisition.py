"""Equal-cap, paired sensor-policy benchmark with an isolated decision process."""

from dataclasses import dataclass
from hashlib import sha256
from math import fsum, isfinite
from pathlib import Path

from world_model.agents.client import AgentProcess
from world_model.agents.contracts import InvestigationConfig, POLICY_VERSION
from world_model.agents.session import Frame
from world_model.core import ModelArtifact
from world_model.core.types import Contract, NonNegativeInt, Probability
from world_model.synthetic.generator import Dataset, GENERATOR_VERSION, generate_dataset
from .metrics import BOOTSTRAP_RESAMPLES, losses, paired_interval, summarize
from .protocol import readings_hash
from .runner import archive_result

ACQUISITION_SPLITS = {"development": tuple(range(20)), "heldout": tuple(range(2000, 2020))}


class AcquisitionPrediction(Contract):
    episode_index: NonNegativeInt
    method: str
    origin_tick: NonNegativeInt
    target_tick: NonNegativeInt
    probability: Probability
    spent_credits: NonNegativeInt
    dispatched_reads: NonNegativeInt
    acquired_readings_hash: str


class AcquisitionAction(Contract):
    episode_index: NonNegativeInt
    method: str
    ordinal: NonNegativeInt
    frame: Frame


@dataclass(frozen=True)
class AcquisitionResult:
    report: dict
    predictions: tuple[AcquisitionPrediction, ...]
    actions: tuple[AcquisitionAction, ...]


def evaluate_acquisition(artifact: ModelArtifact, dataset: Dataset, *, target_state="degraded",
                         budget_credits=None, split="development", bootstrap_resamples=BOOTSTRAP_RESAMPLES):
    if split not in ACQUISITION_SPLITS:
        raise ValueError("unknown acquisition evaluation split")
    if len(dataset.episodes) < 2 or not {episode.seed for episode in dataset.episodes} <= set(ACQUISITION_SPLITS[split]):
        raise ValueError("use at least two episodes from the declared acquisition split")
    if dataset.model_hash != artifact.content_hash or set(dataset.source_ids) != set(artifact.spec.observation_models):
        raise ValueError("dataset and model do not match")
    if target_state not in artifact.spec.states or any(set(e.states) - set(artifact.spec.states) for e in dataset.episodes):
        raise ValueError("unknown state label")
    limits = artifact.spec.acquisition_limits
    budget = limits.episode_budget_credits if budget_credits is None else budget_credits
    sources = tuple(sorted(artifact.spec.observation_models))
    common = dict(allowed_sources=sources, horizon=dataset.horizon, budget_credits=budget,
                  max_reads=limits.max_reads_per_episode, min_gain_bits_per_credit=0.0, entropy_threshold_bits=0.0)
    configs = {"information_gain": InvestigationConfig(**common), "random": InvestigationConfig(**common, policy="random")}
    configs.update({f"fixed_{source}": InvestigationConfig(**common, policy="fixed", fixed_source=source) for source in sources})
    if split == "heldout" and (tuple(e.seed for e in dataset.episodes) != ACQUISITION_SPLITS[split] or dataset.horizon != 200 or budget != 200 or target_state != "degraded" or bootstrap_resamples != BOOTSTRAP_RESAMPLES):
        raise ValueError("heldout acquisition requires the complete prespecified 200-tick, 200-credit protocol")
    predictions, actions, outcomes = [], [], []
    probabilities = {method: [] for method in configs}
    episode_scores = {method: [] for method in configs}
    costs = {method: [] for method in configs}
    read_counts = {method: [] for method in configs}
    with AgentProcess(artifact) as worker:
        for episode_index, episode in enumerate(dataset.episodes):
            available = {(reading.tick, reading.source_id): reading for reading in episode.readings}
            for method, config in configs.items():
                frame = worker.configure(config)
                history, scores = [], []
                for ordinal in range(config.horizon + config.max_reads + 1):
                    actions.append(AcquisitionAction(episode_index=episode_index, method=method, ordinal=ordinal, frame=frame))
                    decision = frame.decision
                    if frame.acquired_readings_hash != readings_hash(history):
                        raise RuntimeError("agent evidence diverged from acquired readings")
                    if frame.spent_credits > budget or frame.dispatched_reads != len(history):
                        raise RuntimeError("agent exceeded its acquisition contract")
                    if decision.kind == "stop":
                        if decision.tick != dataset.horizon:
                            raise RuntimeError("agent stopped before completing the scoring horizon")
                        break
                    if decision.kind == "read":
                        reading = available[decision.tick, decision.source_id]
                        history.append(reading)
                        frame = worker.accept(reading)
                    else:
                        # Freeze the complete prediction returned from a worker
                        # which has never received hidden labels or future data.
                        prediction = AcquisitionPrediction(episode_index=episode_index, method=method,
                            origin_tick=decision.tick, target_tick=decision.tick + 1,
                            probability=frame.predictive[target_state], spent_credits=frame.spent_credits,
                            dispatched_reads=frame.dispatched_reads, acquired_readings_hash=frame.acquired_readings_hash)
                        predictions.append(prediction)
                        outcome = episode.states[prediction.target_tick] == target_state
                        if method == "information_gain":
                            outcomes.append(outcome)
                        probabilities[method].append(prediction.probability)
                        scores.append(losses(prediction.probability, outcome))
                        frame = worker.accept()
                else:
                    raise RuntimeError("agent did not stop within its bounded action count")
                if len(scores) != dataset.horizon:
                    raise RuntimeError("agent did not issue one forecast per tick")
                episode_scores[method].append(tuple(fsum(score[i] for score in scores) / len(scores) for i in range(2)))
                costs[method].append(frame.spent_credits)
                read_counts[method].append(frame.dispatched_reads)
    comparisons = {}
    for baseline in configs:
        if baseline == "information_gain":
            continue
        comparisons[baseline] = {}
        for i, metric in enumerate(("brier", "log_loss")):
            differences = [left[i] - right[i] for left, right in zip(episode_scores["information_gain"], episode_scores[baseline], strict=True)]
            comparisons[baseline][metric] = paired_interval(differences, resamples=bootstrap_resamples) if all(isfinite(d) for d in differences) else None
    improvement = all(item["brier"]["upper_95"] < 0 for item in comparisons.values())
    report = {
        "schema_version": 1, "protocol": "paced-acquisition-v1", "policy_version": POLICY_VERSION,
        "model_hash": artifact.content_hash, "dataset_sha256": sha256(dataset.to_bytes()).hexdigest(),
        "generator_version": GENERATOR_VERSION, "split": split, "episode_count": len(dataset.episodes),
        "origin_ticks_per_episode": dataset.horizon, "target_state": target_state,
        "budget_credits_per_episode": budget,
        "pacing": "At tick t, floor(budget * (t+1) / horizon) credits are released cumulatively. Unspent credits carry forward.",
        "comparison": "Equal credit caps, identical latent worlds and potential readings, same inference and scoring horizon. Actual spend is reported, not forced equal.",
        "configs": {name: config.model_dump(mode="json") for name, config in configs.items()},
        "methods": {name: {**summarize(values, outcomes), "credits_per_episode": costs[name], "reads_per_episode": read_counts[name]} for name, values in probabilities.items()},
        "information_gain_minus_baseline": comparisons,
        "beats_all_baselines_at_95_percent": improvement,
        "heldout_acquisition_gate_passed": improvement if split == "heldout" else None,
        "reserved_first_sprint_seeds_used": False,
        "bootstrap": {"unit": "paired whole episodes", "resamples": bootstrap_resamples, "seed": 20261027},
    }
    return AcquisitionResult(report=report, predictions=tuple(predictions), actions=tuple(actions))


def run_acquisition_benchmark(artifact, *, destination, split="development"):
    if Path(destination).exists():
        raise FileExistsError(f"archive already exists: {destination}")
    if split not in ACQUISITION_SPLITS:
        raise ValueError("unknown acquisition split")
    dataset = generate_dataset(artifact, seeds=ACQUISITION_SPLITS[split])
    result = evaluate_acquisition(artifact, dataset, split=split)
    archive_result(destination, artifact, dataset, result)
    return result

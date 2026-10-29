"""Prespecified complementary-sensor experiment; training and scoring stay separate."""

from dataclasses import dataclass
from hashlib import sha256
from importlib.metadata import version
import io
import json
from math import fsum, isfinite, log, sqrt
from pathlib import Path
import platform

import numpy as np

from world_model.agents.contracts import InvestigationConfig
from world_model.core import ModelArtifact
from world_model.core.serialization import canonical_json
from world_model.evaluation.metrics import paired_interval, summarize
from world_model.evaluation.protocol import readings_hash
from world_model.inference import filter_state
from world_model.synthetic.generator import generate_dataset
from .calibration import BetaPrior, fit_calibration, mean_artifact, verify_revision
from .client import LearnedProcess
from .inference import INFERENCE_VERSION, parameter_bank
from .synthetic import calibration_cutoff, generate_calibration, neutral_structure

LOCKED_PROTOCOL = "9257cc61f7ca45c0a14261f048c56d7d8f74695b3db23b89823550855c8e732e"
LOCKED_TRUTH = "1ac94b72d35429066366c08117efdbae2ceaf508adc8391411150f5a3e90e20d"


@dataclass(frozen=True)
class ReliabilityResult:
    report: dict
    predictions: tuple[dict, ...]
    actions: tuple[dict, ...]


def parameter_metrics(truth, revision, validation):
    cells = {(c.source_id, c.state): c for c in revision.cells}
    errors, parameters = [], []
    for (source, state), cell in cells.items():
        target = truth.spec.observation_models[source].alert_probability[state]
        errors.append(cell.mean - target)
        parameters.append({**cell.model_dump(mode="json"), "synthetic_truth": target,
                           "truth_in_interval": cell.lower_95 <= target <= cell.upper_95})
    probabilities = [cells[o.source_id, o.state].mean for o in validation]
    scores = summarize(probabilities, [o.value for o in validation])
    return {"rmse": sqrt(fsum(e * e for e in errors) / len(errors)),
        "maximum_absolute_error": max(abs(e) for e in errors), "parameters": parameters,
        "interval_coverage_count": sum(p["truth_in_interval"] for p in parameters),
        "parameter_count": len(parameters), "mean_interval_width": fsum(c.upper_95 - c.lower_95 for c in cells.values()) / len(cells),
        "validation": scores}


def _score(distribution, state):
    p = distribution[state]
    return fsum((value - float(label == state))**2 for label, value in distribution.items()), -log(p) if p else float("inf")


def evaluate_reliability(truth, dataset, revision, protocol, *, split="development"):
    if split not in ("development", "heldout"):
        raise ValueError("unknown reliability split")
    structure = neutral_structure(truth)
    verify_revision(structure, revision)
    seeds = tuple(e.seed for e in dataset.episodes)
    if len(seeds) < 2 or not set(seeds) <= set(protocol[f"{split}_seeds"]):
        raise ValueError("use at least two episodes from the declared split")
    if dataset.model_hash != truth.content_hash or set(dataset.source_ids) != set(truth.spec.observation_models):
        raise ValueError("dataset and truth model must match")
    if dataset.horizon != protocol["horizon"] or any(set(e.states) - set(truth.spec.states) for e in dataset.episodes):
        raise ValueError("invalid scoring horizon or state support")
    if split == "heldout":
        if sha256(canonical_json(protocol)).hexdigest() != LOCKED_PROTOCOL or truth.content_hash != LOCKED_TRUTH or seeds != tuple(protocol["heldout_seeds"]):
            raise ValueError("heldout reliability evaluation requires the complete locked protocol")
        expected = fit_calibration(structure, generate_calibration(truth, seed=protocol["calibration_seed"], samples_per_state=protocol["calibration_samples_per_state"]),
            as_of=calibration_cutoff(structure, protocol["calibration_samples_per_state"]), prior=BetaPrior(alpha=protocol["prior_alpha"], beta=protocol["prior_beta"]))
        if revision != expected:
            raise ValueError("heldout evaluation requires the prespecified calibration snapshot")
    sources = tuple(sorted(structure.spec.observation_models))
    common = dict(allowed_sources=sources, horizon=dataset.horizon, budget_credits=protocol["budget_credits"], max_reads=protocol["max_reads"], policy_seed=protocol["policy_seed"])
    configs = {"learned_adaptive": InvestigationConfig(**common),
               **{f"learned_fixed_{source}": InvestigationConfig(**common, policy="fixed", fixed_source=source) for source in sources},
               "learned_random": InvestigationConfig(**common, policy="random")}
    matched = {"mean_likelihood_matched": mean_artifact(structure, revision),
               "neutral_likelihood_matched": structure, "oracle_likelihood_matched": truth}
    methods = [*configs, *matched]
    distributions, outcomes = {m: [] for m in methods}, []
    episode_scores = {m: [] for m in methods}
    costs, reads = {m: [] for m in methods}, {m: [] for m in methods}
    effective = {m: [] for m in configs}
    sensor_counts = {m: {s: 0 for s in sources} for m in configs}
    predictions, actions = [], []
    with LearnedProcess(structure) as worker:
        for episode_index, episode in enumerate(dataset.episodes):
            available = {(r.tick, r.source_id): r for r in episode.readings}
            for method, config in configs.items():
                frame = worker.configure(revision, config, particles=protocol["particle_count"], particle_seed=protocol["particle_seed"])
                history, scores = [], {method: [], **({m: [] for m in matched} if method == "learned_adaptive" else {})}
                for ordinal in range(config.horizon + config.max_reads + 1):
                    actions.append({"episode_index": episode_index, "method": method, "ordinal": ordinal, "frame": frame.model_dump(mode="json")})
                    effective[method].append(frame.effective_particle_count)
                    decision = frame.decision
                    if frame.acquired_readings_hash != readings_hash(history) or frame.spent_credits > config.budget_credits or frame.dispatched_reads != len(history):
                        raise RuntimeError("acquired evidence or accounting diverged from the worker")
                    if frame.calibration_revision != revision.revision_id or frame.structure_hash != structure.content_hash:
                        raise RuntimeError("worker used a different calibrated model")
                    if decision.kind == "stop":
                        if decision.tick != dataset.horizon:
                            raise RuntimeError("unexpected early stopping")
                        break
                    if decision.kind == "read":
                        reading = available[decision.tick, decision.source_id]
                        history.append(reading)
                        sensor_counts[method][decision.source_id] += 1
                        frame = worker.accept(reading)
                    else:
                        frozen = {method: dict(frame.predictive)}
                        if method == "learned_adaptive":
                            # These ablations use exactly the adaptive acquired
                            # prefix. Truth parameters are confined to this
                            # evaluator-only oracle, never sent to the learner.
                            frozen.update({name: dict(filter_state(model, history, state_tick=decision.tick).predictive) for name, model in matched.items()})
                        for name, distribution in frozen.items():
                            predictions.append({"episode_index": episode_index, "method": name,
                                "origin_tick": decision.tick, "target_tick": decision.tick + 1,
                                "distribution": distribution, "calibration_revision": revision.revision_id,
                                "spent_credits": frame.spent_credits, "acquired_readings_hash": frame.acquired_readings_hash})
                        # All predictions exist before looking up the label.
                        outcome = episode.states[decision.tick + 1]
                        if method == "learned_adaptive":
                            outcomes.append(outcome)
                        for name, distribution in frozen.items():
                            distributions[name].append(distribution)
                            scores[name].append(_score(distribution, outcome))
                        frame = worker.accept()
                else:
                    raise RuntimeError("learner failed to stop within its action bound")
                for name, scored in scores.items():
                    if len(scored) != dataset.horizon:
                        raise RuntimeError("missing forecast")
                    episode_scores[name].append(tuple(fsum(s[i] for s in scored) / len(scored) for i in range(2)))
                    costs[name].append(frame.spent_credits)
                    reads[name].append(frame.dispatched_reads)
    comparisons = {}
    for baseline in methods[1:]:
        comparisons[baseline] = {}
        for index, metric in enumerate(("brier", "log_loss")):
            differences = [a[index] - b[index] for a, b in zip(episode_scores["learned_adaptive"], episode_scores[baseline], strict=True)]
            comparisons[baseline][metric] = paired_interval(differences, resamples=protocol["bootstrap_resamples"]) if all(isfinite(d) for d in differences) else None
    summaries = {}
    for name in methods:
        per_state = {s: summarize([d[s] for d in distributions[name]], [o == s for o in outcomes]) for s in truth.spec.states}
        means = episode_scores[name]
        summaries[name] = {"mean_brier": fsum(s[0] for s in means) / len(means),
            "mean_log_loss": fsum(s[1] for s in means) / len(means) if all(isfinite(s[1]) for s in means) else None,
            "forecast_count": len(outcomes), "per_state": per_state, "credits_per_episode": costs[name],
            "reads_per_episode": reads[name], "sensor_counts": sensor_counts.get(name),
            "minimum_effective_particles": min(effective[name]) if name in effective else None}
    beats_fixed = all(comparisons[name]["brier"]["upper_95"] < 0 for name in configs if name != "learned_adaptive")
    numerical_pass = min(min(v) for v in effective.values()) >= .1 * protocol["particle_count"]
    return ReliabilityResult(report={
        "protocol": protocol["protocol"], "split": split, "protocol_sha256": sha256(canonical_json(protocol)).hexdigest(),
        "truth_model_hash": truth.content_hash, "structure_hash": structure.content_hash,
        "calibration_revision": revision.revision_id, "inference_version": INFERENCE_VERSION,
        "episode_count": len(seeds), "origin_ticks_per_episode": dataset.horizon, "states": list(truth.spec.states),
        "acquisition_budget": protocol["budget_credits"], "particle_count": protocol["particle_count"],
        "methods": summaries, "adaptive_minus_baseline": comparisons,
        "numerical_ess_gate_passed": numerical_pass, "beats_fixed_policies": beats_fixed,
        "heldout_acquisition_gate_passed": beats_fixed and numerical_pass if split == "heldout" else None,
        "scope": protocol["scope"], "score_definition": protocol["scores"],
        "uncertainty_definition": protocol["uncertainty"], "automatic_promotion": False,
    }, predictions=tuple(predictions), actions=tuple(actions))


def run_reliability_benchmark(protocol_path, *, destination, split="development"):
    destination, protocol_path = Path(destination), Path(protocol_path)
    if destination.exists():
        raise FileExistsError(f"archive already exists: {destination}")
    protocol = json.loads(protocol_path.read_bytes())
    if sha256(canonical_json(protocol)).hexdigest() != LOCKED_PROTOCOL:
        raise ValueError("this command runs only the locked v1 protocol; use a new protocol version for new experiments")
    if split not in ("development", "heldout"):
        raise ValueError("unknown split")
    truth = ModelArtifact.load(protocol_path.parent / protocol["scenario_file"])
    if truth.content_hash != LOCKED_TRUTH:
        raise ValueError("the prespecified truth artifact changed")
    structure = neutral_structure(truth)
    training = generate_calibration(truth, seed=protocol["calibration_seed"], samples_per_state=protocol["calibration_samples_per_state"])
    validation = generate_calibration(truth, seed=protocol["calibration_validation_seed"], samples_per_state=protocol["calibration_validation_samples_per_state"])
    prior = BetaPrior(alpha=protocol["prior_alpha"], beta=protocol["prior_beta"])
    revisions = [fit_calibration(structure, training, as_of=calibration_cutoff(structure, n), prior=prior) for n in protocol["calibration_curve_sizes"]]
    learning_curve = [{"samples_per_sensor_state": n, "revision_id": rev.revision_id, **parameter_metrics(truth, rev, validation)} for n, rev in zip(protocol["calibration_curve_sizes"], revisions, strict=True)]
    revision = revisions[-1]
    dataset = generate_dataset(truth, seeds=protocol[f"{split}_seeds"], horizon=protocol["horizon"])
    result = evaluate_reliability(truth, dataset, revision, protocol, split=split)
    metric = learning_curve[-1]
    parameter_gate = (metric["rmse"] <= protocol["parameter_gate"]["maximum_rmse"] and
        metric["maximum_absolute_error"] <= protocol["parameter_gate"]["maximum_absolute_error"] and
        metric["validation"]["mean_log_loss"] < learning_curve[0]["validation"]["mean_log_loss"])
    calibration_reads = len(training)
    calibration_labels = len({o.measurement_id for o in training})
    report = result.report | {"learning_curve": learning_curve, "parameter_recovery_gate_passed": parameter_gate,
        "calibration_cost": {"sensor_readings": calibration_reads, "verified_labels": calibration_labels,
            "sensor_credits": sum(truth.spec.observation_models[o.source_id].read_cost_credits for o in training),
            "label_credits": calibration_labels * protocol["calibration_label_cost_credits_per_example"],
            "scope": "One shared upfront calibration set for all learned policies; acquisition caps exclude this disclosed cost. Neutral ablation ignores calibration."}}
    result = ReliabilityResult(report, result.predictions, result.actions)
    bank = parameter_bank(structure, revision, particles=protocol["particle_count"], seed=protocol["particle_seed"])
    buffer = io.BytesIO()
    np.save(buffer, bank, allow_pickle=False)
    contents = {
        "protocol.json": canonical_json(protocol), "truth-model.json": truth.content, "structure.json": structure.content,
        "training.json": canonical_json([o.model_dump(mode="json") for o in training]),
        "calibration-validation.json": canonical_json([o.model_dump(mode="json") for o in validation]),
        "calibration-revisions.json": canonical_json([r.model_dump(mode="json") for r in revisions]),
        "dataset.json": dataset.to_bytes(), "parameter-draws.npy": buffer.getvalue(),
        "predictions.jsonl": b"".join(canonical_json(p) for p in result.predictions),
        "actions.jsonl": b"".join(canonical_json(a) for a in result.actions), "report.json": canonical_json(report),
    }
    package = Path(__file__).resolve().parents[1]
    for path in sorted(package.rglob("*.py")):
        contents[f"source/world_model/{path.relative_to(package).as_posix()}"] = path.read_bytes()
    manifest = {"protocol": protocol["protocol"], "files": {name: sha256(content).hexdigest() for name, content in contents.items()},
        "runtime": {"python": platform.python_version(), **{name: version(name) for name in ("numpy", "scipy", "pydantic", "pydantic_core")}}}
    destination.mkdir(parents=True, exist_ok=False)
    for name, content in contents.items():
        path = destination / name
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as file:
            file.write(content)
    with (destination / "manifest.json").open("xb") as file:
        file.write(canonical_json(manifest))
    return result

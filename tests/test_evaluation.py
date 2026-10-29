import hashlib
import json
import os
import subprocess
import sys

import pytest
from pydantic import ValidationError

from world_model.core import ModelArtifact, Reading
from world_model.evaluation.client import PredictorProcess
from world_model.evaluation.metrics import losses
from world_model.evaluation.protocol import REQUEST, StepRequest
from world_model.evaluation.runner import SPLIT_SEEDS, archive_result, evaluate_dataset, run_benchmark
from world_model.evaluation.worker import ForecastWorker
from world_model.synthetic.generator import Dataset, generate_dataset

from .conftest import ROOT
from .oracle import enumerate_distribution


@pytest.mark.parametrize("extra", ["seed", "states", "ground_truth", "outcome", "future_readings"])
def test_worker_protocol_has_no_channel_for_hidden_truth(extra):
    with pytest.raises(ValidationError):
        REQUEST.validate_python({"operation": "step", "tick": 0, "readings": [], extra: []})


def test_worker_rejects_future_readings_and_duplicate_sensor_dispatches():
    with pytest.raises(ValidationError):
        StepRequest(tick=0, readings=[Reading(tick=1, source_id="cheap", value=True)])
    reading = Reading(tick=0, source_id="cheap", value=True)
    with pytest.raises(ValidationError):
        StepRequest(tick=0, readings=[reading, reading])


def test_baselines_retain_exactly_their_specified_evidence(artifact):
    worker = ForecastWorker(artifact)
    initial = Reading(tick=0, source_id="cheap", value=True)
    latest = Reading(tick=1, source_id="cheap", value=False)
    worker.step(StepRequest(tick=0, readings=[initial]))
    batch = worker.step(StepRequest(tick=1, readings=[latest]))
    evidence = {"filter": [initial, latest], "transition_only": [], "memoryless": [latest]}
    for name, readings in evidence.items():
        expected = enumerate_distribution(artifact.content, readings, 2)
        for state, probability in expected.items():
            assert batch.predictions[name][state] == pytest.approx(float(probability), abs=1e-12, rel=0)
    assert batch.acquired_reading_count == 2


def test_rejected_steps_do_not_advance_worker_state(artifact):
    worker = ForecastWorker(artifact)
    with pytest.raises(ValueError):
        worker.step(StepRequest(tick=1, readings=[]))
    with pytest.raises(ValueError):
        worker.step(StepRequest(tick=0, readings=[Reading(tick=0, source_id="unknown", value=True)]))
    assert worker.step(StepRequest(tick=0, readings=[])).origin_tick == 0


def test_predictor_runs_in_a_separate_process_and_resets_between_episodes(artifact):
    with PredictorProcess(artifact) as predictor:
        process = predictor._process
        assert process.pid != os.getpid()
        reading = Reading(tick=0, source_id="cheap", value=True)
        first = predictor.step(StepRequest(tick=0, readings=[reading]))
        predictor.step(StepRequest(tick=1, readings=[]))
        predictor.reset()
        assert predictor.step(StepRequest(tick=0, readings=[reading])) == first
    assert process.poll() == 0


def test_protocol_errors_close_cleanly(artifact):
    with pytest.raises(RuntimeError, match="exactly one tick"):
        with PredictorProcess(artifact) as predictor:
            predictor.step(StepRequest(tick=2, readings=[]))


def test_hidden_labels_cannot_change_predictions(artifact):
    dataset = generate_dataset(artifact, seeds=[0, 1], horizon=4)
    original = evaluate_dataset(artifact, dataset, bootstrap_resamples=20)
    raw = dataset.model_dump(mode="json")
    for episode in raw["episodes"]:
        episode["states"] = ["degraded" if state == "healthy" else "healthy" for state in episode["states"]]
    changed = evaluate_dataset(artifact, Dataset.model_validate(raw), bootstrap_resamples=20)
    assert changed.predictions == original.predictions
    assert changed.report["methods"]["filter"]["mean_brier"] != original.report["methods"]["filter"]["mean_brier"]


def test_future_readings_cannot_change_earlier_predictions(artifact):
    dataset = generate_dataset(artifact, seeds=[0, 1], horizon=4)
    original = evaluate_dataset(artifact, dataset, bootstrap_resamples=20)
    raw = dataset.model_dump(mode="json")
    for episode in raw["episodes"]:
        for reading in episode["readings"]:
            if reading["tick"] == 3:
                reading["value"] = not reading["value"]
    changed = evaluate_dataset(artifact, Dataset.model_validate(raw), bootstrap_resamples=20)
    for old, new in zip(original.predictions, changed.predictions, strict=True):
        if old.batch.origin_tick < 3:
            assert old == new


def test_scores_align_with_next_state_and_every_tick_is_counted(artifact):
    dataset = generate_dataset(artifact, seeds=[0, 1], horizon=4)
    result = evaluate_dataset(artifact, dataset, bootstrap_resamples=20)
    scores = []
    for record in result.predictions:
        episode = dataset.episodes[record.episode_index]
        outcome = episode.states[record.batch.target_tick] == "degraded"
        scores.append(losses(record.batch.predictions["filter"]["degraded"], outcome)[0])
    assert len(scores) == result.report["forecasts_per_method"] == 8
    assert result.report["methods"]["filter"]["mean_brier"] == pytest.approx(sum(scores) / 8)
    assert result.report["reserved_predictive_gate_passed"] is None
    assert result.report["acquisition"]["credits_per_episode"] == 4


def test_over_budget_schedule_is_rejected_before_starting_predictor(artifact, monkeypatch):
    dataset = generate_dataset(artifact, seeds=[0, 1], horizon=41)
    def fail_if_started(*args, **kwargs):
        pytest.fail("a rejected schedule must not start a worker")
    monkeypatch.setattr("world_model.evaluation.runner.PredictorProcess", fail_if_started)
    with pytest.raises(ValueError, match="budget"):
        evaluate_dataset(artifact, dataset, source_id="precise")


def test_model_and_dataset_mismatch_is_rejected(artifact):
    dataset = generate_dataset(artifact, seeds=[0, 1], horizon=2)
    reformatted = ModelArtifact(json.dumps(json.loads(artifact.content)).encode())
    with pytest.raises(ValueError, match="same model"):
        evaluate_dataset(reformatted, dataset)


def test_development_data_cannot_be_presented_as_reserved_evaluation(artifact):
    # Inspect split configuration without generating any reserved episode.
    assert SPLIT_SEEDS["development"] == tuple(range(20))
    assert SPLIT_SEEDS["reserved"] == tuple(range(10000, 10100))
    dataset = generate_dataset(artifact, seeds=[0, 1], horizon=2)
    with pytest.raises(ValueError, match="declared split"):
        evaluate_dataset(artifact, dataset, split="reserved")


def test_archive_hashes_reproduce_every_file_and_refuse_overwrite(artifact, tmp_path):
    dataset = generate_dataset(artifact, seeds=[0, 1], horizon=2)
    result = evaluate_dataset(artifact, dataset, bootstrap_resamples=20)
    output = archive_result(tmp_path / "run", artifact, dataset, result)
    manifest_bytes = (output / "manifest.json").read_bytes()
    manifest = json.loads(manifest_bytes)
    for name, expected in manifest["files"].items():
        assert hashlib.sha256((output / name).read_bytes()).hexdigest() == expected
    assert (output / "model.json").read_bytes() == artifact.content
    assert (output / "source/world_model/synthetic/generator.py").is_file()
    assert Dataset.model_validate_json((output / "dataset.json").read_bytes()) == dataset
    assert len((output / "predictions.jsonl").read_text().splitlines()) == 4
    with pytest.raises(FileExistsError):
        archive_result(output, artifact, dataset, result)
    assert (output / "manifest.json").read_bytes() == manifest_bytes
    other = generate_dataset(artifact, seeds=[2, 3], horizon=2)
    with pytest.raises(ValueError, match="dataset and results"):
        archive_result(tmp_path / "mismatched", artifact, other, result)


def test_existing_archive_is_rejected_before_regeneration(artifact, tmp_path, monkeypatch):
    def fail_if_generated(*args, **kwargs):
        pytest.fail("an existing archive must not trigger another generation")
    monkeypatch.setattr("world_model.evaluation.runner.generate_dataset", fail_if_generated)
    with pytest.raises(FileExistsError):
        run_benchmark(artifact, destination=tmp_path)


def test_evaluation_cli_creates_a_complete_development_archive(artifact, tmp_path):
    raw = json.loads(artifact.content)
    raw["acquisition_limits"]["episode_origin_ticks"] = 2
    model_path = tmp_path / "short-model.json"
    model_path.write_text(json.dumps(raw))
    output = tmp_path / "development"
    command = [
        sys.executable, "-m", "world_model.evaluation", "--model", str(model_path), "--output", str(output),
    ]
    run = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, check=True, timeout=30)
    summary = json.loads(run.stdout)
    assert summary["split"] == "development"
    assert summary["episode_count"] == 20
    assert summary["forecasts_per_method"] == 40
    assert (output / "manifest.json").is_file()
    retry = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, timeout=30)
    assert retry.returncode == 2
    assert "already exists" in retry.stderr

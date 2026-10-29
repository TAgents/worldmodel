from copy import deepcopy
from datetime import timedelta
from importlib.resources import files
import json
from unittest.mock import Mock

from fastapi.testclient import TestClient
import pytest

from world_model.adapters.github_actions import ActionsConfig, prepare_import, operations
from world_model.api import create_app
from world_model.core import ModelArtifact
from world_model.core.serialization import canonical_json
from world_model.storage.contracts import ConflictError
from world_model.storage.contracts import ReplayIntegrityError, WriteResult
from world_model.storage.store import PostgresStore


@pytest.fixture
def actions():
    root = files("world_model.adapters").joinpath("fixtures")
    return (ModelArtifact(root.joinpath("software-delivery-v1.json").read_bytes()),
            root.joinpath("github-actions-example.json").read_bytes(),
            ActionsConfig.model_validate_json(root.joinpath("github-actions-config.json").read_bytes()))


def test_windows_preserve_timestamps_failures_and_missingness(actions):
    model, content, config = actions
    plan = prepare_import(*actions)
    assert plan.epoch == config.window_start + timedelta(days=1)
    assert plan.final_tick == 6
    assert plan.unique_run_count == 12
    assert len(plan.readings) == 9
    by_slot = {(r.payload.source_id, r.payload.tick): r for r in plan.readings}
    assert by_slot["unit_tests", 3].payload.value is True  # one success cannot erase one failure
    assert len(by_slot["unit_tests", 3].runs) == 2
    assert by_slot["integration_tests", 3].payload.value is True  # timeout
    assert by_slot["unit_tests", 0].payload.value is False
    assert ("integration_tests", 2) not in by_slot  # cancelled
    assert ("unit_tests", 6) not in by_slot  # unfinished
    assert not any(r.payload.tick == 4 for r in plan.readings)  # no records that day
    assert len(operations(plan)) == 15
    for reading in plan.readings:
        assert reading.window_end == plan.epoch + timedelta(days=reading.payload.tick)
        assert all(reading.window_start <= r.updated_at < reading.window_end for r in reading.runs)
        assert all(r.url.startswith("https://github.com/example/delivery-demo/actions/runs/") for r in reading.runs)


def test_pagination_duplicates_and_conflicting_attempts(actions):
    model, content, config = actions
    raw = json.loads(content)
    pages = [{"total_count":12, "workflow_runs":raw["workflow_runs"][:6]}, {"total_count":12, "workflow_runs":raw["workflow_runs"][6:]}]
    assert prepare_import(model, canonical_json(pages), config).readings == prepare_import(*actions).readings
    pages[1]["workflow_runs"].append(raw["workflow_runs"][0])
    assert prepare_import(model, canonical_json(pages), config).duplicate_count == 1
    pages[1]["workflow_runs"][-1] = raw["workflow_runs"][0] | {"run_attempt":2}
    with pytest.raises(ValueError, match="conflicting"):
        prepare_import(model, canonical_json(pages), config)
    with pytest.raises(ValueError, match="incomplete"):
        prepare_import(model, canonical_json(pages[:1]), config)


@pytest.mark.parametrize("change", [lambda r: r.update(total_count=True), lambda r: r["workflow_runs"][0].update(id=True),
    lambda r: r["workflow_runs"][0].update(updated_at="2025-01-01T00:00:00Z"),
    lambda r: r["workflow_runs"][0].update(updated_at="2026-01-01T10:10:00"),
    lambda r: r["workflow_runs"][0].update(repository={"full_name":"another/repository"})])
def test_invalid_exports_are_rejected(actions, change):
    model, content, config = actions
    raw = json.loads(content)
    change(raw)
    with pytest.raises(ValueError):
        prepare_import(model, canonical_json(raw), config)


def test_cohort_boundaries_and_later_records_do_not_change_earlier_readings(actions):
    model, content, config = actions
    plan = prepare_import(*actions)
    raw = json.loads(content)
    for run in raw["workflow_runs"]:
        if run["updated_at"] >= "2026-01-03":
            run["conclusion"] = "failure"
    changed = prepare_import(model, canonical_json(raw), config)
    assert [r for r in plan.readings if r.payload.tick < 2] == [r for r in changed.readings if r.payload.tick < 2]
    raw = json.loads(content)
    raw["workflow_runs"][0]["updated_at"] = "2026-01-02T02:00:00+02:00"
    moved = prepare_import(model, canonical_json(raw), config)
    assert next(r for r in moved.readings if any(x.run_id == 1001 for x in r.runs)).payload.tick == 1
    for field, value in (("head_branch", "feature"), ("event", "pull_request"), ("workflow_id", 999)):
        other = deepcopy(raw)
        other["workflow_runs"][0][field] = value
        selected = prepare_import(model, canonical_json(other), config)
        assert not any(r.run_id == 1001 for reading in selected.readings for r in reading.runs)


def test_unknown_conclusions_do_not_become_success(actions):
    model, content, config = actions
    raw = json.loads(content)
    raw["workflow_runs"][0]["conclusion"] = "future_provider_conclusion"
    plan = prepare_import(model, canonical_json(raw), config)
    assert not any(r.payload.source_id == "unit_tests" and r.payload.tick == 0 for r in plan.readings)
    # A known failure remains evidence even alongside an inconclusive result.
    raw["workflow_runs"][7]["conclusion"] = "cancelled"
    plan = prepare_import(model, canonical_json(raw), config)
    assert next(r for r in plan.readings if r.payload.source_id == "unit_tests" and r.payload.tick == 3).payload.value


def test_mapping_limits_and_generic_core_model(actions):
    model, content, config = actions
    with pytest.raises(ValueError, match="distinct"):
        ActionsConfig.model_validate(config.model_dump() | {"workflow_sources":{"unit_tests":101,"integration_tests":101}})
    with pytest.raises(ValueError, match="sensors"):
        prepare_import(model, content, config.model_copy(update={"workflow_sources":{"unknown":101}}))
    raw = model.spec.model_dump(mode="json")
    raw["observation_models"] = {"workflow_signal": raw["observation_models"]["unit_tests"]}
    raw["states"] = ["green", "amber"]
    raw["initial_distribution"] = {"green":.8, "amber":.2}
    raw["transition_probabilities"] = {"green":{"green":.9,"amber":.1},"amber":{"green":.2,"amber":.8}}
    raw["observation_models"]["workflow_signal"]["alert_probability"] = {"green":.1,"amber":.7}
    other = ModelArtifact(canonical_json(raw))
    plan = prepare_import(other, content, config.model_copy(update={"workflow_sources":{"workflow_signal":101}}))
    assert {r.payload.source_id for r in plan.readings} == {"workflow_signal"}


def test_preview_identity_changes_with_raw_bytes_and_blocks_stale_import(actions):
    model, content, config = actions
    original = prepare_import(*actions)
    reformatted = prepare_import(model, content + b" ", config)
    assert original.readings == reformatted.readings
    assert original.content_hash != reformatted.content_hash
    store = PostgresStore("unused")
    store._connect = Mock(side_effect=AssertionError("must not access storage"))
    with pytest.raises(ConflictError, match="preview changed"):
        store.import_github_actions(model, content, config, run_id="import", target_state="unstable", expected_plan_hash="0"*64)
    store._connect.assert_not_called()


def test_import_http_preview_source_and_input_contract(actions):
    model, content, config = actions
    plan = prepare_import(*actions)
    store = Mock(spec=PostgresStore)
    store.preview_github_actions.return_value = plan
    store.get_import_source.return_value = content
    body = {"model_hash":model.content_hash, "export_json":content.decode(), "config":config.model_dump(mode="json")}
    with TestClient(create_app(store), base_url="http://127.0.0.1") as client:
        response = client.post("/imports/github-actions/preview", json=body)
        assert response.status_code == 200
        assert response.json()["plan_hash"] == plan.content_hash
        store.preview_github_actions.assert_called_once_with(model.content_hash, content, config)
        assert client.post("/imports/github-actions/runs", json=body).status_code == 422
        from .import_fixtures import import_fixture
        publication = import_fixture()[-1][-1].publication
        for created, status in ((True, 201), (False, 200)):
            store.import_github_actions.return_value = WriteResult(created=created, publication=publication)
            request = body | {"run_id":"imported-run", "target_state":"unstable", "expected_plan_hash":plan.content_hash}
            response = client.post("/imports/github-actions/runs", json=request)
            assert response.status_code == status
            assert response.json()["publication"] == publication.model_dump(mode="json")
            assert store.import_github_actions.call_args.args == (model.content_hash, content, config)
        assert client.post("/imports/github-actions/preview", json=body | {"ingested_at":"2026-01-01T00:00:00Z"}).status_code == 422
        assert client.post("/imports/github-actions/preview", json=body, headers={"Origin":"https://elsewhere.example"}).status_code == 403
        source = client.get("/runs/import/import/source")
        assert source.content == content
        assert "attachment" in source.headers["content-disposition"]
        assert client.get("/imports").status_code == 200
        assert "script-src 'self'" in client.get("/imports").headers["content-security-policy"]
        assert client.get("/assets/imports.js").status_code == 200
        assert client.get("/imports/github-actions/example").json()["model_hash"] == model.content_hash


def test_imported_event_history_replays_against_exact_independent_oracle():
    from .import_fixtures import import_fixture
    from world_model.storage.replay import ReplayState
    model, _, _, events, snapshots = import_fixture()
    state = ReplayState(model, snapshots[0].run, "unstable")
    for event, snapshot in zip(events, snapshots, strict=True):
        state.apply(event, snapshot.publication)
    assert state.publication == snapshots[-1].publication
    assert len(state.closed_forecasts) == 6


def test_invalid_json_duplicate_keys_and_size_are_rejected(actions):
    model, _, config = actions
    for content in (b'{"total_count":0,"total_count":0,"workflow_runs":[]}',
                    b'{"total_count":0,"workflow_runs":[],"unexpected":NaN}', b'[]', b'{}', b' ' * (8*1024*1024+1)):
        with pytest.raises(ValueError):
            prepare_import(model, content, config)


def test_stored_source_verification_checks_hashes_translation_and_event_binding():
    from .import_fixtures import import_fixture
    model, content, plan, events, snapshots = import_fixture()
    row = {"adapter_version":plan.adapter_version,"source_hash":plan.source_hash,"plan_hash":plan.content_hash,
           "source_content":content,"plan_content":plan.content,"completed_sequence":15}
    source_events = deepcopy(events)
    def execute(query, parameters):
        result = Mock()
        if "SELECT * FROM run_imports" in query:
            result.fetchone.return_value = row
        elif "SELECT payload FROM events" in query:
            result.fetchone.return_value = {"payload":source_events[0]["payload"]}
        else:
            result.fetchall.return_value = source_events[1:parameters[1]+1]
        return result
    conn = Mock()
    conn.execute.side_effect = execute
    store = PostgresStore("unused")
    run = snapshots[0].run.model_dump()
    assert store._verify_import(conn, model, run, snapshots[-1].publication) == (9, plan.content_hash)
    assert store._verify_import(conn, model, run, snapshots[1].publication) == (1, plan.content_hash)
    source_events[1]["payload"]["payload"]["value"] = not source_events[1]["payload"]["payload"]["value"]
    with pytest.raises(ReplayIntegrityError, match="differs from its source"):
        store._verify_import(conn, model, run, snapshots[-1].publication)
    row["source_content"] += b" "
    with pytest.raises(ReplayIntegrityError, match="provenance"):
        store._read_import(conn, model, run)
    row = None
    with pytest.raises(ReplayIntegrityError, match="missing"):
        store._read_import(conn, model, run)

"""Browser acceptance checks using exact-oracle API fixtures, with no TCP server.

Requires the dev and browser extras. PostgreSQL behavior is checked separately by
scripts/check_postgres.py. Requests still traverse the real FastAPI application.
"""

import argparse
from pathlib import Path
import sys
from unittest.mock import Mock
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fastapi.testclient import TestClient
from playwright.sync_api import expect, sync_playwright

from world_model.api import create_app
from world_model.core import ModelArtifact
from world_model.storage.contracts import ReplayReport, WriteResult
from world_model.storage.store import PostgresStore
from tests.test_api import inspection_fixture


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--browser", help="Existing Chromium executable; otherwise use Playwright's installed browser")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/explorer")
    args = parser.parse_args()
    artifact = ModelArtifact.load(ROOT / "docs/fixtures/binary-world-v1.json")
    snapshots = [inspection_fixture(artifact, i) for i in range(7)]
    state = {"created": False, "sequence": 0, "unavailable": False}
    store = Mock(spec=PostgresStore)
    store.list_models.return_value = (snapshots[0].model.registration,)
    store.get_model.return_value = snapshots[0].model
    store.list_runs.side_effect = lambda **_: (snapshots[0].run,) if state["created"] else ()

    def create_run(**body):
        assert body["run_id"] == "run-1"
        assert body["target_state"] == "degraded"
        state["created"] = True
        return WriteResult(created=True, publication=snapshots[0].publication)

    def inspect(run_id, *, sequence_cutoff=None, **_):
        assert run_id == "run-1"
        if state["unavailable"]:
            import psycopg
            raise psycopg.OperationalError("fixture database unavailable")
        return snapshots[state["sequence"] if sequence_cutoff is None else sequence_cutoff]

    def ingest(run_id, payload):
        assert run_id == "run-1"
        next_sequence = state["sequence"] + 1
        observation = snapshots[next_sequence].observations[-1]
        for field in ("source_id", "tick", "value", "supersedes_id"):
            assert getattr(payload, field) == getattr(observation.payload, field)
        state["sequence"] = next_sequence
        return WriteResult(created=True, publication=snapshots[next_sequence].publication, observation=observation)

    def advance(run_id, tick):
        next_sequence = state["sequence"] + 1
        assert tick == snapshots[next_sequence].publication.belief.state_tick
        state["sequence"] = next_sequence
        return WriteResult(created=True, publication=snapshots[next_sequence].publication)

    def replay(run_id, *, sequence_cutoff=None, **_):
        sequence_cutoff = state["sequence"] if sequence_cutoff is None else sequence_cutoff
        snapshot = snapshots[sequence_cutoff]
        return ReplayReport(run_id=run_id, model_hash=artifact.content_hash, events_checked=sequence_cutoff + 1, publication=snapshot.publication, closed_forecast_ids=tuple(f.forecast.forecast_id for f in snapshot.forecasts if f.closed_at))

    store.create_run.side_effect = create_run
    store.inspect.side_effect = inspect
    store.ingest.side_effect = ingest
    store.advance_tick.side_effect = advance
    store.replay.side_effect = replay
    args.output.mkdir(parents=True, exist_ok=True)
    with TestClient(create_app(store), base_url="http://127.0.0.1:8000") as client, sync_playwright() as playwright:
        browser = playwright.chromium.launch(executable_path=args.browser)
        context = browser.new_context(viewport={"width": 1440, "height": 1050}, reduced_motion="reduce")

        def route_request(route):
            request = route.request
            url = urlsplit(request.url)
            if url.netloc != "127.0.0.1:8000":
                route.abort()
                return
            response = client.request(request.method, url.path + ("?" + url.query if url.query else ""), content=request.post_data_buffer, headers=request.headers)
            headers = {key: value for key, value in response.headers.items() if key not in {"content-length", "content-encoding"}}
            route.fulfill(status=response.status_code, headers=headers, body=response.content)

        context.route("**/*", route_request)
        page = context.new_page()
        errors = []
        page.on("pageerror", lambda error: errors.append(str(error)))
        page.goto("http://127.0.0.1:8000/")
        expect(page.get_by_text("A world begins with a prior.")).to_be_visible()
        page.locator("#new-run").click()
        page.locator("#create-id").fill("run-1")
        expect(page.locator("#create-submit")).to_be_enabled()
        page.locator("#create-submit").click()
        expect(page.locator("#distribution")).to_contain_text("10.00%")
        page.get_by_role("button", name="Record evidence", exact=True).click()
        expect(page.locator("#distribution")).to_contain_text("30.77%")
        page.locator("#advance").click()
        expect(page.locator("#state-tick")).to_have_text("TICK 1")
        page.locator("#reading-value").select_option("false")
        page.get_by_role("button", name="Record evidence", exact=True).click()
        expect(page.locator("#snapshot-meta")).to_contain_text("Revision #3")
        page.locator("#advance").click()
        expect(page.locator("#state-tick")).to_have_text("TICK 2")
        page.locator("#source").select_option("precise")
        page.locator("#reading-tick").fill("0")
        page.get_by_role("button", name="Record evidence", exact=True).click()
        expect(page.locator("#distribution")).to_contain_text("2.91%")
        page.locator("#correction").select_option("precise0")
        expect(page.locator("#source")).to_be_disabled()
        page.get_by_role("button", name="Record evidence", exact=True).click()
        expect(page.locator("#snapshot-meta")).to_contain_text("Revision #6")
        expect(page.locator("#belief-change")).to_contain_text("Correcting the precise reading at tick 0")
        page.locator("#verify").click()
        expect(page.locator("#message")).to_contain_text("Replay verified 7 events")
        page.screenshot(path=str(args.output / "desktop.png"), full_page=True)
        page.locator("#revision-select").select_option("1")
        expect(page.locator("#view-badge")).to_have_text("HISTORICAL SNAPSHOT")
        expect(page.locator("#distribution")).to_contain_text("30.77%")
        expect(page.locator("#evidence-count")).to_have_text("1")
        expect(page.locator("#evidence-list")).not_to_contain_text("corrected0")
        expect(page.locator("#forecasts-body")).not_to_contain_text("Frozen")
        expect(page.locator("#advance")).to_be_disabled()
        expect(page.get_by_role("button", name="Record evidence", exact=True)).to_be_disabled()
        page.locator("#verify").click()
        expect(page.locator("#message")).to_contain_text("Replay verified 2 events")
        page.locator("#return-live").click()
        expect(page.locator("#snapshot-meta")).to_contain_text("Revision #6")
        state["unavailable"] = True
        page.locator("#refresh").click()
        expect(page.locator("#workspace")).to_be_hidden()
        expect(page.locator("#message")).to_contain_text("Database unavailable")
        state["unavailable"] = False
        page.locator("#refresh").click()
        expect(page.locator("#workspace")).to_be_visible()
        page.set_viewport_size({"width": 390, "height": 844})
        assert page.evaluate("document.documentElement.scrollWidth <= innerWidth")
        page.screenshot(path=str(args.output / "mobile.png"), full_page=True)
        assert not errors, errors
        browser.close()
    print("Browser checks passed: creation, evidence, late arrival, correction, temporal views, replay, recovery, and mobile layout.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

// Exercise the shipped UI without a browser process or network listener.
// These are interaction checks; layout and browser APIs need check_explorer.py.
import assert from "node:assert/strict";
import {execFileSync} from "node:child_process";
import {randomUUID} from "node:crypto";
import {readFileSync} from "node:fs";
import {fileURLToPath} from "node:url";
import {JSDOM, VirtualConsole} from "jsdom";

const root = fileURLToPath(new URL("../", import.meta.url));
const {manual: fixtures, agent: agentFixtures} = JSON.parse(execFileSync(process.env.WORLD_MODEL_PYTHON || `${root}.venv/bin/python`, ["-c", [
  "import json",
  "from world_model.core import ModelArtifact",
  "from tests.test_api import inspection_fixture",
  "from tests.investigation_fixtures import investigation_snapshots",
  "artifact = ModelArtifact.load('docs/fixtures/binary-world-v1.json')",
  "print(json.dumps({'manual': [inspection_fixture(artifact, i).model_dump(mode='json') for i in range(7)], 'agent': [s.model_dump(mode='json') for s in investigation_snapshots(artifact)[0]]}))",
].join("\n")], {cwd: root, encoding: "utf8"}));
const errors = [];
const virtualConsole = new VirtualConsole();
virtualConsole.on("jsdomError", (error) => errors.push(error.message));
const dom = new JSDOM(readFileSync(`${root}world_model/api/static/index.html`, "utf8"), {
  url: "http://127.0.0.1:8000/", runScripts: "outside-only", virtualConsole,
});
const w = dom.window;
const $ = (id) => w.document.getElementById(id);
Object.defineProperty(w.crypto, "randomUUID", {value: randomUUID});
w.HTMLDialogElement.prototype.showModal = function () { this.open = true; };
w.HTMLDialogElement.prototype.close = function () { this.open = false; };
let sequence = 0, created = false, unavailable = false, loseResponse = false;
let lastMeasurement = null;
const measurements = new Map();
let agentCreated = false, agentIndex = 0, loseAgentResponse = false, holdAgentResponse = false, releaseAgentResponse;
const agentRequests = new Map();
const json = (value, status = 200) => ({ok: status < 400, status, json: async () => structuredClone(value)});
w.fetch = async (path, options = {}) => {
  const url = new URL(path, w.location.href);
  const body = options.body ? JSON.parse(options.body) : undefined;
  if (url.pathname === "/models") return json([fixtures[0].model.registration]);
  if (url.pathname.startsWith("/models/")) return json(fixtures[0].model);
  if (url.pathname === "/runs" && body) {
    assert.ok(["run-1", "agent-run"].includes(body.run_id));
    assert.equal(body.target_state, "degraded");
    assert.ok(Date.parse(body.epoch) < Date.now());
    if (body.run_id === "agent-run") {
      agentCreated = true;
      return json({created: true, publication: agentFixtures[0].publication}, 201);
    }
    created = true;
    return json({created: true, publication: fixtures[0].publication}, 201);
  }
  if (url.pathname === "/runs") return json([...(agentCreated ? [agentFixtures[0].run] : []), ...(created ? [fixtures[0].run] : [])]);
  if (url.pathname.startsWith("/runs/agent-run/")) {
    if (url.pathname.endsWith("/inspection")) {
      const selected = url.searchParams.has("sequence_cutoff") ? [0, 2, 3, 4, 5][Number(url.searchParams.get("sequence_cutoff"))] : agentIndex;
      return json(agentFixtures[selected]);
    }
    if (url.pathname.endsWith("/investigation")) {
      assert.equal(body.horizon, 2);
      assert.equal(body.budget_credits, 2);
      assert.equal(body.max_reads, 2);
      assert.deepEqual(body.allowed_sources, ["cheap", "precise"]);
      agentIndex = 1;
      return json(agentFixtures[agentIndex].investigation);
    }
    if (url.pathname.endsWith("/steps")) {
      if (agentRequests.has(body.request_id)) return json({created: false, step: agentRequests.get(body.request_id)});
      agentIndex++;
      const step = agentFixtures[agentIndex].investigation.steps.at(-1);
      agentRequests.set(body.request_id, step);
      if (loseAgentResponse) { loseAgentResponse = false; throw new Error("Agent response lost after commit"); }
      if (holdAgentResponse) {
        holdAgentResponse = false;
        await new Promise((resolve) => { releaseAgentResponse = resolve; });
      }
      return json({created: true, step}, 201);
    }
    throw new Error(`Unexpected agent request: ${path}`);
  }
  if (url.pathname.endsWith("/inspection")) {
    if (unavailable) return json({detail: "Database unavailable."}, 503);
    return json(fixtures[url.searchParams.has("sequence_cutoff") ? Number(url.searchParams.get("sequence_cutoff")) : sequence]);
  }
  if (url.pathname.endsWith("/observations")) {
    assert.equal(options.headers["Content-Type"], "application/json");
    if (measurements.has(body.measurement_id)) {
      assert.deepEqual(body, measurements.get(body.measurement_id));
      return json({created: false, publication: fixtures[sequence].publication});
    }
    const expected = fixtures[sequence + 1].observations.at(-1).payload;
    for (const key of ["tick", "source_id", "value", "supersedes_id"]) assert.equal(body[key], expected[key]);
    lastMeasurement = body.measurement_id;
    measurements.set(body.measurement_id, body);
    sequence++;
    if (loseResponse) { loseResponse = false; throw new Error("Connection lost after commit"); }
    return json({created: true, publication: fixtures[sequence].publication}, 201);
  }
  if (url.pathname.endsWith("/ticks")) {
    assert.equal(body.tick, fixtures[sequence + 1].publication.belief.state_tick);
    sequence++;
    return json({created: true, publication: fixtures[sequence].publication}, 201);
  }
  if (url.pathname.endsWith("/replay")) {
    const cutoff = url.searchParams.has("sequence_cutoff") ? Number(url.searchParams.get("sequence_cutoff")) : sequence;
    return json({events_checked: cutoff + 1, publication: fixtures[cutoff].publication});
  }
  throw new Error(`Unexpected request: ${path}`);
};

const wait = async (condition) => {
  for (let i = 0; i < 100; i++) {
    if (condition()) return;
    await new Promise((resolve) => setTimeout(resolve, 10));
  }
  assert.fail(`UI condition timed out. Message: ${$("message").textContent}; dialog: ${$("create-error").textContent}`);
};
const changed = (id, value) => { $(id).value = value; $(id).dispatchEvent(new w.Event("change", {bubbles: true})); };
const submit = (id) => $(id).dispatchEvent(new w.Event("submit", {bubbles: true, cancelable: true}));
const revision = (n) => wait(() => $("snapshot-meta").textContent.startsWith(`Revision #${n} ·`) && !$("refresh").disabled);
w.eval(readFileSync(`${root}world_model/api/static/explorer.js`, "utf8"));
await wait(() => !$("new-run").disabled);
assert.match($("empty").textContent, /A world begins with a prior/);
$("new-run").click();
await wait(() => !$("create-submit").disabled);
$("create-id").value = "run-1";
submit("create-form");
await revision(0);
assert.match($("distribution").textContent, /10.00%/);

// Losing the HTTP response after commit must retain the measurement identity.
loseResponse = true;
submit("observation-form");
await wait(() => !$("refresh").disabled && $("message").textContent.includes("Connection lost"));
const firstMeasurement = lastMeasurement;
submit("observation-form");
await revision(1);
assert.equal(sequence, 1);
assert.equal(lastMeasurement, firstMeasurement);
assert.match($("distribution").textContent, /30.77%/);
$("advance").click();
await revision(2);
changed("reading-value", "false");
submit("observation-form");
await revision(3);
$("advance").click();
await revision(4);
changed("source", "precise");
$("reading-tick").value = "0";
submit("observation-form");
await revision(5);
assert.match($("distribution").textContent, /2.91%/);
changed("correction", "precise0");
assert.equal($("source").disabled, true);
assert.equal($("reading-tick").disabled, true);
assert.equal($("reading-value").value, "true");
submit("observation-form");
await revision(6);
assert.match($("belief-change").textContent, /Correcting the precise reading at tick 0/);
assert.equal($("evidence-count").textContent, "3");
$("verify").click();
await wait(() => $("message").textContent.includes("Replay verified 7 events") && !$("refresh").disabled);

changed("revision-select", "1");
await revision(1);
assert.equal($("view-badge").textContent, "HISTORICAL SNAPSHOT");
assert.equal($("evidence-count").textContent, "1");
assert.doesNotMatch($("evidence-list").textContent, /corrected0|precise0/);
assert.doesNotMatch($("forecasts-body").textContent, /Frozen/);
assert.equal($("history-body").children.length, 2);
assert.equal($("observation-fields").disabled, true);
assert.equal($("advance").disabled, true);
$("verify").click();
await wait(() => $("message").textContent.includes("Replay verified 2 events") && !$("refresh").disabled);
$("return-live").click();
await revision(6);
assert.equal($("advance").disabled, false);
unavailable = true;
$("refresh").click();
await wait(() => $("message").textContent.includes("Database unavailable") && !$("refresh").disabled);
assert.equal($("workspace").hidden, true);
assert.equal($("observation-fields").disabled, true);
unavailable = false;
$("refresh").click();
await wait(() => !$("workspace").hidden && !$("refresh").disabled);
assert.match($("snapshot-meta").textContent, /Revision #6/);

// Configure a fresh synthetic investigation and exercise retries and pause.
$("new-run").click();
await wait(() => !$("create-submit").disabled);
$("create-id").value = "agent-run";
submit("create-form");
await wait(() => $("run-select").value === "agent-run" && !$("refresh").disabled);
$("agent-horizon").value = "2";
$("agent-budget").value = "2";
$("agent-max-reads").value = "2";
submit("agent-config-form");
await wait(() => !$("agent-session").hidden && !$("refresh").disabled);
assert.equal($("observation-fields").disabled, true);
assert.equal($("advance").disabled, true);
loseAgentResponse = true;
$("agent-step").click();
await wait(() => $("message").textContent.includes("Agent response lost") && !$("refresh").disabled);
$("agent-step").click();
await revision(1);
assert.equal(agentRequests.size, 1);
assert.equal($("agent-budget-summary").textContent, "1 / 2 credits spent");
assert.match($("agent-trace").textContent, /maximum expected information gain per credit/);
holdAgentResponse = true;
$("agent-run").click();
await wait(() => !!releaseAgentResponse);
assert.equal($("agent-pause").disabled, false);
$("agent-pause").click();
assert.match($("message").textContent, /paused/);
releaseAgentResponse();
await revision(2);
await new Promise((resolve) => setTimeout(resolve, 150));
assert.equal(agentRequests.size, 2);
$("agent-run").click();
await wait(() => $("agent-status").textContent === "COMPLETED" && !$("refresh").disabled);
assert.equal(agentRequests.size, 5);
assert.equal($("agent-budget-summary").textContent, "2 / 2 credits spent");
assert.equal($("agent-run").disabled, true);
assert.match($("agent-trace").textContent, /episode horizon reached/);
changed("revision-select", "1");
await revision(1);
assert.equal($("agent-trace").children.length, 1);
assert.equal($("agent-budget-summary").textContent, "1 / 2 credits spent");
assert.equal($("agent-step").disabled, true);
$("return-live").click();
await revision(4);
assert.equal($("agent-trace").children.length, 5);
assert.deepEqual(errors, []);
dom.window.close();
console.log("Explorer DOM checks passed: manual evidence, historical isolation, retry identity, replay, recovery, agent configuration, action costs, lost responses, pause/resume, and bounded completion.");

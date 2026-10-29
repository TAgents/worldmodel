import assert from "node:assert/strict";
import {execFileSync} from "node:child_process";
import {readFileSync} from "node:fs";
import {fileURLToPath} from "node:url";
import {JSDOM, VirtualConsole} from "jsdom";

const root = fileURLToPath(new URL("../", import.meta.url));
const {fixture, promoted} = JSON.parse(execFileSync(process.env.WORLD_MODEL_PYTHON || `${root}.venv/bin/python`, ["-c", "import json; from tests.learning_fixtures import learning_fixture, promoted_inspections; print(json.dumps({'fixture': learning_fixture()[0], 'promoted': [s.model_dump(mode='json') for s in promoted_inspections()]}))"], {cwd: root, encoding:"utf8"}));
const errors = [];
const virtualConsole = new VirtualConsole();
virtualConsole.on("jsdomError", (e) => errors.push(e.message));
const dom = new JSDOM(readFileSync(`${root}world_model/api/static/learning.html`, "utf8"), {url:"http://127.0.0.1:8000/learning", runScripts:"outside-only", virtualConsole});
const w = dom.window, $ = (id) => w.document.getElementById(id);
let empty = true, corrupt = false, loseCreation = true;
const creations = [];
w.fetch = async (path, options) => {
  if (path === "/research") return {ok:true, json:async () => empty ? [] : [{name:fixture.name,split:"development",available:true}]};
  if (path === `/research/${fixture.name}/runs`) {
    assert.equal(options.method, "POST");
    assert.equal(options.headers["Content-Type"], "application/json");
    creations.push(JSON.parse(options.body));
    if (loseCreation) { loseCreation = false; throw new Error("Connection lost after creation"); }
    return {ok:true, json:async () => ({created:false,publication:{belief:{run_id:creations.at(-1).run_id}}})};
  }
  assert.equal(path, `/research/${fixture.name}`);
  return {ok:!corrupt, json:async () => corrupt ? {detail:"research archive checksum mismatch"} : structuredClone(fixture)};
};
const wait = async (condition) => {
  for (let i=0;i<100;i++) { if (condition()) return; await new Promise((r) => setTimeout(r,10)); }
  assert.fail(`UI condition timed out: ${$("lab-message").textContent}`);
};
w.eval(readFileSync(`${root}world_model/api/static/learning.js`, "utf8"));
await wait(() => !$("lab-refresh").disabled);
assert.match($("lab-empty").textContent, /No completed calibration experiment/);
empty = false;
$("lab-refresh").click();
await wait(() => !$("lab-content").hidden && !$("lab-refresh").disabled);
assert.equal($("parameter-table").children.length, 9);
assert.equal($("label-count").textContent, "8 readings / cell");
assert.match($("score-table").textContent, /0.3000/);
assert.match($("comparison-table").textContent, /-0.150000 to -0.050000/);
assert.match($("gates").textContent, /NOT EVALUATED/);
assert.match($("lab-costs").textContent, /72 credits/);
$("calibration-revision").value = "0";
$("calibration-revision").dispatchEvent(new w.Event("change"));
assert.equal($("label-count").textContent, "0 readings / cell");
assert.equal($("calibration-evidence").textContent, "[]");
assert.match($("parameter-table").textContent, /50.0%/);
assert.match($("parameter-table").textContent, /2.5%–97.5%/);
assert.match($("benchmark-scope").textContent, /final calibration revision \(8 readings per cell\)/);
assert.match($("promotion-scope").textContent, /Selected snapshot: 0 readings/);
$("calibrated-run-form").dispatchEvent(new w.Event("submit", {cancelable:true}));
await wait(() => !$("lab-refresh").disabled);
assert.equal($("lab-content").hidden, false);
assert.match($("lab-message").textContent, /Connection lost/);
assert.equal(creations[0].calibration_revision, fixture.revisions[0].revision_id);
const retryId = $("calibrated-run-id").value;
$("lab-refresh").click();
await wait(() => !$("lab-refresh").disabled);
assert.equal($("calibrated-run-id").value, retryId);
$("calibrated-run-form").dispatchEvent(new w.Event("submit", {cancelable:true}));
await wait(() => !$("promotion-result").hidden && !$("lab-refresh").disabled);
assert.deepEqual(creations[0], creations[1]);
assert.equal($("promoted-run-link").getAttribute("href"), `/?run=${encodeURIComponent(retryId)}`);
assert.match($("lab-message").textContent, /already saved/);
$("calibration-revision").value = "1";
$("calibration-revision").dispatchEvent(new w.Event("change"));
assert.notEqual($("calibrated-run-id").value, retryId);
assert.equal($("promotion-result").hidden, true);
corrupt = true;
$("lab-refresh").click();
await wait(() => !$("lab-refresh").disabled);
assert.equal($("lab-content").hidden, true);
assert.match($("lab-message").textContent, /checksum mismatch/);
corrupt = false;
$("lab-refresh").click();
await wait(() => !$("lab-content").hidden && !$("lab-refresh").disabled);
assert.deepEqual(errors, []);
dom.window.close();
const explorer = new JSDOM(readFileSync(`${root}world_model/api/static/index.html`, "utf8"), {url:"http://127.0.0.1:8000/?run=learned-run", runScripts:"outside-only", virtualConsole});
const e = explorer.window, element = (id) => e.document.getElementById(id);
let sequence = 0;
e.fetch = async (path, options) => {
  const url = new URL(path, e.location.href);
  let body;
  if (path === "/models") body = [promoted[0].model.registration];
  else if (path === "/runs") body = []; // The requested run may be outside the catalog page.
  else if (url.pathname === "/runs/learned-run/inspection") body = promoted[url.searchParams.has("sequence_cutoff") ? Number(url.searchParams.get("sequence_cutoff")) : sequence];
  else if (path === "/runs/learned-run/observations") {
    const reading = JSON.parse(options.body);
    assert.equal(reading.source_id, "probe_a");
    assert.equal(reading.value, true);
    sequence = 1;
    body = {created:true, publication:promoted[1].publication};
  } else if (path === "/runs/learned-run/replay") body = {events_checked:2, publication:promoted[1].publication, calibration_revision_checked:fixture.revisions[1].revision_id};
  else assert.fail(`Unexpected promoted-run request ${path}`);
  return {ok:true,json:async () => structuredClone(body)};
};
const settled = async () => {
  for (let i=0;i<100;i++) { if (!element("refresh").disabled && !element("workspace").hidden) return; await new Promise((r) => setTimeout(r,10)); }
  assert.fail(`Promoted explorer failed: ${element("message").textContent}`);
};
e.eval(readFileSync(`${root}world_model/api/static/explorer.js`, "utf8"));
await settled();
assert.equal(element("run-select").value, "learned-run");
assert.match(element("model-calibration").textContent, /72 labeled sensor readings/);
assert.equal(element("model-calibration").querySelector("a").getAttribute("href"), `/models/${promoted[0].run.model_hash}/calibration`);
assert.equal(element("evidence-count").textContent, "0");
element("observation-form").dispatchEvent(new e.Event("submit", {cancelable:true}));
await settled();
assert.equal(element("evidence-count").textContent, "1");
assert.match(element("snapshot-meta").textContent, /Revision #1/);
element("verify").click();
await settled();
assert.match(element("message").textContent, /Calibration evidence and derived sensor likelihoods also match/);
assert.deepEqual(errors, []);
explorer.window.close();
console.log("Calibration Lab DOM checks passed: parameter intervals, snapshot selection, gates, integrity, promotion, lost-response retry across refresh, and saved-run links.");
console.log("Promoted-run explorer checks passed: direct run link, learned provenance, three-state observation update, and calibration replay result.");

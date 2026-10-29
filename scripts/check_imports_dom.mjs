import assert from "node:assert/strict";
import {execFileSync} from "node:child_process";
import {readFileSync} from "node:fs";
import {fileURLToPath} from "node:url";
import {JSDOM, VirtualConsole} from "jsdom";

const root = fileURLToPath(new URL("../", import.meta.url));
const fixture = JSON.parse(execFileSync(process.env.WORLD_MODEL_PYTHON || `${root}.venv/bin/python`, ["-c", "import json; from tests.import_fixtures import import_fixture; m,c,p,e,s=import_fixture(); print(json.dumps({'model':s[0].model.model_dump(mode='json'), 'source':c.decode(), 'plan':p.model_dump(mode='json'), 'hash':p.content_hash, 'snapshots':[v.model_dump(mode='json') for v in s]}))"], {cwd:root,encoding:"utf8"}));
const errors = [], vc = new VirtualConsole(); vc.on("jsdomError", (error) => errors.push(error.message));
const dom = new JSDOM(readFileSync(`${root}world_model/api/static/imports.html`,"utf8"), {url:"http://127.0.0.1:8000/imports",runScripts:"outside-only",virtualConsole:vc});
const w = dom.window, $ = (id) => w.document.getElementById(id);
const reply = (data, status=200) => ({ok:status<400,json:async () => structuredClone(data)});
let lose = true, invalid = false;
const requests = [];
w.fetch = async (path, options={}) => {
  if (path === "/models") return reply([fixture.model.registration]);
  if (path.startsWith("/models/")) return reply(fixture.model);
  if (path === "/imports/github-actions/example") return reply({model_hash:fixture.model.registration.content_hash,export_json:fixture.source,config:fixture.plan.config,notice:"Illustrative fixture, not real repository data."});
  const body = JSON.parse(options.body);
  assert.equal(options.headers["Content-Type"], "application/json");
  if (path === "/imports/github-actions/preview") {
    assert.equal(body.export_json, fixture.source);
    assert.deepEqual(body.config.workflow_sources, fixture.plan.config.workflow_sources);
    return invalid ? reply({detail:"incomplete export"},422) : reply({plan:fixture.plan,plan_hash:fixture.hash});
  }
  assert.equal(path, "/imports/github-actions/runs");
  assert.equal(body.expected_plan_hash, fixture.hash);
  requests.push(body);
  if (lose) { lose=false; throw new Error("Response lost after commit"); }
  return reply({created:false,publication:fixture.snapshots.at(-1).publication});
};
const wait = async (check, describe) => {
  for (let i=0;i<200;i++) { if (check()) return; await new Promise((r)=>setTimeout(r,10)); }
  assert.fail(describe());
};
const ready = () => wait(()=>!$("example").disabled, ()=>$("import-message").textContent);
const submit = (id) => $(id).dispatchEvent(new w.Event("submit",{cancelable:true}));
w.eval(readFileSync(`${root}world_model/api/static/imports.js`,"utf8"));
await ready();
assert.equal($("save-fields").disabled,true);
$("example").click(); await ready();
assert.match($("file-name").textContent,/Illustrative/);
assert.match($("model-assumption").textContent,/supplied assumptions/);
submit("preview-form"); await ready();
assert.equal($("preview-readings").children.length,9);
assert.match($("preview-summary").textContent,/12 unique source runs/);
assert.equal($("save-fields").disabled,false);
submit("import-form"); await ready();
assert.match($("import-message").textContent,/Response lost/);
assert.equal($("save-fields").disabled,false);
submit("import-form"); await ready();
assert.deepEqual(requests[0],requests[1]);
assert.equal($("saved-link").getAttribute("href"),"/?run=imported-run");
$("branch").value="main"; $("branch").dispatchEvent(new w.Event("input",{bubbles:true}));
assert.equal($("save-fields").disabled,true);
assert.equal($("preview-content").hidden,true);
invalid=true; submit("preview-form"); await ready();
assert.match($("import-message").textContent,/incomplete export/);
assert.equal($("save-fields").disabled,true);
invalid=false; submit("preview-form"); await ready();
assert.equal($("save-fields").disabled,false);
dom.window.close();

const explorer = new JSDOM(readFileSync(`${root}world_model/api/static/index.html`,"utf8"),{url:"http://127.0.0.1:8000/?run=imported-run",runScripts:"outside-only",virtualConsole:vc});
const e=explorer.window, el=(id)=>e.document.getElementById(id);
e.fetch = async (path) => {
  if(path==="/models") return reply([fixture.model.registration]);
  if(path==="/runs") return reply([fixture.snapshots[0].run]);
  const url=new URL(path,e.location.href);
  const sequence=url.searchParams.has("sequence_cutoff") ? Number(url.searchParams.get("sequence_cutoff")) : 15;
  if(url.pathname.endsWith("/inspection")) return reply(fixture.snapshots[sequence]);
  if(url.pathname.endsWith("/replay")) return reply({events_checked:sequence+1,publication:fixture.snapshots[sequence].publication,imported_observations_checked:fixture.snapshots[sequence].observations.length});
  assert.fail(`Unexpected request ${path}`);
};
const settled=()=>wait(()=>!el("refresh").disabled&&!el("workspace").hidden,()=>el("message").textContent);
e.eval(readFileSync(`${root}world_model/api/static/explorer.js`,"utf8")); await settled();
assert.equal(el("observation-fields").disabled,true);
assert.equal(el("advance").disabled,true);
assert.equal(el("agent-config-fields").disabled,true);
assert.equal(el("import-source-list").children.length,9);
assert.equal(el("import-source-download").getAttribute("href"),"/runs/imported-run/import/source");
el("verify").click(); await settled();
assert.match(el("message").textContent,/Verified 9 imported observations/);
el("revision-select").value="1"; el("revision-select").dispatchEvent(new e.Event("change")); await settled();
assert.equal(el("import-source-list").children.length,1);
assert.match(el("import-source-list").textContent,/Run #1002/);
assert.doesNotMatch(el("import-source-list").textContent,/Run #1001/);
el("revision-select").value="0"; el("revision-select").dispatchEvent(new e.Event("change")); await settled();
assert.match(el("import-source-list").textContent,/No imported observations/);
assert.equal(el("agent-config-fields").disabled,true);
assert.deepEqual(errors,[]);
explorer.window.close();
console.log("Import DOM checks passed: preview, assumptions, stale inputs, lost-response retry, source links, replay, read-only imported runs, and historical isolation.");

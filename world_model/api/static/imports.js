"use strict";
const $ = (id) => document.getElementById(id);
const esc = (s) => String(s).replace(/[&<>"']/g, (c) => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));
let busy = false, model = null, exportJSON = null, preview = null, request = null;
async function api(path, body) {
  const response = await fetch(path, body === undefined ? {cache:"no-store"} : {method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(body)});
  const data = await response.json();
  if (!response.ok) throw new Error(Array.isArray(data.detail) ? data.detail.map((item) => item.msg).join("; ") : data.detail || "Request failed.");
  return data;
}
function controls() {
  $("example").disabled = busy || !$("import-model").options.length;
  $("import-fields").disabled = busy || !model;
  $("save-fields").disabled = busy || !preview;
}
function invalidate() {
  preview = request = null;
  $("preview-content").hidden = true;
  $("preview-empty").hidden = false;
  $("saved-result").hidden = true;
  controls();
}
async function work(action) {
  if (busy) return;
  busy = true; controls(); $("import-message").hidden = true;
  try { await action(); }
  catch (error) { $("import-message").textContent = error.message; $("import-message").hidden = false; }
  finally { busy = false; controls(); }
}
async function loadModel() {
  model = null; invalidate();
  model = await api(`/models/${$("import-model").value}`);
  $("workflow-mapping").innerHTML = Object.keys(model.spec.observation_models).map((source) => `<label>${esc(source)} · workflow ID<input data-source="${esc(source)}" type="number" min="1" max="9007199254740991" step="1"></label>`).join("");
  $("import-target").innerHTML = model.spec.states.map((s) => `<option value="${esc(s)}">${esc(s)}</option>`).join("");
  $("import-target").value = model.spec.states.at(-1);
  $("window-description").textContent = `Each window spans ${model.spec.tick_duration_seconds.toLocaleString()} seconds. A reading is timestamped at its window's end.`;
  $("model-assumption").textContent = model.spec.derivation ? "This model uses calibrated posterior means. Its calibration data must match this workflow/window definition; predictive validity on this repository still needs evaluation." : "This model's probabilities are supplied assumptions. Importing workflow results does not establish calibrated health probabilities or causal relationships.";
}
function body() {
  if (exportJSON === null) throw new Error("Choose a JSON export or load the illustrative example.");
  const workflow_sources = {};
  document.querySelectorAll("[data-source]").forEach((input) => { if (input.value !== "") workflow_sources[input.dataset.source] = Number(input.value); });
  return {model_hash:$("import-model").value, export_json:exportJSON, config:{repository:$("repository").value,branch:$("branch").value,event:$("trigger-event").value,
    window_start:new Date($("window-start").value + "Z").toISOString(),windows:Number($("windows").value),workflow_sources}};
}
$("import-model").addEventListener("change", () => work(loadModel));
$("preview-form").addEventListener("input", invalidate);
$("export-file").addEventListener("change", () => work(async () => {
  exportJSON = null; invalidate();
  const file = $("export-file").files[0];
  if (!file) return;
  if (file.size > 8 * 1024 * 1024) throw new Error("Export must be no larger than 8 MiB.");
  exportJSON = await file.text(); $("file-name").textContent = file.name;
}));
$("example").addEventListener("click", () => work(async () => {
  const example = await api("/imports/github-actions/example");
  if (![...$("import-model").options].some((o) => o.value === example.model_hash)) throw new Error("Restart scripts/run_explorer.py to register the software-delivery example model.");
  $("import-model").value = example.model_hash; await loadModel();
  exportJSON = example.export_json; $("export-file").value = "";
  $("file-name").textContent = example.notice;
  $("repository").value = example.config.repository; $("branch").value = example.config.branch;
  $("trigger-event").value = example.config.event;
  $("window-start").value = example.config.window_start.replace("Z", "").replace("+00:00", "");
  $("windows").value = example.config.windows;
  document.querySelectorAll("[data-source]").forEach((input) => { input.value = example.config.workflow_sources[input.dataset.source] || ""; });
}));
$("preview-form").addEventListener("submit", (event) => {
  event.preventDefault(); work(async () => {
    invalidate(); const input = body();
    const result = await api("/imports/github-actions/preview", input);
    preview = {...input, expected_plan_hash:result.plan_hash};
    const plan = result.plan;
    $("preview-summary").textContent = `${plan.unique_run_count} unique source runs → ${plan.readings.length} observations across ${plan.config.windows} windows. ${plan.duplicate_count} exact duplicate records counted once.`;
    $("preview-readings").innerHTML = plan.readings.map((r) => `<tr><td>${esc(r.window_end)}</td><td>${esc(r.payload.source_id)}</td><td>${r.payload.value ? "Failure observed" : "All observed runs succeeded"}</td><td>${r.runs.map((run) => `<a href="${esc(run.url)}" target="_blank" rel="noreferrer">#${run.run_id}</a>`).join("")}</td></tr>`).join("");
    $("preview-diagnostics").textContent = `${plan.skipped.length} records excluded; ${plan.incomplete_windows.length} windows include inconclusive results. Missing windows contain no sensor reading.`;
    $("preview-json").textContent = JSON.stringify(result, null, 2);
    $("preview-empty").hidden = true; $("preview-content").hidden = false;
  });
});
$("import-form").addEventListener("input", () => { request = null; $("saved-result").hidden = true; });
$("import-form").addEventListener("submit", (event) => {
  event.preventDefault(); if (!preview) return;
  work(async () => {
    if (!request) request = {...preview,run_id:$("import-run-id").value,target_state:$("import-target").value};
    const result = await api("/imports/github-actions/runs", request);
    $("saved-link").href = `/?run=${encodeURIComponent(result.publication.belief.run_id)}`;
    $("saved-result").hidden = false;
    $("import-message").textContent = result.created ? "Source archive, observations, and belief history saved. Open the run to inspect evidence and verify replay." : "The import was already saved. Your original receipt has been recovered.";
    $("import-message").hidden = false;
  });
});
work(async () => {
  const models = await api("/models");
  $("import-model").innerHTML = models.map((m) => `<option value="${esc(m.content_hash)}">${esc(m.model_id)} / ${esc(m.model_version)}</option>`).join("");
  if (!models.length) throw new Error("Start scripts/run_explorer.py to register the example models.");
  const example = models.find((m) => m.model_id === "software-delivery-assumed");
  if (example) $("import-model").value = example.content_hash;
  $("import-run-id").value = `import-${crypto.randomUUID()}`;
  $("window-start").value = new Date(Date.now() - 8 * 86400000).toISOString().slice(0, 10) + "T00:00:00";
  await loadModel();
});

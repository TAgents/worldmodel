"use strict";
const byId = (id) => document.getElementById(id);
const esc = (value) => String(value).replace(/[&<>"']/g, (c) => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));
const pct = (p) => `${(p * 100).toFixed(1)}%`;
const num = (v) => v === null || v === undefined ? "—" : Number(v).toFixed(4);
const difference = (v) => v === null || v === undefined ? "—" : Number(v).toFixed(6);
let current = null;
let loading = false;
let pendingRun = null;
let runContext = null;
const names = {learned_adaptive:"Learned · adaptive", learned_random:"Learned · random", mean_likelihood_matched:"Posterior means · matched readings", neutral_likelihood_matched:"Uncalibrated · matched readings", oracle_likelihood_matched:"Known likelihoods · evaluator reference"};

async function get(path, body) {
  const response = await fetch(path, body === undefined ? {cache:"no-store"} : {method:"POST", headers:{"Content-Type":"application/json"}, body:JSON.stringify(body)});
  const result = await response.json();
  if (!response.ok) throw new Error(Array.isArray(result.detail) ? result.detail.map((item) => item.msg).join("; ") : result.detail || "Could not complete the request.");
  return result;
}
function enabled() {
  byId("experiment").disabled = loading || !byId("experiment").value;
  byId("calibration-revision").disabled = loading || !current;
  byId("lab-refresh").disabled = loading;
  byId("calibrated-run-fields").disabled = loading || !current;
}
async function work(action, preserveExperiment = false) {
  if (loading) return;
  loading = true;
  byId("lab-message").hidden = true;
  enabled();
  try { await action(); }
  catch (error) {
    if (!preserveExperiment) {
      current = null;
      byId("lab-content").hidden = true;
      byId("lab-empty").hidden = false;
      byId("lab-empty").querySelector("h2").textContent = "Experiment unavailable";
      byId("lab-empty").querySelector("p").textContent = "Refresh to retry. Incomplete or inconsistent archives are not displayed.";
    }
    byId("lab-message").textContent = error.message;
    byId("lab-message").hidden = false;
  } finally { loading = false; enabled(); }
}
function renderRevision() {
  const revision = current.revisions[Number(byId("calibration-revision").value)];
  const curve = current.report.learning_curve.find((c) => c.revision_id === revision.revision_id);
  const references = new Map(curve.parameters.map((c) => [`${c.source_id}:${c.state}`, c.synthetic_truth]));
  byId("calibration-meta").textContent = `Available by ${revision.as_of} · ${revision.evidence.length} labeled sensor readings`;
  byId("label-count").textContent = `${curve.samples_per_sensor_state} readings / cell`;
  byId("parameter-table").innerHTML = revision.cells.map((cell) => `<tr><td><strong>${esc(cell.source_id)}</strong><br><span class="mono">${esc(cell.state)}</span></td><td>${cell.alerts} / ${cell.alerts + cell.non_alerts}</td><td>${pct(cell.mean)}</td><td>${pct(cell.lower_95)}–${pct(cell.upper_95)}<svg class="interval-bar" viewBox="0 0 100 20" role="img" aria-label="95 percent interval from ${pct(cell.lower_95)} to ${pct(cell.upper_95)}"><line class="axis" x1="2" x2="98" y1="10" y2="10"/><line class="range" x1="${2 + cell.lower_95 * 96}" x2="${2 + cell.upper_95 * 96}" y1="10" y2="10"/><circle class="mean" cx="${2 + cell.mean * 96}" cy="10" r="4"/></svg></td><td>${pct(references.get(`${cell.source_id}:${cell.state}`))}</td></tr>`).join("");
  byId("lab-provenance").innerHTML = Object.entries({"Calibration revision":revision.revision_id,"Structure SHA-256":revision.structure_hash,"Prior":`Beta(${revision.prior.alpha}, ${revision.prior.beta}) per sensor/state`,"Algorithm":revision.algorithm,"Inference":current.report.inference_version,"Evidence cutoff":revision.as_of,"Runtime":JSON.stringify(current.runtime)}).map(([k,v]) => `<dt>${esc(k)}</dt><dd>${esc(v)}</dd>`).join("");
  byId("calibration-evidence").textContent = JSON.stringify(revision.evidence, null, 2);
  byId("promotion-scope").textContent = `Selected snapshot: ${curve.samples_per_sensor_state} readings per sensor/state, available by ${revision.as_of}.`;
}
function resetRun() {
  pendingRun = null;
  runContext = `${current.name}:${current.revisions[Number(byId("calibration-revision").value)].revision_id}`;
  byId("calibrated-run-id").value = `cal-${crypto.randomUUID()}`;
  const epoch = new Date(Date.now() - 86400000);
  byId("calibrated-run-epoch").value = new Date(epoch.getTime() - epoch.getTimezoneOffset() * 60000).toISOString().slice(0, 19);
  byId("promotion-result").hidden = true;
}
async function loadExperiment() {
  current = await get(`/research/${encodeURIComponent(byId("experiment").value)}`);
  const report = current.report;
  const previousTarget = byId("calibrated-run-target").value;
  byId("calibrated-run-target").innerHTML = report.states.map((state) => `<option value="${esc(state)}">${esc(state)}</option>`).join("");
  if (report.states.includes(previousTarget)) byId("calibrated-run-target").value = previousTarget;
  byId("calibration-revision").innerHTML = current.revisions.map((r,i) => `<option value="${i}">${report.learning_curve[i].samples_per_sensor_state} readings per sensor/state</option>`).join("");
  const previousIndex = current.revisions.findIndex((r) => `${current.name}:${r.revision_id}` === runContext);
  byId("calibration-revision").value = String(previousIndex < 0 ? current.revisions.length - 1 : previousIndex);
  if (previousIndex < 0) resetRun();
  byId("split-badge").textContent = `${report.split.toUpperCase()} · ${report.episode_count} EPISODES`;
  byId("curve-table").innerHTML = report.learning_curve.map((c) => `<tr><td>${c.samples_per_sensor_state}</td><td>${num(c.rmse)}</td><td>${pct(c.mean_interval_width)}</td><td>${num(c.validation.mean_log_loss)}</td></tr>`).join("");
  byId("benchmark-scope").textContent = `Evaluation uses the final calibration revision (${current.protocol.calibration_samples_per_state} readings per cell), with ${report.origin_ticks_per_episode} forecasts and a ${report.acquisition_budget}-credit acquisition cap per episode.`;
  byId("score-table").innerHTML = Object.entries(report.methods).map(([name, m]) => `<tr><td>${esc(names[name] || name.replaceAll("_", " "))}</td><td>${num(m.mean_brier)}</td><td>${num(m.mean_log_loss)}</td><td>${Math.min(...m.credits_per_episode)}–${Math.max(...m.credits_per_episode)}</td><td>${m.minimum_effective_particles === null ? "—" : `${m.minimum_effective_particles.toFixed(1)} / ${report.particle_count}`}</td></tr>`).join("");
  byId("comparison-table").innerHTML = Object.entries(report.adaptive_minus_baseline).map(([name, comparison]) => `<tr><td>${esc(names[name] || name.replaceAll("_", " "))}</td><td>${difference(comparison.brier?.mean_difference)}</td><td>${difference(comparison.brier?.lower_95)} to ${difference(comparison.brier?.upper_95)}</td></tr>`).join("");
  byId("gates").innerHTML = [["Sensor parameter recovery",report.parameter_recovery_gate_passed],["Particle weight check",report.numerical_ess_gate_passed],["Holdout acquisition improvement",report.heldout_acquisition_gate_passed]].map(([label, passed]) => `<div class="gate ${passed === null ? "pending" : passed ? "" : "failed"}"><span>${esc(label)}</span><strong>${passed === null ? "NOT EVALUATED" : passed ? "PASSED" : "NOT MET"}</strong></div>`).join("");
  const cost = report.calibration_cost;
  byId("lab-costs").innerHTML = [["Shared calibration readings",`${cost.sensor_readings} · ${cost.sensor_credits} credits`],["Verified calibration labels",`${cost.verified_labels} · ${cost.label_credits} credits`],["Per-episode acquisition cap",`${report.acquisition_budget} credits`]].map(([k,v]) => `<dt>${esc(k)}</dt><dd>${esc(v)}</dd>`).join("");
  byId("protocol-json").textContent = JSON.stringify(current.protocol, null, 2);
  renderRevision();
  byId("lab-empty").hidden = true;
  byId("lab-content").hidden = false;
}
async function loadCatalog() {
  const selected = byId("experiment").value;
  const entries = await get("/research");
  const valid = entries.filter((item) => item.available);
  byId("experiment").innerHTML = valid.length ? valid.map((item) => `<option value="${esc(item.name)}">${esc(item.name)} · ${esc(item.split)}</option>`).join("") : '<option value="">No completed experiments</option>';
  if (entries.some((item) => !item.available)) {
    byId("lab-message").textContent = "One or more archives failed verification. Their results have been excluded.";
    byId("lab-message").hidden = false;
  }
  if (valid.length) {
    if (valid.some((item) => item.name === selected)) byId("experiment").value = selected;
    await loadExperiment();
  } else {
    current = null;
    byId("lab-content").hidden = true;
    byId("lab-empty").hidden = false;
    byId("lab-empty").querySelector("h2").textContent = "No completed calibration experiment yet.";
    byId("lab-empty").querySelector("p").textContent = "Run .venv/bin/python -m world_model.learning evaluate --output artifacts/reliability-development, then refresh.";
  }
}
byId("experiment").addEventListener("change", () => work(loadExperiment));
byId("calibration-revision").addEventListener("change", () => { renderRevision(); resetRun(); });
byId("lab-refresh").addEventListener("click", () => work(loadCatalog));
byId("calibrated-run-form").addEventListener("input", () => { pendingRun = null; byId("promotion-result").hidden = true; });
byId("calibrated-run-form").addEventListener("submit", (event) => {
  event.preventDefault();
  if (!current) return;
  work(async () => {
    if (!pendingRun) pendingRun = {path:`/research/${encodeURIComponent(current.name)}/runs`, body:{
      run_id:byId("calibrated-run-id").value, epoch:new Date(byId("calibrated-run-epoch").value).toISOString(),
      target_state:byId("calibrated-run-target").value,
      calibration_revision:current.revisions[Number(byId("calibration-revision").value)].revision_id}};
    const receipt = await get(pendingRun.path, pendingRun.body);
    byId("promoted-run-link").href = `/?run=${encodeURIComponent(receipt.publication.belief.run_id)}`;
    byId("promotion-result").hidden = false;
    byId("lab-message").textContent = receipt.created ? "Calibration and new run saved together. Open the run to add evidence and verify replay." : "This run was already saved. Your original creation receipt has been recovered.";
    byId("lab-message").hidden = false;
  }, true);
});
work(loadCatalog);

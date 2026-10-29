"use strict";

const $ = (id) => document.getElementById(id);
const escapeHTML = (value) => String(value).replace(/[&<>"']/g, (c) => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));
const percent = (value) => `${(value * 100).toFixed(2)}%`;
const utc = (value) => value ? new Date(value).toISOString().replace("T", " ").replace(".000Z", "Z") : "—";
const option = (value, label) => `<option value="${escapeHTML(value)}">${escapeHTML(label)}</option>`;
const kindLabel = {run_created: "Initial prior", tick_advanced: "Time advanced", observation_ingested: "Evidence received"};
let inspection = null;
let busy = false;
let pendingReading = null;
let dialogGeneration = 0;
let autoRunning = false;
let autoGeneration = 0;
let pendingStep = null;
let catalogLoaded = false;

async function api(path, body) {
  const response = await fetch(path, body === undefined ? {cache: "no-store"} : {
    method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify(body),
  });
  const result = await response.json();
  if (!response.ok) {
    const detail = Array.isArray(result.detail) ? result.detail.map((item) => `${item.loc.join(".")}: ${item.msg}`).join("; ") : result.detail;
    throw new Error(detail || `Request failed (${response.status})`);
  }
  return result;
}

function message(text, error = false) {
  $("message").textContent = text;
  $("message").className = error ? "error" : "";
  $("message").hidden = !text;
}

function controls() {
  const historic = $("revision-select").value !== "";
  const locked = busy || autoRunning;
  const ready = !!inspection && !locked;
  const agent = inspection?.investigation;
  const imported = inspection?.imported;
  $("run-select").disabled = locked || !$("run-select").value;
  $("revision-select").disabled = !ready;
  $("refresh").disabled = locked;
  $("new-run").disabled = locked || !$("create-model").options.length;
  $("return-live").disabled = locked;
  $("verify").disabled = !ready;
  $("observation-fields").disabled = !ready || historic || !!agent || !!imported;
  $("advance").disabled = !ready || historic || !!agent || !!imported;
  document.querySelectorAll("[data-sequence]").forEach((button) => { button.disabled = locked; });
  $("historical-note").hidden = !historic && !agent && !imported;
  $("historical-note").textContent = imported ? "This run preserves an imported source snapshot. Use Import evidence to create a new run for revised exports." : historic ? "Return to the latest snapshot to make changes." : "This run receives readings and clock advances through its synthetic investigation.";
  $("agent-config-fields").disabled = !ready || historic || !!agent || !!imported || inspection?.publication.belief.sequence_cutoff !== 0;
  $("agent-step").disabled = !ready || historic || !agent || agent.stopped;
  $("agent-run").disabled = !ready || historic || !agent || agent.stopped;
  $("agent-pause").disabled = !autoRunning;
  $("agent-status").textContent = agent?.stopped ? "COMPLETED" : autoRunning ? "RUNNING" : "READY";
}

async function work(action, {clearMessage = true} = {}) {
  if (busy) return;
  busy = true;
  if (clearMessage) message("");
  controls();
  try { await action(); } catch (error) { autoRunning = false; message(error.message, true); }
  finally { busy = false; controls(); }
}

function rowChange(data) {
  const belief = data.publication.belief;
  const target = data.publication.forecast.target_state;
  const previous = data.history.at(-2)?.belief;
  if (!previous) return "Initial prior. No observations have been conditioned on yet.";
  const difference = (belief.distribution[target] - previous.distribution[target]) * 100;
  const trigger = data.history.at(-1).event_kind;
  let reason = trigger === "tick_advanced" ? "Applying the state transition" : "Conditioning on the revised evidence snapshot";
  const added = data.observations.find((item) => item.event_sequence === belief.sequence_cutoff);
  if (added) reason = `${added.payload.supersedes_id ? "Correcting" : "Adding"} the ${added.payload.source_id} reading at tick ${added.payload.tick}`;
  return `${reason} changed P(${target}) by ${difference >= 0 ? "+" : ""}${difference.toFixed(2)} percentage points from revision #${previous.sequence_cutoff}.`;
}

function renderChart(data) {
  const target = data.publication.forecast.target_state;
  const items = data.history;
  const x = (index) => items.length === 1 ? 330 : 40 + index / (items.length - 1) * 560;
  const y = (value) => 120 - value * 100;
  const points = items.map((item, i) => `${x(i)},${y(item.belief.distribution[target])}`).join(" ");
  $("chart").innerHTML = `<svg viewBox="0 0 640 145" role="img" aria-label="${escapeHTML(`P(${target}) across ${items.length} revisions`)}"><title>${escapeHTML(`Posterior probability of ${target}; exact values are in the history table`)}</title>${[0, .5, 1].map((p) => `<line class="grid" x1="40" x2="620" y1="${y(p)}" y2="${y(p)}"/><text x="0" y="${y(p) + 3}">${p * 100}%</text>`).join("")}<polyline class="line" points="${points}"/>${items.map((item, i) => `<circle cx="${x(i)}" cy="${y(item.belief.distribution[target])}" r="3"><title>#${item.belief.sequence_cutoff}: ${percent(item.belief.distribution[target])}</title></circle>`).join("")}<text x="40" y="142">PUBLICATION SEQUENCE</text><text x="580" y="142">#${items.at(-1).belief.sequence_cutoff}</text></svg>`;
}

function render(data) {
  inspection = data;
  const {belief, forecast, evidence} = data.publication;
  const model = data.model.spec;
  const historic = $("revision-select").value !== "";
  $("empty").hidden = true;
  $("workspace").hidden = false;
  $("view-badge").textContent = historic ? "HISTORICAL SNAPSHOT" : "LATEST SNAPSHOT";
  $("view-badge").className = historic ? "badge historical" : "badge";
  $("return-live").hidden = !historic;
  $("snapshot-meta").textContent = `Revision #${belief.sequence_cutoff} · Published ${utc(belief.published_at)}`;
  $("state-tick").textContent = `TICK ${belief.state_tick}`;
  $("variable").textContent = belief.variable;
  $("distribution").innerHTML = model.states.map((state) => `<div class="distribution-row ${state === forecast.target_state ? "target" : ""}"><span class="state-label">${escapeHTML(state)}</span><progress max="1" value="${belief.distribution[state]}" aria-label="${escapeHTML(`Probability of ${state}`)}"></progress><strong>${percent(belief.distribution[state])}</strong></div>`).join("");
  $("belief-change").textContent = rowChange(data);
  $("forecast-title").textContent = `P(${forecast.target_state} at tick ${forecast.target_tick})`;
  $("forecast-detail").textContent = `Forecast from tick ${forecast.origin_tick} · ${model.tick_duration_seconds} seconds per tick`;
  $("forecast-probability").textContent = percent(forecast.probability);
  const selected = $("revision-select").value;
  $("revision-select").innerHTML = option("", "Latest available") + [...data.history].reverse().map(({belief: b}) => option(b.sequence_cutoff, `Revision #${b.sequence_cutoff} · tick ${b.state_tick}`)).join("");
  $("revision-select").value = selected;
  renderChart(data);
  $("history-probability-label").textContent = `P(${forecast.target_state})`;
  $("history-body").innerHTML = [...data.history].reverse().map(({belief: b, event_kind}) => `<tr class="${b.revision_id === belief.revision_id ? "selected-row" : ""}"><td><button data-sequence="${b.sequence_cutoff}" aria-label="Inspect revision ${b.sequence_cutoff}">#${b.sequence_cutoff} ↗</button></td><td>${kindLabel[event_kind]}</td><td>${b.state_tick}</td><td>${percent(b.distribution[forecast.target_state])}</td><td>${escapeHTML(utc(b.published_at))}</td></tr>`).join("");
  $("evidence-count").textContent = String(data.observations.length);
  $("evidence-list").innerHTML = data.observations.length ? data.observations.map((o) => `<article class="evidence-card"><div class="evidence-top"><strong>${escapeHTML(o.payload.source_id)} <span class="muted">· tick ${o.payload.tick}</span></strong><span class="reading ${o.payload.value ? "alert" : ""}">${o.payload.value ? "ALERT · TRUE" : "NO ALERT · FALSE"}</span></div><dl><dt>Observed (UTC)</dt><dd>${escapeHTML(utc(o.observed_at))}</dd><dt>Learned (UTC)</dt><dd>${escapeHTML(utc(o.ingested_at))} · revision #${o.event_sequence}</dd><dt>Observation ID</dt><dd class="mono">${escapeHTML(o.observation_id)}</dd><dt>Measurement ID</dt><dd class="mono">${escapeHTML(o.payload.measurement_id)}</dd>${o.payload.supersedes_id ? `<dt>Corrects</dt><dd class="mono">${escapeHTML(o.payload.supersedes_id)}</dd>` : ""}</dl></article>`).join("") : '<p class="muted">No evidence yet. This belief comes from the model’s prior and elapsed transitions.</p>';
  const lastByOrigin = new Map(data.forecasts.map((f) => [f.forecast.origin_tick, f.forecast.forecast_id]));
  $("forecasts-body").innerHTML = [...data.forecasts].reverse().map(({forecast: f, closed_at}) => `<tr><td>${f.origin_tick} → ${f.target_tick}</td><td>${percent(f.probability)}</td><td>${closed_at ? `Frozen at ${escapeHTML(utc(closed_at))}` : lastByOrigin.get(f.origin_tick) !== f.forecast_id ? "Replaced before tick closed" : "Issued · awaiting next tick"}</td><td>${escapeHTML(utc(f.issued_at))}</td></tr>`).join("");
  $("source").innerHTML = Object.keys(model.observation_models).map((source) => option(source, source)).join("");
  $("reading-tick").value = belief.state_tick;
  $("reading-tick").max = belief.state_tick;
  $("correction").innerHTML = option("", "New observation") + data.observations.map((o) => option(o.observation_id, `Correct ${o.payload.source_id} · tick ${o.payload.tick}`)).join("");
  correctionChanged();
  $("clock-note").textContent = `Next tick starts ${utc(new Date(new Date(data.run.epoch).getTime() + (belief.state_tick + 1) * model.tick_duration_seconds * 1000).toISOString())}.`;
  $("model-name").textContent = `${model.model_id} / ${model.model_version}`;
  $("model-tables").innerHTML = `<h3>Sensor likelihood · P(alert | state)</h3><div class="table-scroll"><table><thead><tr><th>Sensor</th>${model.states.map((state) => `<th>${escapeHTML(state)}</th>`).join("")}</tr></thead><tbody>${Object.entries(model.observation_models).map(([source, sensor]) => `<tr><td>${escapeHTML(source)}</td>${model.states.map((state) => `<td>${percent(sensor.alert_probability[state])}</td>`).join("")}</tr>`).join("")}</tbody></table></div><h3>Transition · current → next state</h3><div class="table-scroll"><table><thead><tr><th>From / to</th>${model.states.map((state) => `<th>${escapeHTML(state)}</th>`).join("")}</tr></thead><tbody>${model.states.map((state) => `<tr><td>${escapeHTML(state)}</td>${model.states.map((next) => `<td>${percent(model.transition_probabilities[state][next])}</td>`).join("")}</tr>`).join("")}</tbody></table></div><p class="footnote">Fixed parameters. Readings are conditionally independent given the hidden state.</p>`;
  $("provenance").innerHTML = Object.entries({"Model SHA-256": belief.model_hash, "Inference version": belief.inference_version, "Belief revision": belief.revision_id, "Previous revision": belief.previous_revision_id || "None (initial prior)", "Evidence snapshot": evidence.snapshot_id, "Sequence cutoff": belief.sequence_cutoff, "Forecast ID": forecast.forecast_id, "Environment epoch (UTC)": utc(data.run.epoch)}).map(([key, value]) => `<dt>${key}</dt><dd>${escapeHTML(value)}</dd>`).join("");
  $("model-json").textContent = JSON.stringify(model, null, 2);
  $("model-calibration").hidden = !model.derivation;
  $("model-calibration").innerHTML = model.derivation ? `<h3>Learned sensor calibration</h3><p>Fixed posterior means from ${model.derivation.evidence_count} labeled sensor readings, available by ${escapeHTML(utc(model.derivation.as_of))}.</p><dl><dt>Calibration revision</dt><dd>${escapeHTML(model.derivation.calibration_revision)}</dd><dt>Evidence SHA-256</dt><dd>${escapeHTML(model.derivation.calibration_content_hash)}</dd></dl><p><a href="/models/${encodeURIComponent(belief.model_hash)}/calibration">Inspect saved calibration evidence (JSON) →</a></p><p class="footnote">Replay verifies this derivation. State probabilities are conditional on the fixed learned means; parameter uncertainty is not propagated in this run.</p>` : "";
  pendingReading = null;
  renderImport(data);
  renderInvestigation(data);
}

function renderImport(data) {
  const imported = data.imported;
  $("import-provenance").hidden = !imported;
  if (!imported) { $("import-source-list").innerHTML = ""; return; }
  $("import-scope").textContent = `${imported.repository} · ${imported.branch} · ${imported.observations.length} mapped observations in this snapshot`;
  $("import-identity").innerHTML = Object.entries({"Adapter":imported.adapter_version,"Source SHA-256":imported.source_hash,"Translation SHA-256":imported.plan_hash,"Imported at":utc(imported.imported_at)}).map(([k,v]) => `<dt>${escapeHTML(k)}</dt><dd>${escapeHTML(v)}</dd>`).join("");
  $("import-source-download").href = `/runs/${encodeURIComponent(data.run.run_id)}/import/source`;
  $("import-source-list").innerHTML = imported.observations.map(({observation_id,reading}) => `<article class="evidence-card"><strong>${escapeHTML(reading.payload.source_id)} · tick ${reading.payload.tick}</strong><p class="footnote">Window ${escapeHTML(utc(reading.window_start))} → ${escapeHTML(utc(reading.window_end))}<br>Observation ${escapeHTML(observation_id)}</p>${reading.runs.map((r) => `<p class="footnote"><a href="${escapeHTML(r.url)}" target="_blank" rel="noreferrer">Run #${r.run_id}, attempt ${r.attempt}</a> · ${escapeHTML(r.conclusion || r.status)} · updated ${escapeHTML(utc(r.updated_at))}</p>`).join("")}</article>`).join("") || '<p class="muted">No imported observations had been published at this snapshot.</p>';
}

function renderInvestigation(data) {
  const view = data.investigation;
  const limits = data.model.spec.acquisition_limits;
  $("agent-config-form").hidden = !!view;
  $("agent-session").hidden = !view;
  $("agent-trace-panel").hidden = !view;
  $("agent-setup-note").textContent = view ? `Policy: ${view.config.policy}. Allowed sensors: ${view.config.allowed_sources.join(", ")}.` : data.publication.belief.sequence_cutoff === 0 ? "Enable before adding evidence. This dedicates the run to the synthetic adapter with immutable limits." : "Create a fresh run to enable autonomous investigation.";
  if (data.imported) $("agent-setup-note").textContent = "Imported runs retain their original source data. Synthetic investigations require a separate run.";
  if (!view) {
    $("agent-horizon").max = Math.min(200, limits.episode_origin_ticks);
    $("agent-horizon").value = Math.min(20, limits.episode_origin_ticks);
    $("agent-budget").max = limits.episode_budget_credits;
    $("agent-budget").value = Math.min(20, limits.episode_budget_credits);
    $("agent-max-reads").max = limits.max_reads_per_episode;
    $("agent-max-reads").value = limits.max_reads_per_episode;
    $("agent-sources").innerHTML = Object.keys(data.model.spec.observation_models).map((source) => `<label><input type="checkbox" value="${escapeHTML(source)}" checked> ${escapeHTML(source)}</label>`).join("");
    return;
  }
  $("agent-budget-summary").textContent = `${view.spent_credits} / ${view.config.budget_credits} credits spent`;
  $("agent-budget-progress").max = Math.max(1, view.config.budget_credits);
  $("agent-budget-progress").value = view.spent_credits;
  $("agent-read-summary").textContent = `${view.dispatched_reads} / ${view.config.max_reads} reads dispatched · ${view.config.horizon} origin ticks`;
  $("agent-trace").innerHTML = view.steps.length ? [...view.steps].reverse().map((step) => {
    const decision = step.decision;
    return `<article class="evidence-card"><div class="evidence-top"><strong>#${step.ordinal} · ${escapeHTML(decision.kind.replaceAll("_", " "))}${decision.source_id ? ` · ${escapeHTML(decision.source_id)}` : ""}</strong><span class="reading">${step.outcome?.charged_cost_credits || 0} credits</span></div><p class="muted">Tick ${decision.tick} · ${escapeHTML(step.outcome?.status || "stopped")} · revision #${step.before_sequence} → #${step.after_sequence}</p><p class="explanation">${escapeHTML(decision.reason)}</p><details><summary>Decision evidence and alternatives</summary><p class="footnote">Entropy before action: ${decision.entropy_bits.toFixed(4)} bits. Available credits: ${decision.available_credits} of ${decision.released_credits} released. Policy: ${escapeHTML(decision.policy_version)}.</p><div class="table-scroll"><table><thead><tr><th>Sensor</th><th>Gain (bits)</th><th>Gain / credit</th><th>Eligibility</th></tr></thead><tbody>${decision.candidates.map((c) => `<tr><td>${escapeHTML(c.source_id)}</td><td>${c.information_gain_bits.toFixed(4)}</td><td>${c.gain_bits_per_credit.toFixed(4)}</td><td>${escapeHTML(c.reason)}</td></tr>`).join("")}</tbody></table></div><p class="footnote mono">Request: ${escapeHTML(step.request_id)}<br>Observation: ${escapeHTML(step.outcome?.observation_id || "none")}<br>Recorded: ${escapeHTML(utc(step.recorded_at))}</p></details></article>`;
  }).join("") : '<p class="muted">No actions yet. Choose One step to inspect a decision, or Run episode to continue automatically within the configured limits.</p>';
}

async function loadInspection() {
  const run = $("run-select").value;
  if (!run) return;
  const sequence = $("revision-select").value;
  try {
    render(await api(`/runs/${encodeURIComponent(run)}/inspection${sequence === "" ? "" : `?sequence_cutoff=${sequence}`}`));
  } catch (error) {
    inspection = null;
    $("workspace").hidden = true;
    $("empty").hidden = false;
    $("empty").querySelector("h2").textContent = "Snapshot unavailable";
    $("empty").querySelector("p").textContent = "Refresh to retry loading the selected run. No stale snapshot is being displayed.";
    throw error;
  }
}

async function loadCatalog(preferredRun = catalogLoaded ? $("run-select").value : "") {
  let runs, models;
  try { [runs, models] = await Promise.all([api("/runs"), api("/models")]); }
  catch (error) {
    inspection = null;
    $("workspace").hidden = true;
    $("empty").hidden = false;
    $("empty").querySelector("h2").textContent = "Could not load the catalog";
    $("empty").querySelector("p").textContent = "Check the local database service, then refresh to try again.";
    throw error;
  }
  $("run-select").innerHTML = runs.length ? runs.map((run) => option(run.run_id, run.run_id)).join("") : option("", "No runs yet");
  catalogLoaded = true;
  $("run-select").title = "100 most recent runs";
  $("create-model").innerHTML = models.map((model) => option(model.content_hash, `${model.model_id} / ${model.model_version}`)).join("");
  if (preferredRun && !runs.some((run) => run.run_id === preferredRun)) {
    $("run-select").insertAdjacentHTML("beforeend", option(preferredRun, preferredRun));
  }
  if (preferredRun) $("run-select").value = preferredRun;
  else $("revision-select").value = "";
  if (runs.length || preferredRun) await loadInspection();
  else {
    inspection = null;
    $("workspace").hidden = true;
    $("empty").hidden = false;
    $("empty").querySelector("h2").textContent = models.length ? "A world begins with a prior." : "No model registered yet.";
    $("empty").querySelector("p").textContent = models.length ? "Create a run, record a sensor reading, and watch the belief update. Its evidence and previous revisions will stay inspectable." : "Start with scripts/run_explorer.py to register the example model, then refresh.";
  }
}

function correctionChanged() {
  const observation = inspection?.observations.find((o) => o.observation_id === $("correction").value);
  $("source").disabled = !!observation;
  $("reading-tick").disabled = !!observation;
  if (observation) {
    $("source").value = observation.payload.source_id;
    $("reading-tick").value = observation.payload.tick;
    $("reading-value").value = String(!observation.payload.value);
  }
  pendingReading = null;
}

$("correction").addEventListener("change", correctionChanged);
$("observation-form").addEventListener("input", () => { pendingReading = null; });
$("run-select").addEventListener("change", () => work(async () => { $("revision-select").value = ""; await loadInspection(); }));
$("revision-select").addEventListener("change", () => work(loadInspection));
$("return-live").addEventListener("click", () => work(async () => { $("revision-select").value = ""; await loadInspection(); }));
$("refresh").addEventListener("click", () => work(loadCatalog));
$("history-body").addEventListener("click", (event) => {
  const button = event.target.closest("[data-sequence]");
  if (button) work(async () => { $("revision-select").value = button.dataset.sequence; await loadInspection(); });
});
$("observation-form").addEventListener("submit", (event) => {
  event.preventDefault();
  work(async () => {
    const body = {measurement_id: pendingReading?.measurement_id || crypto.randomUUID(), source_id: $("source").value, tick: Number($("reading-tick").value), value: $("reading-value").value === "true", supersedes_id: $("correction").value || null};
    pendingReading = body;
    await api(`/runs/${encodeURIComponent(inspection.run.run_id)}/observations`, body);
    await loadInspection();
    message("Evidence recorded. The posterior and its provenance have been published together.");
  });
});
$("advance").addEventListener("click", () => work(async () => {
  await api(`/runs/${encodeURIComponent(inspection.run.run_id)}/ticks`, {tick: inspection.publication.belief.state_tick + 1});
  await loadInspection();
  message("Time advanced by one tick. The previous forecast is now frozen.");
}));
$("verify").addEventListener("click", () => work(async () => {
  const cutoff = $("revision-select").value === "" ? "" : `?sequence_cutoff=${inspection.publication.belief.sequence_cutoff}`;
  const report = await api(`/runs/${encodeURIComponent(inspection.run.run_id)}/replay${cutoff}`);
  message(`Replay verified ${report.events_checked} events through revision #${report.publication.belief.sequence_cutoff}. Saved beliefs, evidence snapshots, and frozen forecasts match recomputation.${report.calibration_revision_checked ? " Calibration evidence and derived sensor likelihoods also match." : ""}${report.import_plan_hash_checked || report.imported_observations_checked ? ` Verified ${report.imported_observations_checked} imported observations against their original source records.` : ""}${report.investigation_steps_checked ? ` Verified ${report.investigation_steps_checked} acquisition decisions and their costs.` : ""}`);
}));

$("agent-config-form").addEventListener("submit", (event) => {
  event.preventDefault();
  work(async () => {
    const config = {policy: "information_gain", allowed_sources: [...document.querySelectorAll("#agent-sources input:checked")].map((input) => input.value),
      horizon: Number($("agent-horizon").value), budget_credits: Number($("agent-budget").value), max_reads: Number($("agent-max-reads").value),
      min_gain_bits_per_credit: Number($("agent-min-gain").value), entropy_threshold_bits: Number($("agent-entropy").value)};
    await api(`/runs/${encodeURIComponent(inspection.run.run_id)}/investigation`, config);
    await loadInspection();
    message("Investigation enabled. The synthetic adapter supplies readings; the policy receives only acquired evidence.");
  });
});

async function investigationStep() {
  const runId = inspection.run.run_id;
  if (!pendingStep || pendingStep.runId !== runId) pendingStep = {runId, requestId: crypto.randomUUID()};
  const receipt = await api(`/runs/${encodeURIComponent(runId)}/investigation/steps`, {request_id: pendingStep.requestId});
  pendingStep = null;
  await loadInspection();
  if (receipt.step.decision.kind === "stop") {
    autoRunning = false;
    message(`Investigation complete. ${receipt.step.dispatched_reads} reads used ${receipt.step.spent_credits} credits.`);
  } else if (receipt.step.outcome?.status === "failed") {
    message("The sensor read failed. Its cost and outcome were recorded; the agent will not retry that sensor at this tick.", true);
  }
}

$("agent-step").addEventListener("click", () => work(investigationStep));
$("agent-run").addEventListener("click", async () => {
  if (busy || autoRunning || !inspection?.investigation || $("revision-select").value !== "") return;
  autoRunning = true;
  const generation = ++autoGeneration;
  const limit = inspection.investigation.config.horizon + inspection.investigation.config.max_reads + 1;
  controls();
  try {
    for (let index = 0; index < limit && autoRunning && generation === autoGeneration; index++) {
      await work(investigationStep);
      if (autoRunning) await new Promise((resolve) => setTimeout(resolve, 100));
    }
  } finally { if (generation === autoGeneration) autoRunning = false; controls(); }
});
$("agent-pause").addEventListener("click", () => {
  autoRunning = false;
  autoGeneration++;
  controls();
  message("Automatic execution paused. Any action already in flight will finish and remain recorded.");
});

async function loadTargets() {
  const generation = ++dialogGeneration;
  $("create-submit").disabled = true;
  $("create-target").innerHTML = "";
  $("create-error").hidden = true;
  try {
    const model = await api(`/models/${$("create-model").value}`);
    if (generation !== dialogGeneration) return;
    $("create-target").innerHTML = model.spec.states.map((state) => option(state, state)).join("");
    $("create-target").value = model.spec.states.includes("degraded") ? "degraded" : model.spec.states.at(-1);
    $("create-submit").disabled = false;
  } catch (error) {
    if (generation !== dialogGeneration) return;
    $("create-error").textContent = error.message;
    $("create-error").hidden = false;
  }
}
$("new-run").addEventListener("click", () => {
  $("create-id").value = `run-${crypto.randomUUID().slice(0, 8)}`;
  const epoch = new Date(Date.now() - 86400000);
  $("create-epoch").value = new Date(epoch.getTime() - epoch.getTimezoneOffset() * 60000).toISOString().slice(0, 19);
  $("create-dialog").showModal();
  loadTargets();
});
$("close-dialog").addEventListener("click", () => $("create-dialog").close());
$("create-model").addEventListener("change", loadTargets);
$("create-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  if ($("create-submit").disabled) return;
  const body = {run_id: $("create-id").value, model_hash: $("create-model").value, target_state: $("create-target").value, epoch: new Date($("create-epoch").value).toISOString()};
  $("create-submit").disabled = true;
  $("create-model").disabled = true;
  $("create-error").hidden = true;
  try {
    await api("/runs", body);
    $("create-dialog").close();
    $("revision-select").value = "";
    await work(() => loadCatalog(body.run_id));
  } catch (error) {
    $("create-error").textContent = error.message;
    $("create-error").hidden = false;
  } finally { $("create-submit").disabled = false; $("create-model").disabled = false; }
});

work(() => loadCatalog(new URLSearchParams(location.search).get("run") || undefined));

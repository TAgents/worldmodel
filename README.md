# World Modeling Engine

A research project for maintaining probabilistic beliefs, tracing their evidence, and choosing useful observations.

This migration brings the evidence and Bayesian engine into `TAgents/worldmodel`.
The original neural navigation project is preserved under
[`experiments/two_room/legacy/`](experiments/two_room/README.md), with original
file hashes and the [`legacy/two-room-v1` preservation branch](https://github.com/TAgents/worldmodel/tree/legacy/two-room-v1). Neural navigation integration is a
subsequent milestone; the archived experiment remains independently runnable.

The implementation provides immutable Pydantic contracts, Bayesian filtering for finite-state models with binary sensors, budgeted investigations, synthetic evaluation, PostgreSQL history, and a local FastAPI belief explorer. Persistent runs read fixed parameters from a versioned model artifact. A research workflow learns sensor likelihoods from labeled calibration evidence and evaluates parameter uncertainty. The Calibration Lab exposes its evidence, revisions, costs, and benchmarks, and can create persistent runs using a selected calibration snapshot's posterior means. A GitHub Actions adapter imports offline workflow exports through the same evidence, inference, and replay interfaces, with links to the original workflow runs.

## Open the explorer

From the repository root, using Python 3.12 or later:

```sh
.venv/bin/python -m pip install -e '.[dev,local-postgres]'
.venv/bin/python scripts/run_explorer.py
```

Open **http://127.0.0.1:8000**. The launcher starts the private local PostgreSQL cluster, applies migrations, registers the synthetic and software-delivery example models idempotently, and serves the API and interface together. Existing runs remain available. Use `--port 8001` if port 8000 is occupied. Ctrl-C stops the application; the database remains running.

Choose **New run**, keep the default model and target, and create it. The environment epoch defaults to yesterday so you can advance synthetic time without waiting. Record a cheap alert at tick 0: P(degraded) moves from 10% to 30.77%, and its one-tick forecast becomes 29.08%. Advance a tick, record more evidence, or select an earlier tick to supply late evidence. To correct a reading, choose it in **Evidence operation**; the engine preserves its original record and publishes a new revision.

The explorer shows posterior state probabilities, next-tick forecasts, changes between adjacent revisions, sensor likelihoods, transition probabilities, observation/ingestion timestamps, correction references, and exact model/revision/snapshot identifiers. Select a history row or knowledge snapshot to inspect the original publication. Historical views disable mutation controls and exclude subsequent evidence, revisions, and forecast closures. **Verify replay** recomputes the selected history from saved events.

Probabilities in the belief explorer are conditional on fixed model parameters. Select **Calibration lab →** to inspect learned sensor parameters and evaluation results separately, or **Imports →** to preview a GitHub Actions export. The interfaces use packaged HTML/CSS/JavaScript with no build step or external assets. Node is optional and only needed for the DOM checks below.

The launcher uses `WORLD_MODEL_DATABASE_URL` when supplied. For a database already migrated and populated, run `.venv/bin/python -m world_model.api` directly; the installed `world-model-serve` command is equivalent. This application is a single-user local research tool, bound to `127.0.0.1`, with local-host and same-origin browser checks. Tenant authentication and remote deployment are not implemented.

## HTTP API

Interactive documentation is at `/docs` (its Swagger assets require internet); the complete schema is at `/openapi.json`. The explorer itself works without internet.

| Method | Path | Result |
| --- | --- | --- |
| GET | `/health` | Database availability |
| GET | `/models`, `/models/{content_hash}` | Registrations and validated model specification |
| GET | `/models/{content_hash}/calibration` | Saved calibration evidence and source structure; 404 for an ordinary fixed model |
| GET / POST | `/runs` | List runs / create a run |
| POST | `/runs/{run_id}/observations` | Ingest a reading or correction atomically |
| POST | `/runs/{run_id}/ticks` | Advance exactly one environment tick |
| GET | `/runs/{run_id}/belief` | Selected saved publication |
| GET | `/runs/{run_id}/belief/history` | Saved publications through the cutoff |
| GET | `/runs/{run_id}/observations/{observation_id}` | Original immutable observation record |
| GET | `/runs/{run_id}/forecasts` | Issued forecasts and closure status at the cutoff |
| GET | `/runs/{run_id}/inspection` | Model, publication, selected evidence, history, and forecasts in one repeatable-read transaction |
| GET | `/runs/{run_id}/replay` | Verification of saved history through the cutoff |
| POST / GET | `/runs/{run_id}/investigation` | Configure immutable synthetic acquisition limits / inspect its trace |
| POST | `/runs/{run_id}/investigation/steps` | Choose and execute one action; requires a stable `request_id` for retries |
| GET | `/learning` | Calibration Lab with explicit creation of calibrated runs |
| GET | `/research`, `/research/{name}` | Completed calibration experiment catalog / verified report and parameter revisions |
| POST | `/research/{name}/runs` | Atomically register the selected calibration and create a new run |
| GET | `/imports` | GitHub Actions export preview and import interface |
| GET | `/imports/github-actions/example` | Illustrative export, configuration, and assumed model hash |
| POST | `/imports/github-actions/preview` | Validate and translate an export without creating a run |
| POST | `/imports/github-actions/runs` | Atomically save a previewed import as a new run |
| GET | `/runs/{run_id}/import/source` | Complete original source export, after integrity verification |

Belief, history, forecast, inspection, and replay reads accept `sequence_cutoff` and/or an offset-aware `as_of` timestamp. Both filters apply when supplied. They select by knowledge availability, not observation time. Observation-by-ID is an unrestricted archival lookup; use inspection for a historical evidence snapshot. The run and model catalogs accept `limit` (1–100) and `offset`; the explorer lists the newest 100 runs and models. Runs are not tenant-isolated in this local application.

Writes require JSON. Run creation accepts `run_id`, `model_hash`, `epoch`, and `target_state`; the run ID provides retry identity. An observation requires `measurement_id`, `source_id`, `tick`, and a boolean `value`, with optional `supersedes_id`. A tick request is `{"tick": 1}`. New publications return 201; identical retries return 200 with their original publication. Conflicts return 409, unavailable records 404, invalid requests 422, and database unavailability 503. Storage never accepts client-supplied knowledge timestamps.

## Run an autonomous investigation

Restart `scripts/run_explorer.py` after updating the code so it applies the acquisition migration. Create a **fresh run**. Under **Autonomous investigation**, choose the episode length, credit budget, read limit, and permitted sensors, then select **Enable on this fresh run**. The default is 20 ticks and 20 credits. Settings are immutable for that run.

Choose **One step** to inspect a single decision, or **Run episode** to let the agent choose readings and advance time until the horizon. **Pause** stops automatic requests after any action already in flight completes. Browser execution lasts while the page remains open; the saved trace can be resumed later. A configured investigation owns its run's observations and clock, so manual readings and corrections are rejected for that run. Manual runs retain their existing workflow.

The agent computes expected information gain exactly over each sensor's two possible outcomes:

`I(state; reading) = H(reading) - sum_state P(state) × H(reading | state)`

It ranks eligible sensors by bits of information per credit, with deterministic ties. This uses the mutual-information criterion from Bayesian experimental design, applied here to the current hidden state with fixed model parameters. It is a one-step heuristic, not a globally optimal scheduling algorithm or a guarantee of improved predictive accuracy. See [Kleinegesse & Gutmann, ICML 2020](https://proceedings.mlr.press/v119/kleinegesse20a.html) for the information-gain formulation.

At tick `t`, cumulative released credits are `floor(budget × (t + 1) / horizon)`, capped at the episode budget. Unspent credits carry forward. The agent can read each permitted sensor once per tick. It advances when no worthwhile read fits the available credits, the read limit is reached, or entropy is at/below the configured threshold. It continues forecasting through the full horizon after exhausting its reading budget, then records a terminal stop. Zero credits or zero permitted reads produce an episode with transitions and forecasts but no sensor dispatches.

The synthetic adapter returns only the requested reading. Its environment seed and hidden states are excluded from policy inputs and public inspection responses. The acquisition policy, adapter, and inference have separate interfaces. Current execution supports only this deterministic local adapter; it does not call remote tools.

Each step records the prior belief revision, policy version, uncertainty, all candidate gains and eligibility reasons, reserved/charged credits, action, outcome, and resulting revision. Writes hold the run lock and commit the evidence, belief, and action receipt together. Retrying a request ID returns the original receipt. A failed read consumes its credits and sensor slot once; unexpected programming or integrity failures roll back the whole local transaction. External tools will require a durable reservation and recovery protocol before they can use this executor.

**Verify replay** also recomputes recorded acquisition decisions and validates costs and publication links. Historical inspection bounds action traces by the selected publication's timestamp as well as its sequence, so a later failed or stopping action cannot appear in an earlier snapshot. Live inspection includes those actions even when they produced no new belief revision.

To execute a configured run from a terminal instead of the browser:

```sh
export WORLD_MODEL_DATABASE_URL="$(.venv/bin/python scripts/local_postgres.py dsn)"
.venv/bin/python -m world_model.agents run --run-id RUN_ID
```

## Compare acquisition policies

```sh
.venv/bin/python -m world_model.agents evaluate \
  --model docs/fixtures/binary-world-v1.json \
  --output artifacts/acquisition-development
```

The benchmark compares information gain per credit, fixed cheap, fixed precise, and a deterministic random sensor schedule. Each has the same 200-credit cap, 200-tick horizon, read limit, model, latent worlds, and potential observations. A fixed or random policy requests at most one source per tick and saves credits when its selected source is unaffordable. Actual costs and read counts are reported; equal caps do not force equal spending.

The decision process receives model bytes, policy settings, and only the current reading it requested. It never receives the environment seed, hidden labels, or unacquired sensor data. Forecasts are recorded before the evaluator consults outcomes. Archives contain actions, predictions, scores, source code, model/data bytes, and checksums. Tests change hidden labels and future observations to check that earlier decisions and predictions remain unchanged.

Development uses seeds 0–19. The prespecified acquisition holdout uses seeds 2000–2019 with unchanged settings and 2,000 paired whole-episode bootstrap samples:

```sh
.venv/bin/python -m world_model.agents evaluate \
  --model docs/fixtures/binary-world-v1.json \
  --output artifacts/acquisition-heldout --split heldout
```

Seeds 10000–10099 remain reserved for the original first-sprint gate. The acquisition gate requires a negative upper 95% bound on Brier-score difference against every fixed/random comparator. A tie with the strongest fixed policy does not pass. Do not tune against the acquisition holdout after inspecting its results.

The original development workspace archived runs at `artifacts/acquisition-development-v1/` and `artifacts/acquisition-heldout-v1/`. Generated archives are excluded from Git and are absent in a fresh checkout; the commands above reproduce the protocol. Those recorded runs produced these mean Brier scores (lower is better):

| Policy | Development | Acquisition holdout | Credits per episode |
| --- | --- | --- | --- |
| Information gain / credit | 0.089512 | 0.072567 | 200 |
| Fixed cheap | 0.089512 | 0.072567 | 200 |
| Fixed precise | 0.107828 | 0.079520 | 200 |
| Random schedule | 0.098516 | 0.078494 | 199 |

The adaptive policy selected the same schedule as fixed cheap in this model. Its paired Brier intervals were below zero against fixed precise and random, and exactly zero against fixed cheap, on both splits. **The acquisition improvement gate did not pass.** This validates bounded execution and exposes a limitation of this benchmark: it does not establish a benefit over the strongest fixed schedule. More expressive sensor/environment settings need a new prespecified protocol and fresh holdout data.

## Learn sensor reliability

The next experiment uses three persistent hidden states and three complementary sensors. Each sensor identifies a different state, so useful sensor selection depends on the current belief. The learner receives state names, transitions, costs, and initial state probabilities, with every sensor likelihood masked to 0.5. It learns those likelihoods from separate, explicitly labeled calibration examples. Labels are assumed trustworthy; this is supervised calibration, not autonomous concept discovery or learning from an LLM confidence score.

Each sensor/state cell starts with a Beta(1,1) prior. An alert adds one to alpha and a non-alert adds one to beta. Revisions retain source references, observation and availability timestamps, deduplicated evidence, posterior means, and equal-tail 95% credible intervals. An `as_of` cutoff excludes labels learned later. Conflicting duplicates or inconsistent labels fail explicitly. Revision hashes identify the exact prior, structure, cutoff, and evidence.

During evaluation, 256 frozen parameter samples are weighted jointly with the hidden state as requested readings arrive. State and parameter uncertainty remain correlated. Inference is exact for this finite sample bank and approximate for the continuous Beta posterior. Effective sample size (ESS) measures weight concentration; the prespecified 10% threshold does not prove numerical convergence. Intervals in the Lab describe the calibration posterior before an episode, while action traces record runtime state beliefs and ESS. Running an experiment does not automatically promote a model. Explicit promotion creates a new persistent run using fixed posterior means.

Run from the repository root, choosing a new archive directory each time:

```sh
.venv/bin/python -m world_model.learning evaluate \
  --output artifacts/reliability-development
.venv/bin/python -m world_model.learning evaluate \
  --output artifacts/reliability-heldout --split heldout
```

The installed `world-model-learn` command is equivalent. The CLI enforces the [frozen v1 protocol](docs/reliability-protocol-v1.json) and [synthetic environment](docs/fixtures/complementary-world-v1.json). Calibration uses seed 30000, independent calibration validation uses 33000, development uses 31000–31019, and holdout uses 32000–32039. No original reserved seeds are consumed. Do not tune this protocol against the holdout; changes require a new protocol and fresh evaluation data.

Every learned policy receives the same calibration set: 384 verified labels and 1,152 sensor readings, costing 384 label credits plus 1,152 sensor credits upfront. Each episode then has 100 acquisition credits and 100 forecasts. The neutral ablation ignores calibration, so it is not an equal-total-cost competitor. Evaluation labels and true sensor likelihoods are excluded from the learner process. Mean-likelihood, neutral, and oracle estimators are evaluator-only ablations on exactly the adaptive agent's acquired readings.

The original development workspace archived runs at `artifacts/reliability-development-v1/` and `artifacts/reliability-heldout-v1/`. These generated archives are excluded from Git. Use the commands above to create new archives; the Calibration Lab is empty until an experiment completes. The recorded 40-episode holdout produced:

| Policy / estimator | Multiclass Brier ↓ | Log loss ↓ | Acquisition credits / episode |
| --- | --- | --- | --- |
| Learned adaptive | 0.289228 | 0.544966 | 100 |
| Learned fixed probe A | 0.451479 | 0.743137 | 100 |
| Learned fixed probe B | 0.460995 | 0.757580 | 100 |
| Learned fixed probe C | 0.464651 | 0.768454 | 100 |
| Learned random | 0.333288 | 0.609548 | 100 |
| Posterior means, matched readings | 0.288794 | 0.544344 | 100 |
| Neutral likelihoods, matched readings | 0.666667 | 1.098612 | 100 |
| Known likelihoods, matched readings | 0.288329 | 0.543272 | 100 |

Multiclass Brier sums squared probability errors across states (range 0–2); it is not directly comparable with the earlier binary Brier scores. The acquisition gate passed: all paired 95% episode-bootstrap intervals against fixed and random policies were below zero. Adaptive minus random Brier was −0.044060, with interval [−0.059458, −0.028820]. Sensor parameter RMSE was 0.019570, and minimum ESS across learned policies was 57.34/256; both prespecified checks passed.

The parameter mixture did **not** improve on the simpler posterior-mean estimator: adaptive minus matched means was +0.000434 Brier, with interval [+0.000020, +0.000880]. These results support learned sensor calibration and adaptive acquisition in this synthetic setting, but do not justify the mixture's additional complexity on predictive performance. Persistent promotion therefore uses the simpler calibrated mean-likelihood artifact, preserving its calibration provenance and replay identity. Automatic structure discovery and real-domain transfer remain unproven.

Restart the explorer and open **http://127.0.0.1:8000/learning**. Select an experiment and calibration snapshot to inspect all nine parameter cells, intervals, evidence, learning curves, costs, paired comparisons, and gates. The benchmark always uses the final calibration snapshot; selecting an earlier snapshot changes the parameter view and the snapshot available for explicit promotion. It does not recompute the benchmark or establish that the earlier snapshot passed its gates. The API reads archives from `artifacts/` by default; `world-model-serve --research-directory PATH` selects another root. Displayed files are checked against manifest hashes, and revisions are rebuilt from their evidence. Checksums detect inconsistency relative to the manifest; they are not signatures.

Archives include protocol, training and validation evidence, neutral structure, evaluator truth, parameter revisions and draws, dataset, predictions, action traces, report, Python source, and runtime versions. The manifest is written last. Existing archives are never overwritten. Reproduction should use the recorded Python/NumPy/SciPy/Pydantic versions: sampling and inverse-CDF rounding may differ across releases.

To fit a revision from your own labeled evidence without running the synthetic benchmark:

```sh
.venv/bin/python -m world_model.learning fit \
  --structure artifacts/reliability-development/structure.json \
  --observations artifacts/reliability-development/training.json \
  --as-of 2026-01-01T00:06:24+00:00 \
  --output artifacts/calibration-revision.json
```

Each input record requires `measurement_id`, `source_id`, `state`, boolean `value`, offset-aware `observed_at` and `available_at`, and `source_ref`. Sensors and state labels must exist in the neutral structure. This command writes a new immutable research revision; operational model registration remains separate.

## Use calibration in a persistent run

Restart with `.venv/bin/python scripts/run_explorer.py` to apply migration 003. In the Calibration Lab, choose the experiment and snapshot, set the run identifier, target state, and environment start, then select **Create run from this snapshot**. Follow **Open saved run →**, record a sensor reading, and select **Verify replay**. The Explorer shows the calibrated likelihoods, calibration revision and evidence hash, and a link to the full saved calibration. The initial state belief starts at the supplied prior; calibration labels determine sensor parameters and are not counted again as observations of the new run.

Creation registers a model, stores the exact neutral structure and complete calibration revision, and publishes the new run's initial belief in one PostgreSQL transaction. An invalid target, conflicting run ID, future calibration cutoff, or failed publication rolls everything back. Retrying the same run ID with identical settings returns the original creation receipt, including after later observations. The Lab retains retry inputs after a lost response and catalog refresh. Changing the selected snapshot starts a new creation request. Existing runs keep their original model and evidence.

For the API, POST `/research/{name}/runs` with `run_id`, `calibration_revision` (the full revision hash), `epoch`, and `target_state`. The server verifies the archive and the requested snapshot before writing. Results use the existing `WriteResult` contract: 201 for creation, 200 for an identical retry, 409 for a conflicting run ID, 404 for a missing snapshot, and 422 for invalid inputs. GET `/models/{content_hash}/calibration` reads the database copy independently of the original research archive.

For your own fitted revision, use the CLI after migrating the database:

```sh
export WORLD_MODEL_DATABASE_URL="$(.venv/bin/python scripts/local_postgres.py dsn)"
.venv/bin/python -m world_model.storage migrate
.venv/bin/python -m world_model.learning promote \
  --structure artifacts/reliability-development/structure.json \
  --revision artifacts/calibration-revision.json \
  --run-id calibrated-example --target-state mode_a \
  --epoch 2026-01-02T00:00:00+00:00
.venv/bin/python -m world_model.storage replay --run-id calibrated-example
```

Use the `fit` example above to create the revision file first. The Python storage API also exposes `register_calibrated_model(structure, revision)` for registration without creating a run. Ordinary `register_model` rejects derived artifacts without their calibration evidence. Once registered, the calibrated model is available in the Explorer's ordinary **New run** selector.

The model artifact embeds the derivation method, revision ID, exact revision-content hash, neutral-structure hash, evidence cutoff, and evidence count. PostgreSQL rejects updates, deletes, and truncation of calibration history. Model reads and replay verify the saved evidence, rebuild the posterior means, and compare the resulting artifact bytes; replay returns `calibration_revision_checked`. A missing or inconsistent calibration fails verification. Only recomputed display interval bounds allow an absolute tolerance of 1e-12 for inverse-CDF library roundoff; saved bytes, evidence, counts, means, and other parameters must match exactly. Historical run inspection retains the same pinned derivation. Model registration and run creation use the server's clock, so a source's calibration cutoff cannot backdate a saved run.

These runs use fixed posterior means throughout their lifetime. New labeled evidence requires fitting and explicitly registering a new calibration and starting a new run. The optional synthetic investigation uses the registered model as its simulated environment; it exercises acquisition and persistence, but is not an independent predictive validation of the learned parameters. Independent validation remains in the separate research benchmark.

## Import GitHub Actions evidence

Restart `.venv/bin/python scripts/run_explorer.py` to apply migration 004 and register the software-delivery example model. Open **http://127.0.0.1:8000/imports**, select **Load illustrative example**, preview it, then import it into a new run. Open the saved run and select **Verify replay**. The seven-window example produces nine observations, six time advances, and source links in the Explorer. Two windows with inconclusive results contribute no observation; a day with no records also contributes no observation.

The included export is fabricated demonstration data. The `software-delivery-assumed` model's prior, transitions, and sensor likelihoods are explicit assumptions, not estimates learned from a repository. Its `stable` and `unstable` states are modeling hypotheses; CI outcomes do not supply ground-truth state labels. A successful import establishes data handling and reproducible inference, not predictive accuracy on real software projects.

Preview the same example without PostgreSQL:

```sh
.venv/bin/python -m world_model.adapters preview
```

After the launcher has applied migrations, import it from the terminal:

```sh
export WORLD_MODEL_DATABASE_URL="$(.venv/bin/python scripts/local_postgres.py dsn)"
.venv/bin/python -m world_model.adapters import --run-id delivery-example
.venv/bin/python -m world_model.storage replay --run-id delivery-example
```

The installed `world-model-import` command is equivalent. Reinstall the editable package to register this new command. Use `--model`, `--export`, and `--config` to supply your own files. A derived model must already be registered with its calibration provenance. Importing does not fit or update its parameters.

For your own data, export every page of the repository's workflow runs, retaining all statuses and conclusions:

```sh
gh api --paginate --slurp 'repos/OWNER/REPO/actions/runs?per_page=100' > workflow-runs.json
```

The adapter accepts one [GitHub workflow-runs REST response](https://docs.github.com/en/rest/actions/workflow-runs?apiVersion=2022-11-28) or an array of responses produced by [`gh api --paginate --slurp`](https://cli.github.com/manual/gh_api). It rejects inconsistent page totals and exports whose unique run count differs from `total_count`. GitHub applies a 1,000-result cap to searches using certain filters; check the API's coverage before supplying filtered exports. Matching counts cannot prove that your export query included every relevant workflow or outcome. Deleted runs and previous rerun attempts cannot be reconstructed from the latest run snapshot. This adapter reads a local export; it does not request credentials, fetch GitHub data, or write to GitHub.

Create a configuration file like this, replacing the repository and numeric workflow IDs with those in your export:

```json
{
  "repository": "OWNER/REPO",
  "branch": "main",
  "event": "push",
  "window_start": "2026-01-01T00:00:00Z",
  "windows": 7,
  "workflow_sources": {"unit_tests": 101, "integration_tests": 202}
}
```

Each mapping key must be a sensor in the selected model; each workflow ID maps to one sensor. Windows have the model's tick duration (one day in the example), and all windows must have closed before a stored preview or import. Each source produces at most one boolean observation per window. A completed `failure` or `timed_out` makes it true; it is false only when every included run completed successfully. Without a known failure, pending, cancelled, skipped, and other outcomes leave the window unobserved. A later success never erases a failure in the same window. Records for other branches, events, workflows, or times are reported as excluded. Exact duplicate run records are deduplicated; conflicting snapshots or attempts for one run ID are rejected. Limits are 8 MiB, 10,000 unique runs, 200 windows, and four mapped workflows per import.

The adapter groups records by GitHub's `updated_at`, which is a source update time, not a guaranteed execution completion time. An observation is dated at its window's end; the raw run IDs, attempts, commit hashes, statuses, and update timestamps remain attached as provenance. The engine records knowledge availability using its own clock during import. It never backdates ingestion to the source timestamp. Historical sequence inspection shows successive replay steps, but importing an old export does not demonstrate what the engine knew at those past dates or establish a retrospective forecasting score. Source links in historical inspection are limited to that snapshot; the explicitly labeled source download always contains the complete export.

The API preview accepts `model_hash`, `export_json` (the JSON file as a string), and `config`. Creation adds `run_id`, `target_state`, and the returned `expected_plan_hash`. The preview hash binds the exact export bytes, mapping, clock, and model. A changed preview fails with 409. Creation stores the raw export, translated plan, clock advances, evidence, and beliefs in one transaction. An identical retry returns the original final publication. Imported runs preserve their snapshot and reject manual evidence, clock advances, and synthetic investigations; a new export requires a new run. Replay verifies the source hash and translation as well as ordinary inference, reporting `imported_observations_checked` and `import_plan_hash_checked`.

Before evaluating real delivery predictions, choose an independently observable target and a temporal evaluation split. Calibrate likelihoods against the same window aggregation, branch/event population, and workflow definitions used by the adapter. Workflow volume and correlated failures can invalidate the illustrative model's assumptions. No real-repository predictive validation has been completed yet.

## Run the example

Use Python 3.12 or later. From the repository root:

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -e '.[dev]'
.venv/bin/python -m world_model \
  --model docs/fixtures/binary-world-v1.json \
  --readings examples/readings.json \
  --tick 2
```

The example estimates P(degraded at tick 2) = 0.6829077698901129 and forecasts P(degraded at tick 3) = 0.6209588375032994. Output also contains the full state distributions, model hash, inference version, and selected readings. After installation, the `world-model` command accepts the same arguments.

The filter counts identical sensor/tick readings once, rejects conflicting or future readings, and advances the state through ticks without observations. Zero-probability evidence raises an explicit error. It has no access to simulator truth, clocks, tools, or storage.

## Run the development benchmark

```sh
.venv/bin/python -m world_model.evaluation \
  --model docs/fixtures/binary-world-v1.json \
  --output artifacts/development
```

The default benchmark uses development seeds 0 through 19, with 200 forecasts per episode and one cheap reading per origin tick. Each episode costs 200 credits. The runner rejects schedules that exceed the model's credit or read limits before starting inference.

The simulator uses a versioned SHA-256 counter generator with independent streams for state changes and each sensor. The inference worker runs in a separate process. Its protocol accepts model bytes, episode resets, and current-tick readings; seeds, hidden states, and future readings are excluded. The evaluator consults each target label only after receiving its immutable forecast.

The filter is compared with a transition-only baseline that ignores measurements and a memoryless baseline that uses only the current reading. All three have the same model parameters and access to the same acquired observations. The report includes Brier score, natural-log loss, calibration counts, and paired confidence intervals from 2,000 bootstrap resamples of complete episodes.

Each output directory contains:

- `model.json` and `dataset.json`: exact model bytes and the generated episodes.
- `predictions.jsonl`: issued forecasts with their origin/target ticks and acquired-evidence digests, without outcome labels.
- `report.json`: metrics, baseline comparisons, costs, and evaluation scope.
- `source/`: the Python source used for the run, including the generator.
- `manifest.json`: SHA-256 hashes of every archived file and Python/Pydantic versions. This file is written last as the completion marker.

Choose a new output directory for each run; existing archives are never overwritten. Generated artifacts are excluded from version control. Development results do not satisfy the reserved predictive gate: the report keeps that field null. Seeds 10000 through 10099 remain reserved for the final sprint evaluation after persistence, replay, and interface checks.

## PostgreSQL storage

Install the optional local server binaries with Python 3.12, then start a private cluster for this checkout:

```sh
.venv/bin/python -m pip install -e '.[dev,local-postgres]'
.venv/bin/python scripts/local_postgres.py start
export WORLD_MODEL_DATABASE_URL="$(.venv/bin/python scripts/local_postgres.py dsn)"
.venv/bin/python -m world_model.storage migrate
.venv/bin/python -m world_model.storage demo \
  --model docs/fixtures/binary-world-v1.json
```

The helper uses a full PostgreSQL installation from `PATH`, `WORLD_MODEL_POSTGRES_BIN`, or the optional bundled server. It keeps data in `.local/postgres/`, listens only on a private Unix socket, and uses no TCP port. Use `scripts/local_postgres.py stop` to stop it without deleting its data.

Alternatively, start the Docker service and use its loopback connection:

```sh
docker compose up -d postgres
export WORLD_MODEL_DATABASE_URL='postgresql://world_model:world_model_local@127.0.0.1:55439/world_model'
.venv/bin/python -m world_model.storage migrate
```

The demo creates a run, ingests observations, advances two ticks, accepts late evidence, and corrects an earlier reading. It verifies the original published belief is unchanged and replays all seven events. The initial cheap alert gives P(degraded) = 4/13; the late precise non-alert gives P(degraded at tick 2) = 49391/1697350. These are the same exact calculations used by the inference tests.

Use the returned run ID to inspect persisted records:

```sh
.venv/bin/python -m world_model.storage belief --run-id RUN_ID
.venv/bin/python -m world_model.storage history --run-id RUN_ID
.venv/bin/python -m world_model.storage replay --run-id RUN_ID
.venv/bin/python -m world_model.storage belief --run-id RUN_ID --sequence-cutoff 1
```

Historical commands also accept an `--as-of` UTC timestamp. Knowledge timestamps are assigned inside the publication transaction and strictly increase within a run. The returned sequence cutoff identifies its exact committed evidence snapshot. Environment time comes from the run's epoch and observed tick; ordinary ingestion cannot supply or backdate knowledge timestamps. Rapid synthetic runs use an epoch in the past or an explicitly injected test clock.

Writes lock one run at a time. An event, observation, evidence snapshot, belief, forecast, and current-state projection commit together. Identical measurement retries return their original publication; conflicting identities fail. Corrections must name the currently active observation in the same run, source, and tick. Database constraints prevent branching correction chains, and triggers reject updates, deletes, and truncation of history tables.

Clock advances freeze the latest preceding forecast in a separate append-only record. Replay reconstructs every posterior and its exact evidence membership from events, then compares them with the stored publications and frozen forecasts. `rebuild-projection --run-id RUN_ID` repairs the mutable current-state projection after a successful replay; it does not replace history.

## Test

```sh
.venv/bin/python -m pytest -q
```

The tests check all reference cases and generated sequences through tick 8 against independent enumeration of hidden-state paths using exact rational arithmetic. Additional checks cover a three-state model with different sensor names, validation, immutable serialization, duplicates, impossible observations, and the command-line interfaces. Simulator/evaluation tests verify fixed generator output, independent sensor streams, baseline definitions, budget limits, scoring, archive hashes, and that changing hidden labels or future readings cannot alter earlier predictions.

PostgreSQL tests are explicitly skipped unless `WORLD_MODEL_TEST_DATABASE_URL` is configured. To start the checkout's private server and run the entire suite:

```sh
.venv/bin/python scripts/check_postgres.py
```

This command fails if server startup fails. With an existing dedicated test database, set `WORLD_MODEL_TEST_DATABASE_URL` before running it. Each test creates and removes only its own randomly named schema. Integration cases exercise concurrent ingestion, conflicting corrections, rollback, terminated connections, immutable history, time cutoffs, restart, and projection repair. The SQL and PL/pgSQL parser checks and pure replay tests can run without a database server.

HTTP tests use FastAPI's in-process test client to check typed requests, original retry receipts, time filters, error responses, and the local browser boundary. The PostgreSQL suite also exercises the HTTP pipeline, including late corrections and original historical snapshots.

For interaction checks without a browser process, install Node 24 and run:

```sh
npm ci
npm test
```

The DOM checks default to `.venv/bin/python`. Set `WORLD_MODEL_PYTHON` to a
Python executable when using a different environment, for example
`WORLD_MODEL_PYTHON=python npm test` with an activated environment.

These DOM checks use exact-oracle fixtures and the shipped JavaScript. They cover creating a run, retries after a lost response, reading ingestion, late evidence, correction controls, historical isolation, replay, and recovery from an unavailable database. They also exercise investigation configuration, action costs, pause/resume, lost action responses, and bounded completion. Calibration Lab checks cover empty archives, parameter intervals, historical evidence selection, paired comparisons, research gates, costs, integrity failures, and recovery. They do not verify visual layout.

Learning tests check conjugate updates, label availability and conflicts, revision integrity, and joint parameter/state inference against independent rational enumeration of parameter atoms and complete state paths. They also check the learner's input boundary, budget accounting, and invariance of predictions to hidden-label or future-reading changes. Research reads and promotion request validation are exercised through the real FastAPI application. PostgreSQL promotion tests cover atomic rollback, concurrent retries, retained history, archive removal, evidence tampering, fixed-mean inference, and budgeted investigation replay. UI checks cover promotion retries and loading a calibrated run directly into the Explorer.

Adapter tests cover pagination, duplicate and conflicting source records, missing outcomes, window boundaries, source translation and hashes, generic model interfaces, and independent exact-oracle replay. PostgreSQL import cases cover atomic rollback, concurrent retries, calibrated models, source integrity, immutability, and time cutoffs through the real API. Import DOM checks exercise stale previews, failed requests, retry after a lost response, source links bounded to historical evidence, and mutation controls on imported runs. Database tests remain skipped without an explicitly configured test database.

For browser checks and desktop/mobile screenshots:

```sh
.venv/bin/python -m pip install -e '.[dev,browser]'
.venv/bin/python -m playwright install chromium
.venv/bin/python scripts/check_explorer.py
```

Pass `--browser /absolute/path/to/chromium` to use an existing executable. The check routes browser requests through the real FastAPI application with exact-oracle storage fixtures and needs no TCP listener. Screenshots are saved under `artifacts/explorer/`. Real persistence is verified separately by `scripts/check_postgres.py`; browser fixtures do not establish database correctness.

## Implementation boundaries

GitHub Actions runs the complete Python suite against PostgreSQL 17 on Python
3.12, builds the package, and checks Explorer interactions with Node 24.
The database job uses `scripts/check_postgres.py`, which fails if it cannot
connect instead of silently running only the tests that do not need PostgreSQL.
Legacy simulator checks run without PyTorch or a GPU. Full neural training and
browser visual verification remain separate from these CI checks.

The pure filter consumes an already-selected set of active readings. PostgreSQL ingestion resolves correction chains and publishes the active evidence snapshot. Historical reads return the original committed publication rather than recomputing it using later evidence. Replay uses the saved model artifact and checks the recorded inference version.

The API and explorer expose the same immutable storage publications. The inspection endpoint reads all of its components in one repeatable-read transaction and bounds historical views by the selected publication. Budgeted acquisition currently uses a deterministic local synthetic adapter. Labeled sensor calibration can be explicitly promoted into persistent fixed-mean runs with verified provenance. Offline GitHub Actions imports exercise the same core interfaces with external evidence; their example parameters are unvalidated assumptions. Online parameter updates within a run, autonomous remote tools, neural discovery, transition learning, and autonomous model evolution remain future work. Ordinary HTTP observation ingestion and offline imports record supplied evidence and do not spend acquisition credits.

## Project files

- `world_model/core/`: validated model artifacts and record contracts.
- `world_model/inference/`: pure filtering and forecasting.
- `world_model/synthetic/`: the versioned simulator and dataset contracts.
- `world_model/evaluation/`: isolated forecasting, baselines, metrics, and reproducible archives.
- `world_model/storage/`: PostgreSQL migrations, transactional ingestion, history, and replay.
- `world_model/api/`: FastAPI routes and packaged explorer assets.
- `world_model/agents/`: information-gain policy, bounded state machine, isolated worker, and trace verification.
- `world_model/learning/`: labeled calibration, joint parameter/state filtering, isolated evaluation, research catalog, and fixed-mean model derivation.
- `world_model/adapters/`: GitHub Actions export validation, window aggregation, CLI, and illustrative fixtures.
- `world_model/storage/imports.py`: atomic snapshot imports, source provenance, and translation verification.
- `world_model/storage/calibrations.py`: atomic calibrated registration/run creation and stored provenance verification.
- `world_model/storage/investigations.py`: atomic synthetic dispatch and immutable credit ledger.
- `scripts/run_explorer.py`: local database setup and application launcher.
- `scripts/local_postgres.py`: private local database lifecycle.
- `scripts/check_postgres.py`: required database acceptance checks.
- `tests/`: exact reference oracle, contract validation, and CLI tests.
- `examples/readings.json`: the three-reading example above.

- [First sprint specification](docs/first-sprint.md): model assumptions, contracts, evaluation, acceptance tests, and implementation order.
- [Model definition](docs/fixtures/binary-world-v1.json): fixed transition probabilities, observation likelihoods, and acquisition costs.
- [Numerical reference cases](docs/fixtures/posterior-cases-v1.json): exact probabilities for independent inference tests.

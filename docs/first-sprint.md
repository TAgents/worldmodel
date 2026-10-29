# First sprint specification

Establish a verifiable observation-to-belief pipeline before adding neural extraction or autonomous model discovery. The engine will track a changing binary state, predict the next state, preserve the evidence behind each revision, and expose those revisions through an API and a small explorer.

This is the implementation baseline for the first two-week sprint, dated 27 October 2026. The numerical parameters below are deliberately chosen simulator assumptions. They are not estimates of a real service's behavior.

## Prediction target and scope

For a synthetic service, estimate the probability that its state will be **degraded one simulated minute ahead**, using only measurements available when the forecast is issued. Also expose the probability of its current state being degraded.

The sprint covers one local workspace, multiple isolated simulation runs, structured observations, a fixed model, and scripted acquisition. The simulator owns hidden states and random seeds; only the evaluator may read them. The inference process receives model parameters, clock events, and observations. It must not receive the simulator seed or ground-truth stream.

Parameter learning, LLM extraction, autonomous acquisition, model evolution, causal claims, and multi-tenant deployment are later milestones. The first result establishes inference and provenance correctness under known assumptions.

## Synthetic world and observation models

The authoritative parameters are in [binary-world-v1.json](fixtures/binary-world-v1.json). One tick is one simulated minute. Tick zero starts with P(degraded) = 0.10, before any observation. The state remains fixed within a tick.

| Current state | Next state healthy | Next state degraded |
| --- | ---: | ---: |
| Healthy | 0.98 | 0.02 |
| Degraded | 0.10 | 0.90 |

Each sensor returns a binary alert. An alert is a noisy measurement, not a declaration of the hidden state.

| Sensor | P(alert given healthy) | P(alert given degraded) | Cost per dispatched read |
| --- | ---: | ---: | ---: |
| cheap | 0.20 | 0.80 | 1 credit |
| precise | 0.05 | 0.95 | 5 credits |

Sensor errors are independent conditional on the full state trajectory. Each sensor can be dispatched at most once per tick. Copied measurements are the same evidence, regardless of the number of messages carrying them. Scripted acquisition, missing readings, and delivery delays are independent of the hidden state in this version. The actions do not change the state.

Given the previous tick's posterior probability p, the next tick's prior is:

    prior = 0.02 + 0.88 * p

Tick zero uses 0.10 directly. For an observed value y, let L_D = P(y given degraded) and L_H = P(y given healthy):

    posterior = prior * L_D / (prior * L_D + (1 - prior) * L_H)
    next_tick_forecast = 0.02 + 0.88 * posterior

Use the complementary likelihoods for a non-alert. Independent readings in the same tick update sequentially without an intervening transition. Advancing without evidence still applies the transition. Perform transitions for every elapsed tick.

For example, a cheap alert at tick zero changes P(degraded) from 0.10 to 4/13, and the next-tick forecast becomes 189/650. Adding an independent precise non-alert in that tick changes the posterior to 4/175. Re-delivering the cheap alert leaves the posterior unchanged. [Reference cases](fixtures/posterior-cases-v1.json) specify these and other exact results as rational numbers; these are deterministic arithmetic checks, not evaluation episodes.

## Evidence and time semantics

Every run declares a UTC epoch. Observed time is epoch plus tick times 60 seconds. A clock controlled by the test harness advances virtual time without real waits. The engine assigns ingestion time and an increasing event sequence per run; ordinary callers cannot backdate ingestion. Reject future-tick measurements until that run's clock has advanced.

Filtering at tick t uses measurements whose observed tick is at most t and whose ingestion time is within the requested knowledge cutoff. Same-timestamp events are ordered by event sequence. A saved revision records its exact sequence cutoff.

Late evidence triggers recomputation from the affected tick through the current tick, under the same model. It creates a new current belief revision and preserves previously published beliefs and forecasts. Forecasts scored by the evaluator are frozen before their target tick begins; later evidence cannot replace them.

A historical query returns the last revision actually published by its knowledge cutoff. It never applies today's evidence to an old revision. Model artifacts must also have been registered by that cutoff. The sprint does not expose retrospective smoothing with observations from later state ticks.

Store observations immutably. A correction is a new observation with a reference to the superseded observation, for the same run, sensor, and observed tick. Corrections form a single chain. At any knowledge cutoff only its latest available member contributes to inference. Earlier cutoffs retain the earlier value.

Ingestion is idempotent on (run_id, source_id, measurement_id). An identical normalized payload returns the original observation and creates no inference revision. A different payload under the same key is a conflict. A second measurement ID for an occupied sensor/tick slot also conflicts unless it explicitly corrects the active observation.

## Minimum contracts and persistence

Keep these contracts independent of the sensor names and state labels. In this sprint the fixed model specifies their supported values.

| Contract | Required meaning and fields |
| --- | --- |
| ModelSpec | Model ID, immutable version, registration time, content hash, state support, prior, transition matrix, observation likelihoods, time unit, and action costs. |
| Observation | Run, assigned observation ID, producer measurement ID, source, observed tick/time, binary value, assigned ingestion time and event sequence, and optional supersedes ID. Preserve the submitted structured payload. |
| BeliefRevision | Run, variable, state tick, publication time, sequence cutoff, distribution over states, model hash, evidence snapshot ID, inference implementation version, and previous revision ID. |
| EvidenceSnapshot | Immutable ordered references to exactly the active observations used by that revision, with their ingestion sequences. |
| Forecast | Origin tick, target tick, issue time, probability, originating belief revision, model hash, and evidence snapshot. |
| Action and Outcome | Action ID, run/tick, permitted source, reserved and charged cost, status, dispatch/completion time, observation link or explicit failure. |

Use PostgreSQL for the append-only event history and persisted artifacts, with rebuildable current-state projections. Serialize updates per run. The observation event, evidence snapshot, posterior revision, forecast, and current projection must commit atomically, so a crash cannot leave an accepted observation with a missing update. Advancing time follows the same rule. A small in-process worker is sufficient; this sprint does not require a separate broker. Compute each model's SHA-256 hash over its exact immutable artifact bytes; registration metadata is stored separately.

Replay consumes recorded observations, clock events, model artifacts, and outcomes. It does not call sensors again. Compare replayed probabilities and evidence membership; new replay execution timestamps need not match original timestamps.

## API and explorer behavior

Use FastAPI and typed Pydantic contracts. The first API is local and bound to loopback.

| Operation | Behavior |
| --- | --- |
| POST /runs | Select the fixed registered model; create a tick-zero prior revision and forecast. The simulator seed stays outside the inference API. |
| POST /runs/{id}/ticks | Advance exactly one tick; close and freeze the preceding tick's latest forecast, then apply its state transition. Tick identity makes retries idempotent. |
| POST /runs/{id}/observations | Ingest and synchronously publish a revised current belief. Return 201 for new evidence, 200 for an identical retry, 409 for an identity conflict, and 422 for invalid content. |
| GET /runs/{id}/belief | Return the current revision, or the last published revision at an optional as_of timestamp. Accept an optional sequence cutoff to disambiguate ties. |
| GET /runs/{id}/belief/history | Return immutable revisions in publication order. |
| GET /runs/{id}/observations/{observation_id} | Return original evidence, provenance, and correction links. |
| GET /runs/{id}/forecasts | Return issued forecasts, their evidence/model references, and whether each was frozen for scoring. |

Every new belief revision issues exactly one forecast for its current origin tick. Only the latest forecast frozen at tick closure is scored; all issued versions remain inspectable. Advancing an empty tick also issues a forecast. Invalid requests produce no domain events or belief changes.

The explorer needs a run selector and a belief detail view showing the current state distribution, one-minute forecast, revision history, model assumptions, and evidence links. Selecting a revision displays its own snapshot. Show the before/after probability change and triggering event; free-form generated explanations are unnecessary. Simulator truth appears only in a separate evaluation view.

## Acquisition boundaries

The permitted actions are read_cheap, read_precise, and advance_tick. The first runner uses a fixed schedule; it does not optimize information gain.

Reserve credit before dispatch. A dispatched read consumes its stated cost even if it fails; a rejected read consumes none. Repeating the same action ID returns its recorded outcome without another charge or dispatch. A failed read produces no measurement and cannot be retried under a new action ID for that sensor/tick. Advancing time costs zero and closes each tick once.

A 200-tick episode allows at most 200 credits, at most one dispatch per sensor/tick, and at most 200 total reads. Tests may use either sensor within that limit. Later acquisition comparisons must use the same observation models, episode horizon, and budget. The first predictive benchmark uses exactly one cheap reading per origin tick, costing 200 credits; it tests filtering, not acquisition strategy.

## Evaluation and acceptance

Generate 201 hidden states per episode, numbered 0 through 200. At each origin tick 0 through 199, ingest its cheap reading, freeze the forecast, and then reveal the next state only to the evaluator. Score every origin tick. Sensor noise and state transitions use separate random streams.

Development episodes use seeds 0 through 19. Reserve seeds 10000 through 10099 for the first final evaluation; do not use them to tune the model or thresholds. Pin the generator implementation and archive the generated dataset and its hash. Reuse that dataset for regression checks; subsequent research claims need fresh evaluation episodes.

Compare the filter with two fixed baselines. Both receive the same observation stream and model parameters. The transition-only baseline deliberately ignores readings and propagates the initial prior. The memoryless baseline uses the same unconditional prior at each tick, conditions on only that tick's reading, and predicts one transition ahead. Neither can use future observations or labels.

Report mean Brier score and log loss, paired differences from each baseline, and calibration in ten equal-width probability bins with counts: [0, 0.1), [0.1, 0.2), through [0.9, 1]. Define Brier score as mean((forecast - next_state_indicator)^2); use natural logarithms for log loss. Bootstrap complete episodes, not individual ticks, with 2,000 paired resamples and evaluation RNG seed 20261027. Report percentile 95% intervals. Calibration is descriptive in this first benchmark; inference correctness has separate exact tests.

| Acceptance test | Pass condition |
| --- | --- |
| Exact inference | Every reference posterior and forecast agrees with independent state-path enumeration to absolute error at most 1e-12. |
| Sequence correctness | For short generated trajectories up to tick 8, exact enumeration and sequential filtering agree to 1e-12, including missing readings and both sensors. |
| Duplicate and conflicting evidence | A retry creates no event/revision or extra charge; conflicting identities fail; a correction replaces rather than adds its likelihood. |
| Temporal history | A late observation or correction changes the current posterior as appropriate, while earlier as_of results and frozen forecasts remain byte-for-byte unchanged. |
| Isolation and no future leakage | Revisions refer only to their own run, eligible evidence, and registered model. The inference process has no ground-truth/seed access. |
| Persistence and replay | Restart and replay reconstruct probabilities and exact evidence membership; interrupted transactions leave either the entire update or none of it. |
| Acquisition accounting | Exhausted budgets and repeated sensor dispatches fail before dispatch; failure and idempotent retry costs match the rules above. |
| Predictive value | On the reserved evaluation, the upper endpoint of the paired 95% interval for filter-minus-baseline Brier score is below zero against both baselines. If it fails, investigate before extending scope. |
| Inspectability | From a selected belief revision, the user can inspect its evidence, model, forecast, and preceding revision without application logs. |

Passing these tests supports correct filtering and predictive value in this specified synthetic world. It does not establish robustness to an incorrect likelihood, real-world calibration, parameter learning, or autonomous discovery. Those require separate experiments.

## Implementation order

1. Implement typed contracts, the pure filter, and exact numerical tests using the fixtures.
2. Add the independent simulator and evaluation harness; freeze the generator before opening the reserved evaluation.
3. Add PostgreSQL migrations, atomic ingestion, correction handling, persistence, and replay tests.
4. Expose the API and build the belief detail view.
5. Run the reserved evaluation and the complete observation-to-history demonstration.

The sprint demonstration starts a run, ingests an alert, displays its posterior and evidence, advances the clock, delivers late evidence, and proves that the original published revision still has its original meaning.

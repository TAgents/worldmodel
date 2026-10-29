CREATE TABLE models (
    content_hash text PRIMARY KEY CHECK (content_hash ~ '^[a-f0-9]{64}$'),
    model_id text NOT NULL,
    model_version text NOT NULL,
    content bytea NOT NULL,
    registered_at timestamptz NOT NULL,
    UNIQUE (model_id, model_version)
);

CREATE TABLE runs (
    run_id text PRIMARY KEY,
    model_hash text NOT NULL REFERENCES models(content_hash),
    target_state text NOT NULL,
    epoch timestamptz NOT NULL,
    created_at timestamptz NOT NULL CHECK (created_at >= epoch),
    UNIQUE (run_id, model_hash),
    UNIQUE (run_id, target_state)
);

CREATE TABLE events (
    run_id text NOT NULL REFERENCES runs(run_id),
    sequence bigint NOT NULL CHECK (sequence >= 0),
    kind text NOT NULL CHECK (kind IN ('run_created', 'tick_advanced', 'observation_ingested')),
    recorded_at timestamptz NOT NULL,
    payload jsonb NOT NULL,
    PRIMARY KEY (run_id, sequence)
);
CREATE INDEX events_by_time ON events (run_id, recorded_at, sequence);

CREATE TABLE observations (
    run_id text NOT NULL,
    observation_id text NOT NULL,
    source_id text NOT NULL,
    measurement_id text NOT NULL,
    observed_tick bigint NOT NULL CHECK (observed_tick >= 0),
    value boolean NOT NULL,
    observed_at timestamptz NOT NULL,
    ingested_at timestamptz NOT NULL CHECK (ingested_at >= observed_at),
    event_sequence bigint NOT NULL CHECK (event_sequence > 0),
    supersedes_id text CHECK (supersedes_id <> observation_id),
    PRIMARY KEY (run_id, observation_id),
    UNIQUE (run_id, source_id, measurement_id),
    UNIQUE (run_id, event_sequence),
    UNIQUE (run_id, observation_id, event_sequence),
    UNIQUE (run_id, observation_id, source_id, observed_tick),
    UNIQUE (run_id, supersedes_id),
    FOREIGN KEY (run_id, event_sequence) REFERENCES events(run_id, sequence),
    FOREIGN KEY (run_id, supersedes_id, source_id, observed_tick)
        REFERENCES observations(run_id, observation_id, source_id, observed_tick)
);
CREATE UNIQUE INDEX one_root_measurement_per_slot
    ON observations (run_id, source_id, observed_tick) WHERE supersedes_id IS NULL;

CREATE TABLE snapshots (
    run_id text NOT NULL,
    snapshot_id text NOT NULL,
    sequence_cutoff bigint NOT NULL,
    PRIMARY KEY (run_id, snapshot_id),
    UNIQUE (run_id, sequence_cutoff),
    UNIQUE (run_id, snapshot_id, sequence_cutoff),
    FOREIGN KEY (run_id, sequence_cutoff) REFERENCES events(run_id, sequence)
);
CREATE TABLE snapshot_members (
    run_id text NOT NULL,
    snapshot_id text NOT NULL,
    observation_id text NOT NULL,
    event_sequence bigint NOT NULL,
    PRIMARY KEY (run_id, snapshot_id, observation_id),
    UNIQUE (run_id, snapshot_id, event_sequence),
    FOREIGN KEY (run_id, snapshot_id) REFERENCES snapshots(run_id, snapshot_id),
    FOREIGN KEY (run_id, observation_id, event_sequence)
        REFERENCES observations(run_id, observation_id, event_sequence)
);

CREATE TABLE revisions (
    run_id text NOT NULL,
    revision_id text NOT NULL,
    variable text NOT NULL,
    state_tick bigint NOT NULL CHECK (state_tick >= 0),
    published_at timestamptz NOT NULL,
    sequence_cutoff bigint NOT NULL,
    distribution jsonb NOT NULL,
    model_hash text NOT NULL,
    evidence_snapshot_id text NOT NULL,
    inference_version text NOT NULL,
    previous_revision_id text CHECK (previous_revision_id <> revision_id),
    PRIMARY KEY (run_id, revision_id),
    UNIQUE (run_id, sequence_cutoff),
    UNIQUE (run_id, revision_id, state_tick, sequence_cutoff),
    FOREIGN KEY (run_id, model_hash) REFERENCES runs(run_id, model_hash),
    FOREIGN KEY (run_id, evidence_snapshot_id, sequence_cutoff)
        REFERENCES snapshots(run_id, snapshot_id, sequence_cutoff),
    FOREIGN KEY (run_id, previous_revision_id) REFERENCES revisions(run_id, revision_id)
);
CREATE INDEX revisions_by_time ON revisions (run_id, published_at, sequence_cutoff);

CREATE TABLE forecasts (
    run_id text NOT NULL,
    forecast_id text NOT NULL,
    origin_tick bigint NOT NULL CHECK (origin_tick >= 0),
    target_tick bigint NOT NULL CHECK (target_tick = origin_tick + 1),
    target_state text NOT NULL,
    probability double precision NOT NULL CHECK (probability >= 0 AND probability <= 1),
    issued_at timestamptz NOT NULL,
    belief_revision_id text NOT NULL,
    model_hash text NOT NULL,
    evidence_snapshot_id text NOT NULL,
    PRIMARY KEY (run_id, forecast_id),
    UNIQUE (run_id, belief_revision_id),
    UNIQUE (run_id, forecast_id, origin_tick),
    FOREIGN KEY (run_id, target_state) REFERENCES runs(run_id, target_state),
    FOREIGN KEY (run_id, model_hash) REFERENCES runs(run_id, model_hash),
    FOREIGN KEY (run_id, belief_revision_id) REFERENCES revisions(run_id, revision_id),
    FOREIGN KEY (run_id, evidence_snapshot_id) REFERENCES snapshots(run_id, snapshot_id)
);
CREATE TABLE forecast_closures (
    run_id text NOT NULL,
    origin_tick bigint NOT NULL,
    forecast_id text NOT NULL,
    closed_event_sequence bigint NOT NULL,
    PRIMARY KEY (run_id, origin_tick),
    UNIQUE (run_id, closed_event_sequence),
    FOREIGN KEY (run_id, forecast_id, origin_tick) REFERENCES forecasts(run_id, forecast_id, origin_tick),
    FOREIGN KEY (run_id, closed_event_sequence) REFERENCES events(run_id, sequence)
);

CREATE TABLE run_heads (
    run_id text PRIMARY KEY REFERENCES runs(run_id),
    state_tick bigint NOT NULL,
    sequence bigint NOT NULL,
    revision_id text NOT NULL,
    FOREIGN KEY (run_id, revision_id, state_tick, sequence)
        REFERENCES revisions(run_id, revision_id, state_tick, sequence_cutoff)
);

CREATE FUNCTION reject_history_mutation() RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN
    RAISE EXCEPTION 'table % is append-only', TG_TABLE_NAME USING ERRCODE = '23514';
END;
$$;
DO $$
DECLARE table_name text;
BEGIN
    FOREACH table_name IN ARRAY ARRAY[
        'models', 'runs', 'events', 'observations', 'snapshots', 'snapshot_members',
        'revisions', 'forecasts', 'forecast_closures'
    ] LOOP
        EXECUTE format(
            'CREATE TRIGGER immutable_history BEFORE UPDATE OR DELETE OR TRUNCATE ON %I '
            'FOR EACH STATEMENT EXECUTE FUNCTION reject_history_mutation()', table_name
        );
    END LOOP;
END;
$$;

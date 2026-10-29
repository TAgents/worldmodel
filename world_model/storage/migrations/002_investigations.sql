CREATE TABLE investigations (
    run_id text PRIMARY KEY REFERENCES runs(run_id),
    config jsonb NOT NULL,
    adapter_seed bigint NOT NULL CHECK (adapter_seed >= 0),
    adapter_version text NOT NULL,
    created_at timestamptz NOT NULL
);

CREATE TABLE investigation_steps (
    run_id text NOT NULL REFERENCES investigations(run_id),
    request_id text NOT NULL,
    ordinal bigint NOT NULL CHECK (ordinal >= 0),
    before_sequence bigint NOT NULL,
    after_sequence bigint NOT NULL CHECK (after_sequence >= before_sequence),
    recorded_at timestamptz NOT NULL,
    data jsonb NOT NULL,
    PRIMARY KEY (run_id, request_id),
    UNIQUE (run_id, ordinal),
    FOREIGN KEY (run_id, before_sequence) REFERENCES revisions(run_id, sequence_cutoff),
    FOREIGN KEY (run_id, after_sequence) REFERENCES revisions(run_id, sequence_cutoff)
);
CREATE INDEX investigation_steps_by_time ON investigation_steps (run_id, recorded_at, ordinal);

CREATE TRIGGER immutable_investigation BEFORE UPDATE OR DELETE OR TRUNCATE ON investigations
    FOR EACH STATEMENT EXECUTE FUNCTION reject_history_mutation();
CREATE TRIGGER immutable_investigation_steps BEFORE UPDATE OR DELETE OR TRUNCATE ON investigation_steps
    FOR EACH STATEMENT EXECUTE FUNCTION reject_history_mutation();

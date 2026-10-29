CREATE TABLE run_imports (
    run_id text PRIMARY KEY REFERENCES runs(run_id),
    adapter_version text NOT NULL,
    source_hash text NOT NULL CHECK (source_hash ~ '^[a-f0-9]{64}$'),
    plan_hash text NOT NULL CHECK (plan_hash ~ '^[a-f0-9]{64}$'),
    source_content bytea NOT NULL,
    plan_content bytea NOT NULL,
    completed_sequence bigint NOT NULL CHECK (completed_sequence >= 0)
);

CREATE TRIGGER immutable_history
    BEFORE UPDATE OR DELETE OR TRUNCATE ON run_imports
    FOR EACH STATEMENT EXECUTE FUNCTION reject_history_mutation();

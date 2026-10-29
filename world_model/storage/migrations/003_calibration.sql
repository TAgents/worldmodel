-- The model bytes contain the derivation hashes; this table preserves their
-- exact source bytes independently of any local research archive.
CREATE TABLE model_calibrations (
    model_hash text PRIMARY KEY REFERENCES models(content_hash),
    structure_content bytea NOT NULL,
    revision_content bytea NOT NULL
);

CREATE TRIGGER immutable_history
    BEFORE UPDATE OR DELETE OR TRUNCATE ON model_calibrations
    FOR EACH STATEMENT EXECUTE FUNCTION reject_history_mutation();

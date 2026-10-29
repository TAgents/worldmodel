"""Transactional calibration registration and independent provenance verification."""

from world_model.learning.promotion import calibrated_artifact, verify_calibrated_artifact
from world_model.learning.calibration import CalibrationRevision
from .contracts import ConflictError, NotFoundError, ReplayIntegrityError


class CalibrationStoreMixin:
    @staticmethod
    def _calibration(conn, artifact, registered_at):
        row = conn.execute("SELECT * FROM model_calibrations WHERE model_hash = %s", (artifact.content_hash,)).fetchone()
        if artifact.spec.derivation is None:
            if row is not None:
                raise ReplayIntegrityError("underived model has unexpected calibration evidence")
            return None
        if row is None:
            raise ReplayIntegrityError("calibrated model is missing its evidence")
        try:
            view = verify_calibrated_artifact(artifact, bytes(row["structure_content"]), bytes(row["revision_content"]))
            if view.revision.as_of > registered_at:
                raise ValueError("calibration was unavailable at model registration")
            return view
        except (ValueError, TypeError) as error:
            raise ReplayIntegrityError("stored calibration failed provenance verification") from error

    def _register_calibration(self, conn, structure, revision):
        revision = CalibrationRevision.model_validate(revision)
        artifact, content = calibrated_artifact(structure, revision)
        timestamp = self._now(conn)
        if revision.as_of > timestamp:
            raise ValueError("calibration cutoff must already be available at registration")
        registration = self._register_model(conn, artifact, timestamp)
        conn.execute("""INSERT INTO model_calibrations (model_hash, structure_content, revision_content)
            VALUES (%s, %s, %s) ON CONFLICT DO NOTHING""", (artifact.content_hash, structure.content, content))
        view = self._calibration(conn, artifact, registration.registered_at)
        if view.revision != revision:
            raise ConflictError("model is associated with a different calibration revision")
        return registration

    def register_calibrated_model(self, structure, revision):
        with self._connect() as conn:
            return self._register_calibration(conn, structure, revision)

    def create_calibrated_run(self, structure, revision, *, epoch, target_state, run_id):
        with self._connect() as conn:
            registration = self._register_calibration(conn, structure, revision)
            return self._create_run(conn, registration.content_hash, epoch=epoch, target_state=target_state, run_id=run_id)

    def get_calibration(self, content_hash):
        with self._connect() as conn:
            self._read_transaction(conn)
            artifact, registered_at = self._model(conn, content_hash)
            view = self._calibration(conn, artifact, registered_at)
            if view is None:
                raise NotFoundError("model has no calibration evidence")
            return view

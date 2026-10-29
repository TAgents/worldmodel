"""Derive a fixed model whose bytes bind it to complete calibration evidence."""

from hashlib import sha256

from world_model.core import ModelArtifact
from world_model.core.models import ModelDerivation, ModelSpec
from world_model.core.serialization import canonical_json
from world_model.core.types import Contract
from .calibration import CalibrationRevision, verify_revision


class CalibrationView(Contract):
    derivation: ModelDerivation
    structure: ModelSpec
    revision: CalibrationRevision


def calibrated_artifact(structure: ModelArtifact, revision: CalibrationRevision):
    if structure.spec.derivation is not None:
        raise ValueError("calibration must start from an underived neutral structure")
    revision = verify_revision(structure, revision, allow_interval_roundoff=True)
    content = canonical_json(revision.model_dump(mode="json"))
    digest = sha256(content).hexdigest()
    raw = structure.spec.model_dump(mode="json")
    raw["model_id"] = "calibrated-mean-" + structure.content_hash[:24]
    raw["model_version"] = digest
    for cell in revision.cells:
        raw["observation_models"][cell.source_id]["alert_probability"][cell.state] = cell.mean
    raw["derivation"] = ModelDerivation(method="beta-posterior-mean-v1",
        calibration_revision=revision.revision_id, calibration_content_hash=digest,
        structure_hash=structure.content_hash, as_of=revision.as_of,
        evidence_count=len(revision.evidence)).model_dump(mode="json")
    return ModelArtifact(canonical_json(raw)), content


def verify_calibrated_artifact(artifact, structure_content, revision_content):
    if artifact.spec.derivation is None:
        raise ValueError("model has no calibration derivation")
    if sha256(revision_content).hexdigest() != artifact.spec.derivation.calibration_content_hash:
        raise ValueError("calibration evidence bytes do not match the model")
    structure = ModelArtifact(structure_content)
    revision = CalibrationRevision.model_validate_json(revision_content)
    expected, _ = calibrated_artifact(structure, revision)
    if expected.content != artifact.content:
        raise ValueError("model does not match its calibration evidence and structure")
    return CalibrationView(derivation=artifact.spec.derivation, structure=structure.spec, revision=revision)

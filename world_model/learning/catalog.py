"""Read completed local research archives without exposing arbitrary file paths."""

from hashlib import sha256
import json
from pathlib import Path
import re

from world_model.core import ModelArtifact
from world_model.core.serialization import canonical_json
from world_model.storage.contracts import NotFoundError
from .calibration import CalibrationRevision, verify_revision


class ResearchCatalog:
    def __init__(self, root):
        self.root = Path(root).resolve()

    def _folder(self, name):
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", name):
            raise NotFoundError("research archive not found")
        folder = (self.root / name).resolve()
        if folder.parent != self.root or not folder.is_dir():
            raise NotFoundError("research archive not found")
        return folder

    @staticmethod
    def _bytes(folder, name):
        path = (folder / name).resolve()
        if path.parent != folder or not path.is_file():
            raise ValueError("research archive is incomplete or contains an external file link")
        return path.read_bytes()

    def _load(self, name):
        folder = self._folder(name)
        try:
            manifest = json.loads(self._bytes(folder, "manifest.json"))
            if manifest.get("protocol") != "sensor-reliability-v1":
                raise NotFoundError("unsupported research archive")
            def checked(filename):
                content = self._bytes(folder, filename)
                if sha256(content).hexdigest() != manifest["files"][filename]:
                    raise ValueError("research archive checksum mismatch")
                return content
            report = json.loads(checked("report.json"))
            protocol = json.loads(checked("protocol.json"))
            structure = ModelArtifact(checked("structure.json"))
            revisions = [CalibrationRevision.model_validate(r) for r in json.loads(checked("calibration-revisions.json"))]
            if report["protocol_sha256"] != sha256(canonical_json(protocol)).hexdigest() or report["structure_hash"] != structure.content_hash or not revisions or report["calibration_revision"] != revisions[-1].revision_id:
                raise ValueError("research archive provenance mismatch")
            for revision in revisions:
                verify_revision(structure, revision)
            return ({"name": name, "report": report, "protocol": protocol,
                    "revisions": [r.model_dump(mode="json") for r in revisions], "runtime": manifest["runtime"]}, structure, revisions)
        except (KeyError, TypeError, json.JSONDecodeError) as error:
            raise ValueError("research archive metadata is invalid") from error

    def load(self, name):
        return self._load(name)[0]

    def calibration(self, name, revision_id):
        _, structure, revisions = self._load(name)
        revision = next((r for r in revisions if r.revision_id == revision_id), None)
        if revision is None:
            raise NotFoundError("calibration snapshot not found in this archive")
        return structure, revision

    def list(self):
        if not self.root.is_dir():
            return []
        result = []
        for path in sorted(self.root.iterdir()):
            if not path.is_dir() or path.is_symlink() or not (path / "manifest.json").is_file():
                continue
            try:
                manifest = json.loads(self._bytes(path, "manifest.json"))
                if manifest.get("protocol") != "sensor-reliability-v1":
                    continue
                item = self.load(path.name)
                result.append({"name": path.name, "split": item["report"]["split"], "available": True})
            except (ValueError, NotFoundError, OSError):
                result.append({"name": path.name, "available": False, "error": "Archive is incomplete or failed verification."})
        return result

from unittest.mock import Mock

from fastapi.testclient import TestClient
import pytest

from world_model.api import create_app
from world_model.learning.catalog import ResearchCatalog
from world_model.storage.contracts import NotFoundError
from world_model.storage.store import PostgresStore
from .learning_fixtures import write_learning_fixture


def test_completed_archive_exposes_parameter_revisions_with_integrity_checks(tmp_path):
    folder, expected = write_learning_fixture(tmp_path)
    catalog = ResearchCatalog(tmp_path)
    assert catalog.list() == [{"name": folder.name, "split": "development", "available": True}]
    assert catalog.load(folder.name) == expected
    (folder / "report.json").write_text('{"modified": true}')
    with pytest.raises(ValueError, match="checksum"):
        catalog.load(folder.name)
    assert catalog.list()[0]["available"] is False


def test_catalog_does_not_follow_external_paths_or_links(tmp_path):
    root = tmp_path / "archives"
    folder, _ = write_learning_fixture(root)
    catalog = ResearchCatalog(root)
    for name in ("..", "../archives", "/etc", "a/b"):
        with pytest.raises(NotFoundError):
            catalog.load(name)
    external = tmp_path / "external"
    external.mkdir()
    (root / "linked").symlink_to(external, target_is_directory=True)
    with pytest.raises(NotFoundError):
        catalog.load("linked")
    (folder / "report.json").unlink()
    (external / "report.json").write_text('{}')
    (folder / "report.json").symlink_to(external / "report.json")
    with pytest.raises(ValueError, match="external"):
        catalog.load(folder.name)


def test_learning_api_is_read_only_and_does_not_need_database_access(tmp_path):
    folder, expected = write_learning_fixture(tmp_path)
    store = Mock(spec=PostgresStore)
    with TestClient(create_app(store, research_directory=tmp_path), base_url="http://127.0.0.1") as client:
        page = client.get("/learning")
        assert page.status_code == 200
        assert "Calibration Lab" in page.text
        assert "script-src 'self'" in page.headers["content-security-policy"]
        assert client.get("/assets/learning.js").status_code == 200
        assert client.get("/assets/learning.css").status_code == 200
        assert client.get("/research").json()[0]["name"] == folder.name
        assert client.get(f"/research/{folder.name}").json() == expected
        assert client.get("/research/missing").status_code == 404
        assert client.post("/research", json={}).status_code == 405
        (folder / "calibration-revisions.json").write_text('[]')
        assert client.get(f"/research/{folder.name}").status_code == 422
    assert not store.mock_calls

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client(tmp_path, monkeypatch):
    """Fresh app + isolated DB per test, with data materialized before requests."""
    db_path = str(tmp_path / "serving_test.db")
    monkeypatch.setenv("FEATURE_STORE_DB_PATH", db_path)

    # Import after setting the env var so the module-level store picks it up.
    import importlib
    import feature_store_lite.serving as serving_module
    importlib.reload(serving_module)

    from feature_store_lite.registry_app import build_registry, RAW_LOGINS, RAW_SIGNUP

    registry = build_registry()
    serving_module._store.materialize(registry.get("rolling_login_count"), RAW_LOGINS)
    serving_module._store.materialize(registry.get("has_signed_up"), RAW_SIGNUP)

    return TestClient(serving_module.app)


def test_health_check(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_list_views_returns_registered_views(client):
    response = client.get("/views")
    assert response.status_code == 200
    names = {v["name"] for v in response.json()}
    assert names == {"rolling_login_count", "has_signed_up"}


def test_online_features_returns_latest_values_for_known_entity(client):
    response = client.get("/features/online/u1")
    assert response.status_code == 200
    body = response.json()
    assert body["entity_id"] == "u1"
    assert body["features"]["rolling_login_count"]["value"] == 3
    assert body["features"]["has_signed_up"]["value"] == 1


def test_online_features_404_for_unknown_entity(client):
    response = client.get("/features/online/does-not-exist")
    assert response.status_code == 404

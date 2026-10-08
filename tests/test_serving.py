import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client(tmp_path, monkeypatch):
    """Fresh app + isolated DB per test, with data materialized before requests.
    Auth and rate limiting are left at their default (disabled / generous) so
    existing behavioral tests don't need to juggle headers."""
    db_path = str(tmp_path / "serving_test.db")
    monkeypatch.setenv("FEATURE_STORE_DB_PATH", db_path)
    monkeypatch.delenv("FEATURE_STORE_API_KEY", raising=False)
    monkeypatch.setenv("FEATURE_STORE_RATE_LIMIT", "1000")

    # Import after setting the env var so the module-level store picks it up.
    import importlib
    import feature_store_lite.serving as serving_module
    importlib.reload(serving_module)

    from feature_store_lite.registry_app import build_registry, RAW_LOGINS, RAW_SIGNUP

    registry = build_registry()
    serving_module._store.materialize(registry.get("rolling_login_count"), RAW_LOGINS)
    serving_module._store.materialize(registry.get("has_signed_up"), RAW_SIGNUP)

    return TestClient(serving_module.app)


@pytest.fixture
def auth_client(tmp_path, monkeypatch):
    """Same setup, but with FEATURE_STORE_API_KEY set, to test the auth gate."""
    db_path = str(tmp_path / "serving_auth_test.db")
    monkeypatch.setenv("FEATURE_STORE_DB_PATH", db_path)
    monkeypatch.setenv("FEATURE_STORE_API_KEY", "secret123")
    monkeypatch.setenv("FEATURE_STORE_RATE_LIMIT", "1000")

    import importlib
    import feature_store_lite.serving as serving_module
    importlib.reload(serving_module)

    from feature_store_lite.registry_app import build_registry, RAW_LOGINS

    registry = build_registry()
    serving_module._store.materialize(registry.get("rolling_login_count"), RAW_LOGINS)

    return TestClient(serving_module.app)


@pytest.fixture
def limited_client(tmp_path, monkeypatch):
    """Same setup, but with a tiny rate limit, to test the 429 path."""
    db_path = str(tmp_path / "serving_limited_test.db")
    monkeypatch.setenv("FEATURE_STORE_DB_PATH", db_path)
    monkeypatch.delenv("FEATURE_STORE_API_KEY", raising=False)
    monkeypatch.setenv("FEATURE_STORE_RATE_LIMIT", "2")

    import importlib
    import feature_store_lite.serving as serving_module
    importlib.reload(serving_module)

    from feature_store_lite.registry_app import build_registry, RAW_LOGINS

    registry = build_registry()
    serving_module._store.materialize(registry.get("rolling_login_count"), RAW_LOGINS)

    return TestClient(serving_module.app)


def test_health_check(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert response.json()["auth_enabled"] is False


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


def test_auth_rejects_missing_api_key(auth_client):
    response = auth_client.get("/features/online/u1")
    assert response.status_code == 401


def test_auth_rejects_wrong_api_key(auth_client):
    response = auth_client.get("/features/online/u1", headers={"x-api-key": "wrong"})
    assert response.status_code == 401


def test_auth_accepts_correct_api_key(auth_client):
    response = auth_client.get("/features/online/u1", headers={"x-api-key": "secret123"})
    assert response.status_code == 200


def test_health_check_bypasses_auth(auth_client):
    response = auth_client.get("/health")
    assert response.status_code == 200
    assert response.json()["auth_enabled"] is True


def test_rate_limit_returns_429_after_threshold(limited_client):
    # FEATURE_STORE_RATE_LIMIT=2 for this client; third request in the window
    # should be rejected.
    r1 = limited_client.get("/views")
    r2 = limited_client.get("/views")
    r3 = limited_client.get("/views")
    assert r1.status_code == 200
    assert r2.status_code == 200
    assert r3.status_code == 429

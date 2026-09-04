"""Online-serving API.

A small FastAPI app that reads from the same FeatureStore + registry used offline,
so a model service can fetch features at inference time the way it would in a real
deployment: no separately maintained "serving logic" to drift out of sync with what
was used to build the training set.

Run: uvicorn feature_store_lite.serving:app --reload
"""

import os
from fastapi import FastAPI, HTTPException

from .registry_app import build_registry
from .store import FeatureStore

app = FastAPI(
    title="FeatureStoreLite Serving API",
    description="Online feature lookups backed by the same store used for offline training sets.",
    version="0.1.0",
)

_DB_PATH = os.environ.get("FEATURE_STORE_DB_PATH", "feature_store.db")
_registry = build_registry()
_store = FeatureStore(db_path=_DB_PATH)


@app.get("/views")
def list_views():
    """List registered feature views and their descriptions."""
    return [
        {"name": view.name, "entity_column": view.entity_column, "description": view.description}
        for view in _registry.all()
    ]


@app.get("/features/online/{entity_id}")
def get_online_features(entity_id: str):
    """Return the latest known value (and as-of timestamp) for every feature
    registered for this entity. 404s if the entity has never been materialized,
    rather than silently returning an empty dict."""
    features = _store.get_online_features(entity_id)
    if not features:
        raise HTTPException(
            status_code=404,
            detail=f"No materialized features found for entity_id='{entity_id}'. "
            "Has it been materialized yet? Try POST-ing via the CLI first.",
        )
    return {"entity_id": entity_id, "features": features}


@app.get("/health")
def health():
    return {"status": "ok", "db_path": _DB_PATH}

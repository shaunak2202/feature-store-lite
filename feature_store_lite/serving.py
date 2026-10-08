"""Online-serving API.

A small FastAPI app that reads from the same FeatureStore + registry used offline,
so a model service can fetch features at inference time the way it would in a real
deployment: no separately maintained "serving logic" to drift out of sync with what
was used to build the training set.

Auth: if FEATURE_STORE_API_KEY is set in the environment, every request other than
/health must send a matching `x-api-key` header or gets a 401. If unset, auth is a
no-op (keeps local/demo use friction-free).

Rate limiting: a simple in-memory fixed-window limiter caps requests per client IP
(FEATURE_STORE_RATE_LIMIT per minute, default 60). See rate_limit.py for caveats.

Run: uvicorn feature_store_lite.serving:app --reload
"""

import os
from fastapi import FastAPI, HTTPException, Request

from .registry_app import build_registry
from .store import FeatureStore
from .rate_limit import FixedWindowRateLimiter

app = FastAPI(
    title="FeatureStoreLite Serving API",
    description="Online feature lookups backed by the same store used for offline training sets.",
    version="0.2.0",
)

_DB_PATH = os.environ.get("FEATURE_STORE_DB_PATH", "feature_store.db")
_API_KEY = os.environ.get("FEATURE_STORE_API_KEY")  # None => auth disabled
_RATE_LIMIT = int(os.environ.get("FEATURE_STORE_RATE_LIMIT", "60"))

_registry = build_registry()
_store = FeatureStore(db_path=_DB_PATH)
_limiter = FixedWindowRateLimiter(max_requests=_RATE_LIMIT, window_seconds=60)


@app.middleware("http")
async def auth_and_rate_limit(request: Request, call_next):
    if request.url.path == "/health":
        return await call_next(request)

    if _API_KEY is not None:
        provided = request.headers.get("x-api-key")
        if provided != _API_KEY:
            return _json_error(401, "missing or invalid x-api-key header")

    client_key = request.client.host if request.client else "unknown"
    if not _limiter.allow(client_key):
        return _json_error(429, f"rate limit exceeded ({_RATE_LIMIT} requests/minute)")

    return await call_next(request)


def _json_error(status_code: int, detail: str):
    from fastapi.responses import JSONResponse

    return JSONResponse(status_code=status_code, content={"detail": detail})


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
    if not entity_id.strip():
        raise HTTPException(status_code=422, detail="entity_id must not be empty")

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
    return {
        "status": "ok",
        "db_path": _DB_PATH,
        "auth_enabled": _API_KEY is not None,
        "rate_limit_per_minute": _RATE_LIMIT,
    }

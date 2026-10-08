# FeatureStoreLite

A small, from-scratch feature store for tabular ML pipelines. Built to understand (and
solve) the core problems real feature stores solve at a scale where tools like Feast
or Tecton are overkill: **defining features once**, **avoiding training/serving skew**,
and **doing point-in-time-correct joins** so training data doesn't leak the future.

## Why

In most ML projects, feature logic gets duplicated: once in a notebook/training
pipeline, once in whatever serves predictions online. They drift apart. Worse, naive
joins between an entity table and a feature log ignore *when* a feature value was
known, silently leaking future information into training sets. FeatureStoreLite is a
small, readable implementation of the pattern real feature stores use to avoid both
problems, so it's something I can actually explain line-by-line in an interview
rather than a black box.

## Core ideas

- **Feature definitions as code**: a `FeatureView` is a Python function + schema that
  computes one or more feature columns from raw source data. Define it once, use it
  for both offline training sets and online serving.
- **Offline store, two backends**: materialized feature values are written to an
  append-only feature log, either in a local SQLite table (`feature_values`, the
  default) or as partitioned Parquet files under a directory
  (`ParquetOfflineStore`), for when the offline log gets large enough that
  row-by-row SQLite writes stop being the right tool. Both implement the same
  `write`/`read` interface, so `point_in_time_join` doesn't care which one produced
  the DataFrame it's given.
- **Point-in-time joins**: given a table of `(entity_id, label_timestamp)` rows (an
  "entity spine"), the store returns, for each row, the most recent feature value
  known *at or before* that timestamp for that entity -- never a value from the
  future. This is the part naive `pandas.merge` gets wrong, and the part this repo
  has the most tests around. `examples/naive_vs_pit.py` makes the difference visible
  on real numbers, not just an assertion.
- **Online store**: a simple latest-value cache (SQLite, could be swapped for Redis)
  that always holds the freshest value per `(entity_id, feature_name)` for
  low-latency single-row lookups at inference time.
- **Serving API**: a FastAPI app (`feature_store_lite/serving.py`) exposing
  `GET /features/online/{entity_id}` and `GET /views` for real-time lookups, so a
  model service can fetch features the same way it would in a real deployment. It
  now supports optional API-key auth and a basic in-memory rate limiter, the two
  things flagged in the previous session as missing for anything beyond a demo.
- **Materialization CLI**: `python -m feature_store_lite.cli materialize --view <name>`
  runs a registered view's compute function against its raw source and writes the
  result into both stores, the same operation a scheduled batch job would run. The
  CLI now validates its own arguments more defensively and reports errors without
  stack traces.

## Project layout

```
feature_store_lite/
  __init__.py
  store.py            # SQLite-backed offline + online storage engine
  parquet_store.py    # Parquet-backed offline store (alternate backend)
  feature_view.py     # FeatureView definition + registry
  pit_join.py          # point-in-time-correct join logic
  serving.py           # FastAPI online-serving app (+ auth, rate limiting)
  cli.py               # materialization CLI
  registry_app.py      # shared registry + raw-data sources used by CLI and serving
examples/
  toy_pipeline.py      # end-to-end example: define features, materialize, query
  naive_vs_pit.py      # side-by-side naive join vs point-in-time join on the same data
  parquet_backend.py   # same toy pipeline, but writing the offline log to Parquet
tests/
  test_pit_join.py
  test_store.py
  test_serving.py
  test_cli.py
  test_parquet_store.py
requirements.txt
README.md
```

## Quickstart

```bash
pip install -r requirements.txt

# Toy end-to-end walkthrough
python examples/toy_pipeline.py

# See the skew problem on real numbers
python examples/naive_vs_pit.py

# Same pipeline, writing the offline log to Parquet instead of SQLite
python examples/parquet_backend.py

# Materialize a registered view via CLI (writes to feature_store.db)
python -m feature_store_lite.cli materialize --view rolling_login_count
python -m feature_store_lite.cli list-views

# Run the online-serving API
uvicorn feature_store_lite.serving:app --reload
# then: curl http://127.0.0.1:8000/features/online/u1

# With auth enabled (set before starting uvicorn):
export FEATURE_STORE_API_KEY=some-secret
curl -H "x-api-key: some-secret" http://127.0.0.1:8000/features/online/u1

pytest
```

The toy pipeline defines two features on a fake "user activity" dataset (rolling
login count, has-signed-up flag), materializes them into a local `feature_store.db`
SQLite file, and then runs a point-in-time join against a small "labels" table to
build a training set.

`naive_vs_pit.py` takes that same data and builds the training set two ways: a naive
`pandas.merge` that just grabs the latest value per entity regardless of timestamp,
and the point-in-time join. It prints both side by side so the leakage is visible
in the actual numbers, not just an assertion in a test.

## Offline store backends

`feature_store_lite/store.py`'s `FeatureStore` writes the offline log to SQLite by
default -- fine for a demo or a dataset that fits comfortably in a single file.
`feature_store_lite/parquet_store.py` adds `ParquetOfflineStore`, a second backend
with the same shape (`write(feature_name, df)` / `read(feature_name=None)`) that
instead writes one Parquet file per `feature_name` under a directory, using pandas'
built-in Parquet support (via `pyarrow`). This is the natural next step once an
offline log is too large to comfortably query row-by-row from SQLite: Parquet reads
can be columnar and partitioned, and the files can be handed directly to a Spark /
DuckDB / pandas job without going through a database driver at all.

`examples/parquet_backend.py` runs the same toy pipeline as `toy_pipeline.py` but
materializes into `ParquetOfflineStore` instead, and shows that `point_in_time_join`
works identically against the resulting DataFrame -- the join logic has no idea
which backend produced its input, which is the point: swapping storage backends
shouldn't change feature semantics.

## Online serving

`feature_store_lite/serving.py` is a small FastAPI app backed by the same
`FeatureStore` used offline:

- `GET /views` -- lists registered feature views and their descriptions.
- `GET /features/online/{entity_id}` -- returns the latest known value (and its
  `as_of` timestamp) for every feature registered for that entity, from the same
  `feature_values_latest` table the CLI materializes into. This is the piece that
  closes the loop: the exact same `FeatureView.compute_fn` that built the training
  set is what populated the values this endpoint serves, so there's no separate
  "serving logic" to drift out of sync with training logic.

**Auth**: if the `FEATURE_STORE_API_KEY` environment variable is set, every request
(other than `/health`) must include a matching `x-api-key` header, or the API
returns `401`. If the env var is unset, auth is a no-op -- the default stays
friction-free for local demo use, but the gate exists for anything that'd actually
be deployed.

**Rate limiting**: a small in-memory fixed-window limiter
(`feature_store_lite/rate_limit.py`) caps requests per client IP
(`FEATURE_STORE_RATE_LIMIT`, default 60/minute). It's intentionally simple --
in-process, not distributed, reset on restart -- a real deployment behind multiple
workers would need Redis or a proxy-level limiter instead. It's here to make the
concept concrete and testable, not to be production-grade.

## Materialization CLI

`feature_store_lite/cli.py` wraps the registry + store into commands a scheduler
(cron, Airflow, etc.) could call directly:

```bash
python -m feature_store_lite.cli list-views
python -m feature_store_lite.cli materialize --view rolling_login_count
python -m feature_store_lite.cli materialize --all
python -m feature_store_lite.cli materialize --view rolling_login_count --backend parquet
```

Views and their raw data sources are registered once in
`feature_store_lite/registry_app.py`, which both the CLI and the serving app import,
so there is exactly one definition of each feature used everywhere. The CLI now
rejects `--view` and `--all` being passed together, rejects an empty `--view ""`,
and exits with a clean error message (not a traceback) on every validated failure
path, all covered by `tests/test_cli.py`.

## Status

This is now feature-complete for what was scoped: storage engine (two backends),
`FeatureView` API, point-in-time join, toy + comparison + Parquet examples,
FastAPI serving layer with auth and rate limiting, and a materialization CLI with
defensive argument handling -- all covered by tests (`pytest`, 30+ test cases
across join logic, both storage backends, the CLI, and the serving API).

## Roadmap (closed out this session)

- [x] Storage engine (offline log + online latest-value table)
- [x] `FeatureView` definition API + registry
- [x] Point-in-time join with tests
- [x] Toy end-to-end example
- [x] Naive-vs-PIT comparison script to make the skew concrete
- [x] FastAPI serving layer for online lookups
- [x] Materialization CLI (`featurestore materialize --view rolling_login_count`)
- [x] Parquet-backed offline store option for larger datasets
- [x] Basic auth / rate limiting on the serving API

No open roadmap items remain. Further work (not planned) would be things like a
streaming ingestion path or a distributed online store -- genuinely out of scope
for a project whose value is in being small enough to read end to end.

## Honest scope

This is a learning-grade reimplementation of ideas from real feature stores (Feast,
Tecton, Uber Michelangelo's Palette), not a production system: no distributed
storage, no streaming ingestion, auth is a single shared API key rather than
per-client credentials, and the rate limiter is in-process and resets on restart.
The value is in the point-in-time join correctness, the clean separation between
definition/materialize/serve, having two interchangeable offline storage backends,
and a small enough codebase to reason about and extend.

---

Built by a personal automation project Shaunak set up: it uses Claude to design and write real, working code within his actual skill set, and pushes it here on a regular schedule as an ongoing practice/portfolio project.

---

Built by a personal automation project Shaunak set up: it uses Claude to design and write real, working code within his actual skill set, and pushes it here on a regular schedule as an ongoing practice/portfolio project.

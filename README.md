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
- **Offline store**: materialized feature values are written to a local SQLite table
  (`feature_values`) keyed by `(entity_id, feature_name, event_timestamp, value)`,
  functioning like an append-only feature log.
- **Point-in-time joins**: given a table of `(entity_id, label_timestamp)` rows (an
  "entity spine"), the store returns, for each row, the most recent feature value
  known *at or before* that timestamp for that entity -- never a value from the
  future. This is the part naive `pandas.merge` gets wrong, and the part this repo
  has the most tests around. `examples/naive_vs_pit.py` makes the difference visible
  on real numbers, not just an assertion.
- **Online store**: a simple latest-value cache (also SQLite, could be swapped for
  Redis) that always holds the freshest value per `(entity_id, feature_name)` for
  low-latency single-row lookups at inference time.
- **Serving API**: a FastAPI app (`feature_store_lite/serving.py`) exposing
  `GET /features/online/{entity_id}` and `GET /views` for real-time lookups, so a
  model service can fetch features the same way it would in a real deployment.
- **Materialization CLI**: `python -m feature_store_lite.cli materialize --view <name>`
  runs a registered view's compute function against its raw source and writes the
  result into both stores, the same operation a scheduled batch job would run.

## Project layout

```
feature_store_lite/
  __init__.py
  store.py          # SQLite-backed offline + online storage engine
  feature_view.py   # FeatureView definition + registry
  pit_join.py        # point-in-time-correct join logic
  serving.py         # FastAPI online-serving app
  cli.py             # materialization CLI
  registry_app.py    # shared registry + raw-data sources used by CLI and serving
examples/
  toy_pipeline.py    # end-to-end example: define features, materialize, query
  naive_vs_pit.py    # side-by-side naive join vs point-in-time join on the same data
tests/
  test_pit_join.py
  test_store.py
  test_serving.py
  test_cli.py
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

# Materialize a registered view via CLI (writes to feature_store.db)
python -m feature_store_lite.cli materialize --view rolling_login_count
python -m feature_store_lite.cli list-views

# Run the online-serving API
uvicorn feature_store_lite.serving:app --reload
# then: curl http://127.0.0.1:8000/features/online/u1

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

## Materialization CLI

`feature_store_lite/cli.py` wraps the registry + store into commands a scheduler
(cron, Airflow, etc.) could call directly:

```bash
python -m feature_store_lite.cli list-views
python -m feature_store_lite.cli materialize --view rolling_login_count
python -m feature_store_lite.cli materialize --all
```

Views and their raw data sources are registered once in
`feature_store_lite/registry_app.py`, which both the CLI and the serving app import,
so there is exactly one definition of each feature used everywhere.

## Status

Core engine, point-in-time join, storage, online-serving API, and materialization
CLI are all in and tested. Remaining before this is fully "done": a Parquet-backed
offline store option for larger-than-SQLite datasets, and a bit more polish on CLI
error handling. Planned as the last milestone for this project.

## Roadmap

- [x] Storage engine (offline log + online latest-value table)
- [x] `FeatureView` definition API + registry
- [x] Point-in-time join with tests
- [x] Toy end-to-end example
- [x] Naive-vs-PIT comparison script to make the skew concrete
- [x] FastAPI serving layer for online lookups
- [x] Materialization CLI (`featurestore materialize --view rolling_login_count`)
- [ ] Parquet-backed offline store option for larger datasets
- [ ] Basic auth / rate limiting on the serving API (would be needed for anything
      beyond a portfolio demo)

## Honest scope

This is a learning-grade reimplementation of ideas from real feature stores (Feast,
Tecton, Uber Michelangelo's Palette), not a production system -- no distributed
storage, no streaming ingestion, no auth on the API. The value is in the
point-in-time join correctness, the clean separation between definition/materialize/
serve, and having a small enough codebase to reason about and extend.

---

Built by a personal automation project Shaunak set up: it uses Claude to design and write real, working code within his actual skill set, and pushes it here on a regular schedule as an ongoing practice/portfolio project.

---

Built by a personal automation project Shaunak set up: it uses Claude to design and write real, working code within his actual skill set, and pushes it here on a regular schedule as an ongoing practice/portfolio project.

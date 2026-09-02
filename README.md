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
  has the most tests around.
- **Online store**: a simple latest-value cache (also SQLite, could be swapped for
  Redis) that always holds the freshest value per `(entity_id, feature_name)` for
  low-latency single-row lookups at inference time.
- **Serving API**: a small FastAPI app exposing `/features/online/{entity_id}` for
  real-time lookups, so a model service can fetch features the same way it would in
  a real deployment.

## Status

Work in progress. This first pass has the core storage engine, the feature
definition API, the point-in-time join logic (with tests), and a toy end-to-end
example. Still to come: the FastAPI serving layer wiring, a materialization CLI, and
a worked example comparing a naive join vs. the point-in-time-correct join on the
same data so the skew problem is visible, not just asserted.

## Project layout

```
feature_store_lite/
  __init__.py
  store.py          # SQLite-backed offline + online storage engine
  feature_view.py   # FeatureView definition + registry
  pit_join.py        # point-in-time-correct join logic
examples/
  toy_pipeline.py    # end-to-end example: define features, materialize, query
tests/
  test_pit_join.py
  test_store.py
requirements.txt
README.md
```

## Quickstart

```bash
pip install -r requirements.txt
python examples/toy_pipeline.py
pytest
```

The toy pipeline defines two features on a fake "user activity" dataset (rolling
login count, days since signup), materializes them into a local `feature_store.db`
SQLite file, and then runs a point-in-time join against a small "labels" table to
build a training set -- printing before/after to show what a naive join would have
gotten wrong.

## Roadmap

- [x] Storage engine (offline log + online latest-value table)
- [x] `FeatureView` definition API + registry
- [x] Point-in-time join with tests
- [x] Toy end-to-end example
- [ ] FastAPI serving layer for online lookups
- [ ] Materialization CLI (`featurestore materialize --view rolling_logins`)
- [ ] Naive-vs-PIT comparison notebook/script to make the skew concrete
- [ ] Parquet-backed offline store option for larger datasets

## Honest scope

This is a learning-grade reimplementation of ideas from real feature stores (Feast,
Tecton, Uber Michelangelo's Palette), not a production system -- no distributed
storage, no streaming ingestion, no auth. The value is in the point-in-time join
correctness and having a small enough codebase to reason about and extend.

---

Built by a personal automation project Shaunak set up: it uses Claude to design and write real, working code within his actual skill set, and pushes it here on a regular schedule as an ongoing practice/portfolio project.

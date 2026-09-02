import os
import sys
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from feature_store_lite import FeatureView, FeatureStore


@pytest.fixture
def tmp_store(tmp_path):
    db_path = tmp_path / "test_store.db"
    return FeatureStore(db_path=str(db_path))


def simple_compute(raw_df: pd.DataFrame) -> pd.DataFrame:
    df = raw_df.copy()
    df["feature_name"] = "score"
    return df[["entity_id", "event_timestamp", "feature_name", "value"]]


def test_materialize_writes_offline_log(tmp_store):
    view = FeatureView(name="score", entity_column="entity_id", compute_fn=simple_compute)
    raw = pd.DataFrame(
        [
            {"entity_id": "u1", "event_timestamp": "2024-01-01", "value": 10},
            {"entity_id": "u1", "event_timestamp": "2024-01-02", "value": 20},
        ]
    )
    written = tmp_store.materialize(view, raw)
    assert written == 2

    log = tmp_store.get_offline_log()
    assert len(log) == 2


def test_online_store_keeps_only_latest_value(tmp_store):
    view = FeatureView(name="score", entity_column="entity_id", compute_fn=simple_compute)
    raw = pd.DataFrame(
        [
            {"entity_id": "u1", "event_timestamp": "2024-01-01", "value": 10},
            {"entity_id": "u1", "event_timestamp": "2024-01-02", "value": 20},
        ]
    )
    tmp_store.materialize(view, raw)
    online = tmp_store.get_online_features("u1")
    assert online["score"]["value"] == 20


def test_materializing_twice_does_not_duplicate_offline_rows(tmp_store):
    view = FeatureView(name="score", entity_column="entity_id", compute_fn=simple_compute)
    raw = pd.DataFrame([{"entity_id": "u1", "event_timestamp": "2024-01-01", "value": 10}])
    tmp_store.materialize(view, raw)
    tmp_store.materialize(view, raw)
    log = tmp_store.get_offline_log()
    assert len(log) == 1


def test_unregistered_view_raises_on_missing_columns(tmp_store):
    def bad_compute(raw_df):
        return raw_df  # missing required columns

    view = FeatureView(name="bad", entity_column="entity_id", compute_fn=bad_compute)
    raw = pd.DataFrame([{"entity_id": "u1"}])
    with pytest.raises(ValueError):
        tmp_store.materialize(view, raw)

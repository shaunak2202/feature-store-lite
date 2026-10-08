import os
import sys
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from feature_store_lite.parquet_store import ParquetOfflineStore
from feature_store_lite.pit_join import point_in_time_join


@pytest.fixture
def tmp_pq_store(tmp_path):
    return ParquetOfflineStore(base_path=str(tmp_path / "pq"))


def make_computed():
    return pd.DataFrame(
        [
            {"entity_id": "u1", "feature_name": "count", "event_timestamp": "2024-01-01", "value": 1},
            {"entity_id": "u1", "feature_name": "count", "event_timestamp": "2024-01-05", "value": 2},
        ]
    )


def test_write_then_read_round_trips(tmp_pq_store):
    written = tmp_pq_store.write("count", make_computed())
    assert written == 2

    log = tmp_pq_store.read(feature_name="count")
    assert len(log) == 2
    assert set(log["value"]) == {1, 2}


def test_read_unknown_feature_returns_empty_frame(tmp_pq_store):
    log = tmp_pq_store.read(feature_name="does_not_exist")
    assert log.empty
    assert list(log.columns) == ["entity_id", "feature_name", "event_timestamp", "value"]


def test_writing_twice_does_not_duplicate_rows(tmp_pq_store):
    tmp_pq_store.write("count", make_computed())
    tmp_pq_store.write("count", make_computed())
    log = tmp_pq_store.read(feature_name="count")
    assert len(log) == 2


def test_read_combines_multiple_features(tmp_pq_store):
    tmp_pq_store.write("count", make_computed())
    other = pd.DataFrame(
        [{"entity_id": "u2", "feature_name": "other", "event_timestamp": "2024-01-01", "value": 9}]
    )
    tmp_pq_store.write("other", other)

    combined = tmp_pq_store.read()
    assert set(combined["feature_name"].unique()) == {"count", "other"}


def test_known_feature_names(tmp_pq_store):
    tmp_pq_store.write("count", make_computed())
    assert tmp_pq_store.known_feature_names() == ["count"]


def test_point_in_time_join_works_against_parquet_backed_log(tmp_pq_store):
    tmp_pq_store.write("count", make_computed())
    log = tmp_pq_store.read()

    spine = pd.DataFrame([{"entity_id": "u1", "label_timestamp": "2024-01-02"}])
    result = point_in_time_join(spine, log)
    assert result.iloc[0]["count"] == 1

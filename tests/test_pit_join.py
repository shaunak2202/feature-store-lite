import pandas as pd
import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from feature_store_lite.pit_join import point_in_time_join


def make_log():
    return pd.DataFrame(
        [
            {"entity_id": "u1", "feature_name": "count", "event_timestamp": "2024-01-01", "value": 1},
            {"entity_id": "u1", "feature_name": "count", "event_timestamp": "2024-01-05", "value": 2},
            {"entity_id": "u1", "feature_name": "count", "event_timestamp": "2024-01-10", "value": 3},
            {"entity_id": "u2", "feature_name": "count", "event_timestamp": "2024-01-05", "value": 5},
        ]
    )


def test_returns_most_recent_value_at_or_before_timestamp():
    spine = pd.DataFrame([{"entity_id": "u1", "label_timestamp": "2024-01-06"}])
    result = point_in_time_join(spine, make_log())
    assert result.iloc[0]["count"] == 2


def test_does_not_leak_future_values():
    spine = pd.DataFrame([{"entity_id": "u1", "label_timestamp": "2024-01-02"}])
    result = point_in_time_join(spine, make_log())
    assert result.iloc[0]["count"] == 1


def test_exact_timestamp_match_is_inclusive():
    spine = pd.DataFrame([{"entity_id": "u1", "label_timestamp": "2024-01-05"}])
    result = point_in_time_join(spine, make_log())
    assert result.iloc[0]["count"] == 2


def test_no_value_yet_returns_none():
    spine = pd.DataFrame([{"entity_id": "u1", "label_timestamp": "2023-12-31"}])
    result = point_in_time_join(spine, make_log())
    assert pd.isna(result.iloc[0]["count"])


def test_unknown_entity_returns_none():
    spine = pd.DataFrame([{"entity_id": "u404", "label_timestamp": "2024-01-06"}])
    result = point_in_time_join(spine, make_log())
    assert pd.isna(result.iloc[0]["count"])


def test_multiple_entities_independent():
    spine = pd.DataFrame(
        [
            {"entity_id": "u1", "label_timestamp": "2024-01-06"},
            {"entity_id": "u2", "label_timestamp": "2024-01-06"},
        ]
    )
    result = point_in_time_join(spine, make_log())
    assert result[result["entity_id"] == "u1"].iloc[0]["count"] == 2
    assert result[result["entity_id"] == "u2"].iloc[0]["count"] == 5

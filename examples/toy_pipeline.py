"""End-to-end example: define two features, materialize them, and build a
point-in-time-correct training set.

Run: python examples/toy_pipeline.py
"""

import os
import sys
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from feature_store_lite import FeatureView, FeatureViewRegistry, FeatureStore, point_in_time_join


# --- Fake raw data: login events per user ---------------------------------
raw_logins = pd.DataFrame(
    [
        {"entity_id": "u1", "event_timestamp": "2024-01-01"},
        {"entity_id": "u1", "event_timestamp": "2024-01-03"},
        {"entity_id": "u1", "event_timestamp": "2024-01-10"},
        {"entity_id": "u2", "event_timestamp": "2024-01-05"},
        {"entity_id": "u2", "event_timestamp": "2024-01-09"},
    ]
)

raw_signup = pd.DataFrame(
    [
        {"entity_id": "u1", "signup_date": "2023-12-01"},
        {"entity_id": "u2", "signup_date": "2024-01-01"},
    ]
)


def compute_rolling_login_count(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Cumulative login count per user, as of each login event."""
    df = raw_df.sort_values(["entity_id", "event_timestamp"]).copy()
    df["value"] = df.groupby("entity_id").cumcount() + 1
    df["feature_name"] = "rolling_login_count"
    return df[["entity_id", "event_timestamp", "feature_name", "value"]]


def compute_days_since_signup(raw_df: pd.DataFrame) -> pd.DataFrame:
    """A single feature value per user, timestamped at signup -- available from
    that point onward."""
    df = raw_df.copy()
    df["event_timestamp"] = df["signup_date"]
    df["feature_name"] = "has_signed_up"
    df["value"] = 1
    return df[["entity_id", "event_timestamp", "feature_name", "value"]]


def main():
    registry = FeatureViewRegistry()
    registry.register(
        FeatureView(
            name="rolling_login_count",
            entity_column="entity_id",
            compute_fn=compute_rolling_login_count,
            description="Cumulative number of logins seen so far for a user.",
        )
    )
    registry.register(
        FeatureView(
            name="has_signed_up",
            entity_column="entity_id",
            compute_fn=compute_days_since_signup,
            description="Flag that becomes 1 once a user has signed up.",
        )
    )

    store = FeatureStore(db_path="feature_store.db")
    store.materialize(registry.get("rolling_login_count"), raw_logins)
    store.materialize(registry.get("has_signed_up"), raw_signup)

    print("=== Online lookup (latest known values) for u1 ===")
    print(store.get_online_features("u1"))

    # Entity spine: one row per training example, with the timestamp we're
    # "as of" -- e.g. when a label was assigned.
    spine = pd.DataFrame(
        [
            {"entity_id": "u1", "label_timestamp": "2024-01-02"},  # before 2nd login
            {"entity_id": "u1", "label_timestamp": "2024-01-05"},  # after 2nd, before 3rd
            {"entity_id": "u2", "label_timestamp": "2024-01-06"},  # after 1st login only
        ]
    )

    log = store.get_offline_log()
    training_set = point_in_time_join(spine, log)

    print("\n=== Point-in-time-correct training set ===")
    print(training_set)
    print(
        "\nNote: u1's rolling_login_count on 2024-01-02 is 1, not 3 -- a naive join"
        " on the full history would have leaked the future logins."
    )


if __name__ == "__main__":
    main()

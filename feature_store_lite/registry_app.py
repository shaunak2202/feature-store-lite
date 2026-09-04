"""Shared feature view registry + raw data sources.

This is the single place feature definitions are wired up to their raw source data,
so both the materialization CLI and the online-serving app import the exact same
definitions -- there's no second copy of feature logic anywhere.

In a real deployment, `RAW_SOURCES` would be database queries / data warehouse
tables instead of in-memory DataFrames; the shape of the rest of the system
(FeatureView, registry, store) doesn't change either way.
"""

import pandas as pd

from .feature_view import FeatureView, FeatureViewRegistry


def compute_rolling_login_count(raw_df: pd.DataFrame) -> pd.DataFrame:
    df = raw_df.sort_values(["entity_id", "event_timestamp"]).copy()
    df["value"] = df.groupby("entity_id").cumcount() + 1
    df["feature_name"] = "rolling_login_count"
    return df[["entity_id", "event_timestamp", "feature_name", "value"]]


def compute_has_signed_up(raw_df: pd.DataFrame) -> pd.DataFrame:
    df = raw_df.copy()
    df["event_timestamp"] = df["signup_date"]
    df["feature_name"] = "has_signed_up"
    df["value"] = 1
    return df[["entity_id", "event_timestamp", "feature_name", "value"]]


RAW_LOGINS = pd.DataFrame(
    [
        {"entity_id": "u1", "event_timestamp": "2024-01-01"},
        {"entity_id": "u1", "event_timestamp": "2024-01-03"},
        {"entity_id": "u1", "event_timestamp": "2024-01-10"},
        {"entity_id": "u2", "event_timestamp": "2024-01-05"},
        {"entity_id": "u2", "event_timestamp": "2024-01-09"},
    ]
)

RAW_SIGNUP = pd.DataFrame(
    [
        {"entity_id": "u1", "signup_date": "2023-12-01"},
        {"entity_id": "u2", "signup_date": "2024-01-01"},
    ]
)

# Maps view name -> raw DataFrame the CLI should pass to `store.materialize`.
RAW_SOURCES = {
    "rolling_login_count": RAW_LOGINS,
    "has_signed_up": RAW_SIGNUP,
}


def build_registry() -> FeatureViewRegistry:
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
            compute_fn=compute_has_signed_up,
            description="Flag that becomes 1 once a user has signed up.",
        )
    )
    return registry

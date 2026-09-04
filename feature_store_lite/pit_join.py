"""Point-in-time-correct join: the core thing a feature store has to get right.

Given an "entity spine" (one row per training example: which entity, and at what
timestamp we're making a prediction / assigning a label), and a long-format feature
log (entity_id, feature_name, event_timestamp, value), produce a wide training table
where each feature column holds the most recent value known STRICTLY AT OR BEFORE
the spine's timestamp for that entity -- never a value that was only known later.

This is deliberately implemented without a `pd.merge_asof` shortcut hidden behind a
one-liner -- it's written so the "as-of" semantics are explicit and testable.
"""

import pandas as pd


def point_in_time_join(
    entity_spine: pd.DataFrame,
    feature_log: pd.DataFrame,
    entity_col: str = "entity_id",
    timestamp_col: str = "label_timestamp",
) -> pd.DataFrame:
    """
    entity_spine: DataFrame with columns [entity_col, timestamp_col, ...other cols]
    feature_log: DataFrame with columns [entity_id, feature_name, event_timestamp, value]

    Returns entity_spine with one extra column per distinct feature_name, filled
    with the point-in-time-correct value (or NaN if no value existed yet).
    """
    if entity_spine.empty:
        return entity_spine.copy()

    spine = entity_spine.copy()
    spine[timestamp_col] = pd.to_datetime(spine[timestamp_col])

    log = feature_log.copy()
    log["event_timestamp"] = pd.to_datetime(log["event_timestamp"])

    result = spine.copy()

    for feature_name in sorted(log["feature_name"].unique()):
        feat = log[log["feature_name"] == feature_name].sort_values("event_timestamp")
        values = []
        # Grouping by entity keeps this O(n log n) per feature rather than a full
        # cross join; still simple/explicit rather than clever.
        feat_by_entity = {
            entity: grp.reset_index(drop=True) for entity, grp in feat.groupby("entity_id")
        }
        for _, row in result.iterrows():
            entity_feat = feat_by_entity.get(row[entity_col])
            if entity_feat is None:
                values.append(None)
                continue
            # Most recent event_timestamp <= label_timestamp (as-of, not future).
            eligible = entity_feat[entity_feat["event_timestamp"] <= row[timestamp_col]]
            if eligible.empty:
                values.append(None)
            else:
                values.append(eligible.iloc[-1]["value"])
        result[feature_name] = values

    return result


def naive_join(
    entity_spine: pd.DataFrame,
    feature_log: pd.DataFrame,
    entity_col: str = "entity_id",
) -> pd.DataFrame:
    """The join a lot of first-draft training pipelines actually write: for each
    entity, grab the LATEST feature value overall, ignoring the spine's timestamp
    entirely. This is exactly what `point_in_time_join` exists to avoid -- it's
    included here (not hidden in a script) so the two can be tested and compared
    side by side on the same inputs.
    """
    spine = entity_spine.copy()
    log = feature_log.copy()
    log["event_timestamp"] = pd.to_datetime(log["event_timestamp"])

    latest = (
        log.sort_values("event_timestamp")
        .groupby([entity_col, "feature_name"], as_index=False)
        .last()
    )

    result = spine.copy()
    for feature_name in sorted(log["feature_name"].unique()):
        feat = latest[latest["feature_name"] == feature_name].set_index(entity_col)["value"]
        result[feature_name] = result[entity_col].map(feat)
    return result

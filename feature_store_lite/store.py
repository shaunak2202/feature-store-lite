"""SQLite-backed offline + online feature storage.

Offline store: append-only log of every computed feature value, keyed by
(entity_id, feature_name, event_timestamp). This is what point-in-time joins read
from to build training sets.

Online store: a latest-value table, one row per (entity_id, feature_name), updated
on every materialization. This is what a serving layer reads from for low-latency
lookups at inference time.

See parquet_store.py for an alternate offline-store backend for larger datasets.
"""

import sqlite3
from typing import Optional
import pandas as pd

from .feature_view import FeatureView


OFFLINE_TABLE = "feature_values"
ONLINE_TABLE = "feature_values_latest"


class FeatureStore:
    def __init__(self, db_path: str = "feature_store.db"):
        self.db_path = db_path
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path)

    def _init_schema(self) -> None:
        with self._connect() as conn:
            conn.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {OFFLINE_TABLE} (
                    entity_id TEXT NOT NULL,
                    feature_name TEXT NOT NULL,
                    event_timestamp TEXT NOT NULL,
                    value REAL,
                    PRIMARY KEY (entity_id, feature_name, event_timestamp)
                )
                """
            )
            conn.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {ONLINE_TABLE} (
                    entity_id TEXT NOT NULL,
                    feature_name TEXT NOT NULL,
                    event_timestamp TEXT NOT NULL,
                    value REAL,
                    PRIMARY KEY (entity_id, feature_name)
                )
                """
            )
            conn.commit()

    def materialize(self, view: FeatureView, raw_df: pd.DataFrame) -> int:
        """Compute a FeatureView's values and write them to both the offline log
        and the online latest-value table. Returns number of rows written offline.
        """
        computed = view.compute(raw_df)
        if computed.empty:
            return 0

        with self._connect() as conn:
            # Offline: append every value (ignore exact duplicates).
            computed.to_sql("_staging", conn, if_exists="replace", index=False)
            conn.execute(
                f"""
                INSERT OR IGNORE INTO {OFFLINE_TABLE}
                    (entity_id, feature_name, event_timestamp, value)
                SELECT entity_id, feature_name, event_timestamp, value FROM _staging
                """
            )

            # Online: keep only the latest value per (entity_id, feature_name).
            latest = (
                computed.sort_values("event_timestamp")
                .groupby(["entity_id", "feature_name"], as_index=False)
                .last()
            )
            for _, row in latest.iterrows():
                conn.execute(
                    f"""
                    INSERT INTO {ONLINE_TABLE} (entity_id, feature_name, event_timestamp, value)
                    VALUES (?, ?, ?, ?)
                    ON CONFLICT(entity_id, feature_name) DO UPDATE SET
                        event_timestamp = excluded.event_timestamp,
                        value = excluded.value
                    WHERE excluded.event_timestamp >= {ONLINE_TABLE}.event_timestamp
                    """,
                    (row.entity_id, row.feature_name, str(row.event_timestamp), row.value),
                )
            conn.execute("DROP TABLE _staging")
            conn.commit()

        return len(computed)

    def get_offline_log(self, feature_name: Optional[str] = None) -> pd.DataFrame:
        query = f"SELECT * FROM {OFFLINE_TABLE}"
        params = ()
        if feature_name:
            query += " WHERE feature_name = ?"
            params = (feature_name,)
        with self._connect() as conn:
            df = pd.read_sql(query, conn, params=params)
        if not df.empty:
            df["event_timestamp"] = pd.to_datetime(df["event_timestamp"])
        return df

    def get_online_features(self, entity_id: str) -> dict:
        """Latest value per feature_name for one entity -- what an online serving
        endpoint would call at inference time."""
        with self._connect() as conn:
            df = pd.read_sql(
                f"SELECT feature_name, value, event_timestamp FROM {ONLINE_TABLE} WHERE entity_id = ?",
                conn,
                params=(entity_id,),
            )
        return {
            row.feature_name: {"value": row.value, "as_of": row.event_timestamp}
            for row in df.itertuples()
        }

    def known_entity_ids(self) -> list:
        """All entity ids that have at least one online feature value. Used by the
        serving layer to give a clean 404 instead of a silent empty response for
        entities that were never materialized."""
        with self._connect() as conn:
            df = pd.read_sql(f"SELECT DISTINCT entity_id FROM {ONLINE_TABLE}", conn)
        return df["entity_id"].tolist() if not df.empty else []

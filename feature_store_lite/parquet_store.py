"""Parquet-backed offline store: an alternate backend for the feature log.

SQLite is fine for a demo-sized offline log, but row-by-row inserts and a single
database file stop being the right tool once the log is large. This module writes
one Parquet file per feature_name under a base directory instead, giving columnar,
append-friendly storage that downstream tools (Spark, DuckDB, pandas) can read
directly without going through a database driver.

It deliberately implements the same shape of interface the rest of the codebase
already expects from an offline log producer (`write` to append values for a
feature, `read` to get back a combined long-format DataFrame), so
`point_in_time_join` doesn't need to know or care which backend produced its input.
"""

import os
from typing import Optional
import pandas as pd


class ParquetOfflineStore:
    def __init__(self, base_path: str = "feature_store_parquet"):
        self.base_path = base_path
        os.makedirs(self.base_path, exist_ok=True)

    def _path_for(self, feature_name: str) -> str:
        return os.path.join(self.base_path, f"{feature_name}.parquet")

    def write(self, feature_name: str, computed_df: pd.DataFrame) -> int:
        """Append rows for one feature_name, de-duplicating on
        (entity_id, feature_name, event_timestamp, value) like the SQLite backend
        does, so repeated materialization runs stay idempotent.
        """
        if computed_df.empty:
            return 0

        df = computed_df.copy()
        df["event_timestamp"] = pd.to_datetime(df["event_timestamp"])

        path = self._path_for(feature_name)
        if os.path.exists(path):
            existing = pd.read_parquet(path)
            existing["event_timestamp"] = pd.to_datetime(existing["event_timestamp"])
            combined = pd.concat([existing, df], ignore_index=True)
        else:
            combined = df

        combined = combined.drop_duplicates(
            subset=["entity_id", "feature_name", "event_timestamp", "value"]
        ).sort_values("event_timestamp")
        combined.to_parquet(path, index=False)
        return len(df)

    def read(self, feature_name: Optional[str] = None) -> pd.DataFrame:
        """Read back the combined feature log, optionally filtered to one feature."""
        if feature_name is not None:
            path = self._path_for(feature_name)
            if not os.path.exists(path):
                return pd.DataFrame(
                    columns=["entity_id", "feature_name", "event_timestamp", "value"]
                )
            return pd.read_parquet(path)

        frames = []
        if os.path.isdir(self.base_path):
            for fname in sorted(os.listdir(self.base_path)):
                if fname.endswith(".parquet"):
                    frames.append(pd.read_parquet(os.path.join(self.base_path, fname)))
        if not frames:
            return pd.DataFrame(
                columns=["entity_id", "feature_name", "event_timestamp", "value"]
            )
        return pd.concat(frames, ignore_index=True)

    def known_feature_names(self) -> list:
        if not os.path.isdir(self.base_path):
            return []
        return sorted(
            fname[: -len(".parquet")]
            for fname in os.listdir(self.base_path)
            if fname.endswith(".parquet")
        )

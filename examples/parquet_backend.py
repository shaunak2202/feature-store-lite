"""Same toy pipeline as toy_pipeline.py, but using the Parquet-backed offline
store instead of SQLite -- shows that `point_in_time_join` works identically
regardless of which backend produced the feature log DataFrame.

Run: python examples/parquet_backend.py
"""

import os
import shutil
import sys
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from feature_store_lite import point_in_time_join
from feature_store_lite.parquet_store import ParquetOfflineStore
from feature_store_lite.registry_app import build_registry, RAW_LOGINS, RAW_SIGNUP

PARQUET_DIR = "feature_store_parquet"


def main():
    # Start clean so repeated runs are easy to reason about.
    if os.path.isdir(PARQUET_DIR):
        shutil.rmtree(PARQUET_DIR)

    registry = build_registry()
    store = ParquetOfflineStore(base_path=PARQUET_DIR)

    login_view = registry.get("rolling_login_count")
    signup_view = registry.get("has_signed_up")

    store.write(login_view.name, login_view.compute(RAW_LOGINS))
    store.write(signup_view.name, signup_view.compute(RAW_SIGNUP))

    print(f"=== Wrote Parquet files under ./{PARQUET_DIR}/ ===")
    for fname in sorted(os.listdir(PARQUET_DIR)):
        print(f"  {fname}")

    log = store.read()
    print(f"\n=== Combined feature log read back from Parquet ({len(log)} rows) ===")
    print(log)

    spine = pd.DataFrame(
        [
            {"entity_id": "u1", "label_timestamp": "2024-01-02"},
            {"entity_id": "u1", "label_timestamp": "2024-01-05"},
            {"entity_id": "u2", "label_timestamp": "2024-01-06"},
        ]
    )
    training_set = point_in_time_join(spine, log)

    print("\n=== Point-in-time-correct training set (same join logic, Parquet input) ===")
    print(training_set)


if __name__ == "__main__":
    main()

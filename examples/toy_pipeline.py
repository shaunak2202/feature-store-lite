"""End-to-end example: define two features, materialize them, and build a
point-in-time-correct training set.

Run: python examples/toy_pipeline.py
"""

import os
import sys
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from feature_store_lite import FeatureStore, point_in_time_join
from feature_store_lite.registry_app import build_registry, RAW_LOGINS, RAW_SIGNUP


def main():
    registry = build_registry()

    store = FeatureStore(db_path="feature_store.db")
    store.materialize(registry.get("rolling_login_count"), RAW_LOGINS)
    store.materialize(registry.get("has_signed_up"), RAW_SIGNUP)

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

"""Side-by-side comparison: a naive join vs. the point-in-time-correct join, on
the exact same entity spine and feature log.

This is meant to make the training/serving skew problem visible in real numbers,
rather than something only asserted in a test. It reuses the same login-count
feature from the toy pipeline so the two examples tell a consistent story.

Run: python examples/naive_vs_pit.py
"""

import os
import sys
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from feature_store_lite import FeatureStore, point_in_time_join
from feature_store_lite.pit_join import naive_join
from feature_store_lite.registry_app import build_registry, RAW_LOGINS


def main():
    registry = build_registry()
    store = FeatureStore(db_path="feature_store.db")
    store.materialize(registry.get("rolling_login_count"), RAW_LOGINS)

    log = store.get_offline_log(feature_name="rolling_login_count")

    # A training spine where the label was assigned well BEFORE a user's later
    # logins happened -- exactly the situation where naive joins leak the future.
    spine = pd.DataFrame(
        [
            {"entity_id": "u1", "label_timestamp": "2024-01-02", "label": 0},
            {"entity_id": "u1", "label_timestamp": "2024-01-05", "label": 1},
            {"entity_id": "u2", "label_timestamp": "2024-01-06", "label": 0},
        ]
    )

    naive_result = naive_join(spine, log)
    pit_result = point_in_time_join(spine, log)

    print("=== Naive join (ignores label_timestamp, grabs latest value overall) ===")
    print(naive_result[["entity_id", "label_timestamp", "label", "rolling_login_count"]])

    print("\n=== Point-in-time join (only uses values known as of label_timestamp) ===")
    print(pit_result[["entity_id", "label_timestamp", "label", "rolling_login_count"]])

    diff = naive_result["rolling_login_count"] != pit_result["rolling_login_count"]
    n_leaked = int(diff.sum())

    print(f"\n{n_leaked} of {len(spine)} training rows would have leaked future information")
    print("under the naive join -- the model would have trained on a feature value that")
    print("didn't exist yet at prediction time, inflating validation metrics in a way that")
    print("won't hold up once the model actually serves live traffic.")


if __name__ == "__main__":
    main()

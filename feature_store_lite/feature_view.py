"""Feature definition API.

A FeatureView bundles:
  - a name
  - the entity column it's keyed on (e.g. "user_id")
  - a compute function: (raw_df) -> DataFrame with columns
      [entity_id, event_timestamp, feature_name, value]

This is the single place feature logic lives, so the same definition can be used
both to materialize historical values for training and to compute a fresh value for
online serving.
"""

from dataclasses import dataclass, field
from typing import Callable, Dict
import pandas as pd


REQUIRED_COLUMNS = {"entity_id", "event_timestamp", "feature_name", "value"}


@dataclass
class FeatureView:
    name: str
    entity_column: str
    compute_fn: Callable[[pd.DataFrame], pd.DataFrame]
    description: str = ""

    def compute(self, raw_df: pd.DataFrame) -> pd.DataFrame:
        """Run the compute function and validate its output shape."""
        out = self.compute_fn(raw_df)
        missing = REQUIRED_COLUMNS - set(out.columns)
        if missing:
            raise ValueError(
                f"FeatureView '{self.name}' compute_fn is missing columns: {missing}. "
                f"Expected at least {REQUIRED_COLUMNS}."
            )
        out = out.copy()
        out["event_timestamp"] = pd.to_datetime(out["event_timestamp"])
        return out[["entity_id", "event_timestamp", "feature_name", "value"]]


class FeatureViewRegistry:
    """Simple in-memory registry so views can be looked up by name.

    A real feature store would persist this (e.g. as YAML/DB rows); here it's kept
    in-process for simplicity, mirroring how a small project would actually use it
    (import definitions once, register at startup).
    """

    def __init__(self) -> None:
        self._views: Dict[str, FeatureView] = {}

    def register(self, view: FeatureView) -> None:
        if view.name in self._views:
            raise ValueError(f"FeatureView '{view.name}' already registered")
        self._views[view.name] = view

    def get(self, name: str) -> FeatureView:
        return self._views[name]

    def all(self):
        return list(self._views.values())

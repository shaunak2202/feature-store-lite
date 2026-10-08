"""Materialization CLI.

Wraps the shared registry + FeatureStore into commands a scheduler (cron, Airflow,
a CI job, etc.) could call directly -- the same operation that, in a real feature
store, runs on a schedule to keep the offline log and online cache up to date.

Usage:
    python -m feature_store_lite.cli list-views
    python -m feature_store_lite.cli materialize --view rolling_login_count
    python -m feature_store_lite.cli materialize --all
    python -m feature_store_lite.cli materialize --view rolling_login_count --db-path my.db
    python -m feature_store_lite.cli materialize --view rolling_login_count --backend parquet
"""

import argparse
import sys

from .registry_app import build_registry, RAW_SOURCES
from .store import FeatureStore
from .parquet_store import ParquetOfflineStore


def cmd_list_views(args) -> int:
    registry = build_registry()
    views = registry.all()
    if not views:
        print("No feature views are registered.")
        return 0
    for view in views:
        print(f"{view.name}: {view.description}")
    return 0


def cmd_materialize(args) -> int:
    registry = build_registry()

    if args.all and args.view:
        print("error: pass either --view <name> or --all, not both", file=sys.stderr)
        return 2
    if args.all:
        view_names = registry.names()
    elif args.view:
        view_names = [args.view]
    else:
        print("error: materialize requires either --view <name> or --all", file=sys.stderr)
        return 2

    if any(not name.strip() for name in view_names):
        print("error: --view requires a non-empty view name", file=sys.stderr)
        return 2

    if args.backend not in ("sqlite", "parquet"):
        print(f"error: unknown backend '{args.backend}'. Use 'sqlite' or 'parquet'.", file=sys.stderr)
        return 2

    if args.backend == "sqlite":
        store = FeatureStore(db_path=args.db_path)
    else:
        store = ParquetOfflineStore(base_path=args.parquet_path)

    total_written = 0
    for name in view_names:
        try:
            view = registry.get(name)
        except KeyError:
            print(f"error: unknown view '{name}'. Run 'list-views' to see options.", file=sys.stderr)
            return 2
        raw_df = RAW_SOURCES.get(name)
        if raw_df is None:
            print(f"error: no raw data source registered for view '{name}'", file=sys.stderr)
            return 2

        try:
            if args.backend == "sqlite":
                written = store.materialize(view, raw_df)
            else:
                computed = view.compute(raw_df)
                store.write(view.name, computed)
                written = len(computed)
        except ValueError as exc:
            print(f"error: materializing '{name}' failed: {exc}", file=sys.stderr)
            return 1

        total_written += written
        print(f"materialized '{name}': {written} rows written to offline log ({args.backend})")

    print(f"done. {total_written} total rows written across {len(view_names)} view(s).")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="featurestore", description="FeatureStoreLite CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    list_parser = subparsers.add_parser("list-views", help="List registered feature views")
    list_parser.set_defaults(func=cmd_list_views)

    mat_parser = subparsers.add_parser("materialize", help="Materialize one or all feature views")
    mat_parser.add_argument("--view", help="Name of a single registered view to materialize")
    mat_parser.add_argument("--all", action="store_true", help="Materialize every registered view")
    mat_parser.add_argument(
        "--db-path", default="feature_store.db", help="Path to the SQLite store (default: feature_store.db)"
    )
    mat_parser.add_argument(
        "--backend",
        default="sqlite",
        choices=["sqlite", "parquet"],
        help="Offline storage backend to materialize into (default: sqlite)",
    )
    mat_parser.add_argument(
        "--parquet-path",
        default="feature_store_parquet",
        help="Directory for the Parquet backend (default: feature_store_parquet)",
    )
    mat_parser.set_defaults(func=cmd_materialize)

    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except Exception as exc:  # defensive: CLI should report, never traceback, on bad input
        print(f"error: unexpected failure: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())

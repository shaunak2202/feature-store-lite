"""Materialization CLI.

Wraps the shared registry + FeatureStore into commands a scheduler (cron, Airflow,
a CI job, etc.) could call directly -- the same operation that, in a real feature
store, runs on a schedule to keep the offline log and online cache up to date.

Usage:
    python -m feature_store_lite.cli list-views
    python -m feature_store_lite.cli materialize --view rolling_login_count
    python -m feature_store_lite.cli materialize --all
    python -m feature_store_lite.cli materialize --view rolling_login_count --db-path my.db
"""

import argparse
import sys

from .registry_app import build_registry, RAW_SOURCES
from .store import FeatureStore


def cmd_list_views(args) -> int:
    registry = build_registry()
    for view in registry.all():
        print(f"{view.name}: {view.description}")
    return 0


def cmd_materialize(args) -> int:
    registry = build_registry()
    store = FeatureStore(db_path=args.db_path)

    if args.all:
        view_names = registry.names()
    elif args.view:
        view_names = [args.view]
    else:
        print("error: materialize requires either --view <name> or --all", file=sys.stderr)
        return 2

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
        written = store.materialize(view, raw_df)
        total_written += written
        print(f"materialized '{name}': {written} rows written to offline log")

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
    mat_parser.set_defaults(func=cmd_materialize)

    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())

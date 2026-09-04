import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from feature_store_lite.cli import main
from feature_store_lite.store import FeatureStore


def test_list_views_runs_without_error(capsys):
    exit_code = main(["list-views"])
    assert exit_code == 0
    out = capsys.readouterr().out
    assert "rolling_login_count" in out
    assert "has_signed_up" in out


def test_materialize_single_view_writes_to_db(tmp_path, capsys):
    db_path = str(tmp_path / "cli_test.db")
    exit_code = main(["materialize", "--view", "rolling_login_count", "--db-path", db_path])
    assert exit_code == 0

    store = FeatureStore(db_path=db_path)
    log = store.get_offline_log(feature_name="rolling_login_count")
    assert len(log) == 5  # matches RAW_LOGINS row count


def test_materialize_all_writes_every_view(tmp_path):
    db_path = str(tmp_path / "cli_test_all.db")
    exit_code = main(["materialize", "--all", "--db-path", db_path])
    assert exit_code == 0

    store = FeatureStore(db_path=db_path)
    log = store.get_offline_log()
    feature_names = set(log["feature_name"].unique())
    assert feature_names == {"rolling_login_count", "has_signed_up"}


def test_materialize_unknown_view_errors_cleanly(tmp_path, capsys):
    db_path = str(tmp_path / "cli_test_bad.db")
    exit_code = main(["materialize", "--view", "does_not_exist", "--db-path", db_path])
    assert exit_code == 2
    err = capsys.readouterr().err
    assert "unknown view" in err


def test_materialize_without_view_or_all_errors(tmp_path, capsys):
    db_path = str(tmp_path / "cli_test_noargs.db")
    exit_code = main(["materialize", "--db-path", db_path])
    assert exit_code == 2

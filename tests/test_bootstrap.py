"""Tests for deployment bootstrap script (scripts/bootstrap.py).

Verifies safe, idempotent initialization of schema, demo data, weather cache,
and model training on fresh startup or redeployments.
"""

import os
import sqlite3
import pytest
from app import create_app
from scripts.bootstrap import run_bootstrap


@pytest.fixture
def fresh_bootstrap_env(tmp_path, monkeypatch):
    """Set up an isolated temporary directory with no database and clean models."""
    test_db = str(tmp_path / "fresh_bakery.db")
    test_models = str(tmp_path / "models")
    os.makedirs(test_models, exist_ok=True)

    monkeypatch.setenv("DATABASE_PATH", test_db)

    # Point MODEL_DIR to temporary path
    import ml.train
    import ml.train_model
    import ml.registry

    monkeypatch.setattr(ml.train_model, "MODEL_DIR", test_models)
    monkeypatch.setattr(ml.registry, "MODELS_DIR", test_models)

    app = create_app({
        "TESTING": True,
        "DATABASE": test_db,
        "SECRET_KEY": "bootstrap-test-secret",
        "WTF_CSRF_ENABLED": False,
        "RL_ENABLED": False,
    })

    yield app, test_db, test_models


def test_bootstrap_fresh_rebuild_and_idempotence(fresh_bootstrap_env):
    """Deleting database and models and running bootstrap must rebuild a working demo, and repeated runs must be idempotent."""
    app, test_db, test_models = fresh_bootstrap_env

    # Ensure database does not exist before bootstrap
    assert not os.path.exists(test_db)

    # 1. First run: Fresh rebuild from scratch
    success = run_bootstrap(app=app)
    assert success is True

    # 1a. Database file was created
    assert os.path.exists(test_db)

    conn = sqlite3.connect(test_db)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    # 1b. All 9 tables exist
    tables = [r["name"] for r in cur.execute("SELECT name FROM sqlite_master WHERE type='table';").fetchall()]
    expected_tables = {
        "users", "shops", "items", "daily_log", "weather_daily",
        "forecasts", "model_registry", "policies", "evaluation_results"
    }
    assert expected_tables.issubset(set(tables))

    # 1c. Demo shop exists
    shop = cur.execute("SELECT id, name FROM shops WHERE name = 'Demo Bakery';").fetchone()
    assert shop is not None
    shop_id = shop["id"]

    # 1d. Five default items exist
    items = cur.execute("SELECT id, name FROM items WHERE shop_id = ?;", (shop_id,)).fetchall()
    assert len(items) == 5

    # 1e. Daily log has sales history
    log_count = cur.execute("SELECT COUNT(*) AS c FROM daily_log WHERE shop_id = ?;", (shop_id,)).fetchone()["c"]
    assert log_count > 100

    # 1f. Weather cache has rows
    weather_count = cur.execute("SELECT COUNT(*) AS c FROM weather_daily WHERE shop_id = ?;", (shop_id,)).fetchone()["c"]
    assert weather_count > 0

    # 1g. Model registry contains active promoted models
    active_models = cur.execute(
        "SELECT COUNT(*) AS c FROM model_registry WHERE shop_id = ? AND active = 1;",
        (shop_id,),
    ).fetchone()["c"]
    assert active_models > 0

    initial_shop_count = cur.execute("SELECT COUNT(*) AS c FROM shops;").fetchone()["c"]
    initial_item_count = cur.execute("SELECT COUNT(*) AS c FROM items;").fetchone()["c"]
    initial_model_count = cur.execute("SELECT COUNT(*) AS c FROM model_registry;").fetchone()["c"]
    conn.close()

    # 2. Second run: Idempotency verification (must skip all operations cleanly)
    success_second = run_bootstrap(app=app)
    assert success_second is True

    conn2 = sqlite3.connect(test_db)
    conn2.row_factory = sqlite3.Row
    cur2 = conn2.cursor()

    assert cur2.execute("SELECT COUNT(*) AS c FROM shops;").fetchone()["c"] == initial_shop_count == 1
    assert cur2.execute("SELECT COUNT(*) AS c FROM items;").fetchone()["c"] == initial_item_count == 5
    assert cur2.execute("SELECT COUNT(*) AS c FROM model_registry;").fetchone()["c"] == initial_model_count

    conn2.close()

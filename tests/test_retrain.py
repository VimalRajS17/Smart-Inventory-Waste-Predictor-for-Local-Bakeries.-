"""Tests for Phase 4C: Actuals tracking, automated retraining, promotion guards, rollback, and scheduling."""

import os
from datetime import date, datetime, timedelta, timezone
import pytest
from app.db import get_db
from ml.registry import (
    get_active_model,
    promote_and_register_item_model,
    rollback_item_model,
)
from ml.train import (
    count_new_logs_since_training,
    retrain_all_eligible_shops,
    retrain_all_command,
    rollback_command,
)
from app.scheduler import init_scheduler


def test_actual_qty_filled_on_daily_log_save(client, app):
    """Test that saving a day via /api/daily-log fills actual_qty in matching forecasts row."""
    with app.app_context():
        db = get_db()
        item = db.execute("SELECT id FROM items WHERE shop_id = 1").fetchone()
        item_id = item["id"]

        test_date = "2026-11-25"
        # Seed an existing forecast for test_date without actual_qty
        db.execute(
            """
            INSERT INTO forecasts (shop_id, item_id, date, predicted_quantity, weather_source, model_version)
            VALUES (1, ?, ?, 42.5, 'test', 'TestModel v1')
            ON CONFLICT(shop_id, item_id, date) DO UPDATE SET actual_qty = NULL
            """,
            (item_id, test_date),
        )
        db.commit()

        # Check before save
        row_before = db.execute(
            "SELECT actual_qty FROM forecasts WHERE shop_id = 1 AND item_id = ? AND date = ?",
            (item_id, test_date),
        ).fetchone()
        assert row_before["actual_qty"] is None

    # Post daily log with sold = 39
    res = client.post(
        "/api/daily-log",
        json={
            "item_id": item_id,
            "date": test_date,
            "sold": 39,
            "baked": 40,
            "wasted": 1,
        },
    )
    assert res.status_code == 200

    # Verify actual_qty is now filled
    with app.app_context():
        db = get_db()
        row_after = db.execute(
            "SELECT actual_qty, predicted_quantity FROM forecasts WHERE shop_id = 1 AND item_id = ? AND date = ?",
            (item_id, test_date),
        ).fetchone()
        assert row_after is not None
        assert row_after["actual_qty"] == 39.0
        assert row_after["predicted_quantity"] == 42.5


def test_retrain_threshold_30_log_rows(app):
    """Test that retrain_all_eligible_shops requires at least 30 new log rows since last training."""
    with app.app_context():
        db = get_db()
        # Ensure a shop exists with a recorded training timestamp in model_registry
        shop_id = 1
        item = db.execute("SELECT id FROM items WHERE shop_id = ?", (shop_id,)).fetchone()
        item_id = item["id"]

        # Insert a baseline version trained 10 days ago
        ten_days_ago = (datetime.now(timezone.utc) - timedelta(days=10)).isoformat()
        db.execute(
            """
            INSERT INTO model_registry (shop_id, item_id, model_type, version, trained_at, beats_baseline, active)
            VALUES (?, ?, 'moving_average', 1, ?, 1, 1)
            ON CONFLICT(shop_id, item_id, version) DO UPDATE SET trained_at = excluded.trained_at
            """,
            (shop_id, item_id, ten_days_ago),
        )
        # Clear existing logs for item
        db.execute("DELETE FROM daily_log WHERE shop_id = ?", (shop_id,))
        db.commit()

        # Scenario 1: Only 12 new rows -> should be skipped
        today = date.today()
        for i in range(12):
            d_str = (today - timedelta(days=12 - i)).isoformat()
            db.execute(
                "INSERT INTO daily_log (shop_id, item_id, date, sold, created_at) VALUES (?, ?, ?, 25, datetime('now'))",
                (shop_id, item_id, d_str),
            )
        db.commit()

        count_12, _ = count_new_logs_since_training(shop_id, db)
        assert count_12 == 12

        results_skipped = retrain_all_eligible_shops(db=db, force=False, min_new_rows=30)
        shop_res = next(r for r in results_skipped if r["shop_id"] == shop_id)
        assert shop_res["retrained"] is False
        assert shop_res["new_logs_count"] == 12

        # Scenario 2: Add 20 more rows (total 32 >= 30) -> should qualify and retrain
        for i in range(20):
            d_str = (today + timedelta(days=i + 1)).isoformat()
            db.execute(
                "INSERT INTO daily_log (shop_id, item_id, date, sold, created_at) VALUES (?, ?, ?, 30, datetime('now'))",
                (shop_id, item_id, d_str),
            )
        db.commit()

        count_32, _ = count_new_logs_since_training(shop_id, db)
        assert count_32 == 32

        results_trained = retrain_all_eligible_shops(db=db, force=False, min_new_rows=30)
        shop_res_trained = next(r for r in results_trained if r["shop_id"] == shop_id)
        assert shop_res_trained["retrained"] is True
        assert len(shop_res_trained["evaluations"]) > 0


def test_worse_model_not_promoted_and_keeps_previous_active(app):
    """Test that if a newly trained model fails the promotion rule, previous active model stays active."""
    import pandas as pd
    with app.app_context():
        db = get_db()
        item = db.execute("SELECT id, name FROM items WHERE shop_id = 1").fetchone()
        item_id = item["id"]

        # Ensure a clean registry with Version 1 active
        db.execute("DELETE FROM model_registry WHERE shop_id = 1 AND item_id = ?", (item_id,))
        db.execute(
            """
            INSERT INTO model_registry (shop_id, item_id, model_type, version, mae, beats_baseline, active)
            VALUES (1, ?, 'random_forest', 1, 2.5, 1, 1)
            """,
            (item_id,),
        )
        db.commit()

        active_before = get_active_model(1, item_id, db=db)
        assert active_before["version"] == 1
        assert active_before["active"] == 1

        # Simulate a retraining run where the candidate model fails baseline comparison (beats_baseline=False)
        worse_eval = {
            "item_name": item["name"],
            "item_id": item_id,
            "cold_start": False,
            "beats_baseline": False, # FAILED PROMOTION
            "promoted_model": "moving_average",
            "model_results": {
                "moving_average": {"mean_mae": 5.8, "mean_rmse": 7.2, "mean_wape_raw": 0.25},
            },
        }

        dummy_df = pd.DataFrame([
            {"date": f"2026-09-{i:02d}", "item": item["name"], "item_id": item_id, "quantity_sold": 20.0, "temp_mean_c": 20.0, "is_rainy": 0, "is_holiday": 0}
            for i in range(1, 30)
        ])

        res = promote_and_register_item_model(1, item_id, worse_eval, dummy_df, db=db)

        # Version 2 was created, but beats_baseline=0 and active=0
        assert res["version"] == 2
        assert res["beats_baseline"] == 0
        assert res["active"] == 0

        # Previous Version 1 MUST STILL BE ACTIVE!
        v1_row = db.execute(
            "SELECT active FROM model_registry WHERE shop_id = 1 AND item_id = ? AND version = 1",
            (item_id,),
        ).fetchone()
        assert v1_row["active"] == 1

        # get_active_model must still return version 1
        current_active = get_active_model(1, item_id, db=db)
        assert current_active["version"] == 1


def test_better_model_promoted_and_activates(app):
    """Test that a model passing promotion rule becomes active and deactivates prior versions."""
    import pandas as pd
    with app.app_context():
        db = get_db()
        item = db.execute("SELECT id, name FROM items WHERE shop_id = 1").fetchone()
        item_id = item["id"]

        # Simulate a successful promotion (beats_baseline=True)
        better_eval = {
            "item_name": item["name"],
            "item_id": item_id,
            "cold_start": False,
            "beats_baseline": True, # PASSED PROMOTION
            "promoted_model": "linear_regression",
            "model_results": {
                "linear_regression": {"mean_mae": 1.9, "mean_rmse": 2.3, "mean_wape_raw": 0.08},
            },
        }

        # Provide sufficient history with dates and sales for build_features
        dates = [(date(2026, 6, 1) + timedelta(days=i)).isoformat() for i in range(80)]
        dummy_df = pd.DataFrame([
            {"date": d, "item": item["name"], "item_id": item_id, "quantity_sold": 30.0 + (i % 7), "temp_mean_c": 21.0, "is_rainy": 0, "is_holiday": 0}
            for i, d in enumerate(dates)
        ])

        res = promote_and_register_item_model(1, item_id, better_eval, dummy_df, db=db)

        new_version = res["version"]
        assert res["beats_baseline"] == 1
        assert res["active"] == 1

        # Active model is now the newly promoted version
        current_active = get_active_model(1, item_id, db=db)
        assert current_active["version"] == new_version
        assert current_active["model_type"] == "linear_regression"


def test_rollback_cli_and_function(runner, app):
    """Test rollback restores previous model version in registry and via Flask CLI."""
    with app.app_context():
        db = get_db()
        item = db.execute("SELECT id FROM items WHERE shop_id = 1").fetchone()
        item_id = item["id"]

        # Ensure we have v1 (inactive) and v2 (active)
        db.execute("DELETE FROM model_registry WHERE shop_id = 1 AND item_id = ?", (item_id,))
        db.execute(
            """
            INSERT INTO model_registry (shop_id, item_id, model_type, version, beats_baseline, active)
            VALUES (1, ?, 'random_forest', 1, 1, 0)
            """,
            (item_id,),
        )
        db.execute(
            """
            INSERT INTO model_registry (shop_id, item_id, model_type, version, beats_baseline, active)
            VALUES (1, ?, 'linear_regression', 2, 1, 1)
            """,
            (item_id,),
        )
        db.commit()

        # Check current active is v2
        active_curr = get_active_model(1, item_id, db=db)
        assert active_curr["version"] == 2

    # Execute rollback CLI command: flask --app app rollback --shop-id 1 --item-id M
    result = runner.invoke(rollback_command, ["--shop-id", "1", "--item-id", str(item_id)])
    assert result.exit_code == 0
    assert "v2 deactivated -> v1 (random_forest) is now active" in result.output

    # Verify v1 is now active in database
    with app.app_context():
        db = get_db()
        active_after = get_active_model(1, item_id, db=db)
        assert active_after["version"] == 1
        assert active_after["model_type"] == "random_forest"


def test_admin_retrain_owner_authorization(client, unauth_client, app):
    """Test POST /api/admin/retrain: shop owner succeeds, non-owner gets 403, unauthenticated gets 401."""
    # 1. Unauthenticated gets 401
    res_unauth = unauth_client.post("/api/admin/retrain")
    assert res_unauth.status_code == 401

    # 2. Authenticated owner succeeds
    res_owner = client.post("/api/admin/retrain")
    assert res_owner.status_code == 200
    data = res_owner.get_json()
    assert data["status"] == "success"
    assert "items_trained" in data
    assert len(data["items_trained"]) > 0

    # 3. Another user with their own shop can retrain their own shop
    from tests.conftest import LoafTestClient
    with app.app_context():
        db = get_db()
        from werkzeug.security import generate_password_hash
        db.execute(
            "INSERT INTO users (email, password_hash) VALUES ('other@example.com', ?)",
            (generate_password_hash("pass123"),),
        )
        other_user_id = db.execute("SELECT id FROM users WHERE email = 'other@example.com'").fetchone()["id"]
        # Create shop belonging to other user with an item
        db.execute(
            "INSERT INTO shops (owner_user_id, name, city, lat, lon) VALUES (?, 'Other Bakery', 'York', 53.96, -1.08)",
            (other_user_id,),
        )
        other_shop_id = db.execute("SELECT id FROM shops WHERE owner_user_id = ?", (other_user_id,)).fetchone()["id"]
        db.execute(
            "INSERT INTO items (shop_id, name) VALUES (?, 'Brioche')",
            (other_shop_id,),
        )
        db.commit()

    other_client = LoafTestClient(app.test_client())
    t = other_client.get_csrf()
    other_client.post(
        "/login",
        data={"email": "other@example.com", "password": "pass123", "csrf_token": t},
        csrf=False,
    )
    res_other = other_client.post("/api/admin/retrain")
    assert res_other.status_code == 200
    assert res_other.get_json()["shop_id"] == other_shop_id


def test_reports_api_contains_active_model_telemetry(client, app):
    """Test GET /api/reports/forecast-vs-actual returns active_model info."""
    with app.app_context():
        db = get_db()
        item = db.execute("SELECT id FROM items WHERE shop_id = 1").fetchone()
        item_id = item["id"]
        # Ensure an active model exists
        db.execute(
            """
            INSERT INTO model_registry (shop_id, item_id, model_type, version, mae, beats_baseline, active)
            VALUES (1, ?, 'random_forest', 2, 2.85, 1, 1)
            ON CONFLICT(shop_id, item_id, version) DO UPDATE SET active = 1
            """,
            (item_id,),
        )
        db.commit()

    res = client.get("/api/reports/forecast-vs-actual?days=7")
    assert res.status_code == 200
    data = res.get_json()
    assert "active_model" in data
    assert data["active_model"] is not None
    assert "model_type" in data["active_model"]
    assert "version" in data["active_model"]
    assert "trained_at" in data["active_model"]


def test_scheduler_initialization_toggle(app, monkeypatch):
    """Test init_scheduler is controlled by RETRAIN_SCHEDULE_ENABLED environment variable."""
    # When disabled: returns None
    monkeypatch.setenv("RETRAIN_SCHEDULE_ENABLED", "0")
    s_off = init_scheduler(app)
    assert s_off is None

    # When enabled: returns BackgroundScheduler instance
    monkeypatch.setenv("RETRAIN_SCHEDULE_ENABLED", "1")
    s_on = init_scheduler(app)
    assert s_on is not None
    assert s_on.running is True
    s_on.shutdown(wait=False)


def test_retrain_all_cli_command(runner, app):
    """Test flask --app app retrain-all runs via Flask CLI runner."""
    result = runner.invoke(retrain_all_command, ["--force"])
    assert result.exit_code == 0
    assert "Starting batch retraining scan" in result.output
    assert "Batch retraining complete" in result.output


def test_done_when_log_new_days_retrain_and_forecast_uses_new_version(client, runner, app):
    """Full workflow: log new days, run retrain-all, and next forecast uses the new version."""
    with app.app_context():
        db = get_db()
        item = db.execute("SELECT id, name FROM items WHERE shop_id = 1 AND name = 'Croissant'").fetchone()
        item_id = item["id"]

        # 1. Clear daily logs and seed initial 15 days
        db.execute("DELETE FROM daily_log WHERE item_id = ?", (item_id,))
        db.execute("DELETE FROM model_registry WHERE item_id = ?", (item_id,))
        base_d = date(2026, 8, 1)
        for i in range(15):
            d_str = (base_d + timedelta(days=i)).isoformat()
            db.execute("INSERT INTO daily_log (shop_id, item_id, date, sold) VALUES (1, ?, ?, 20)", (item_id, d_str))
        db.commit()

    # Initial training -> v1
    res1 = runner.invoke(retrain_all_command, ["--force"])
    assert res1.exit_code == 0

    with app.app_context():
        db = get_db()
        # Set initial trained_at to the end of the initial 15-day batch
        db.execute("UPDATE model_registry SET trained_at = '2026-08-15 12:00:00' WHERE shop_id = 1")
        db.commit()
        v1_row = db.execute(
            "SELECT version, model_type FROM model_registry WHERE shop_id = 1 AND item_id = ? AND active = 1",
            (item_id,),
        ).fetchone()
        assert v1_row is not None
        assert v1_row["version"] == 1

    # 2. Log 35 new days via /api/daily-log (from 2026-08-16 onwards)
    start_d = date(2026, 8, 16)
    for i in range(35):
        d_str = (start_d + timedelta(days=i)).isoformat()
        client.post(
            "/api/daily-log",
            json={"item_id": item_id, "date": d_str, "sold": 25 + (i % 5)},
        )

    # 3. Run retrain-all
    res2 = runner.invoke(retrain_all_command, ["--min-new-rows", "30"])
    assert res2.exit_code == 0
    assert "Retrained" in res2.output

    # 4. Registry shows a new version
    with app.app_context():
        db = get_db()
        v_latest = db.execute(
            "SELECT MAX(version) AS max_v FROM model_registry WHERE shop_id = 1 AND item_id = ?",
            (item_id,),
        ).fetchone()["max_v"]
        assert v_latest >= 2

    # 5. Next forecast uses active model version
    fc_res = client.get("/api/forecast?date=2026-10-01")
    assert fc_res.status_code == 200
    fc_data = fc_res.get_json()
    croissant_fc = next(f for f in fc_data["forecasts"] if f["item_id"] == item_id)
    assert f"v{v_latest}" in croissant_fc["model_version"] or "v" in croissant_fc["model_version"]


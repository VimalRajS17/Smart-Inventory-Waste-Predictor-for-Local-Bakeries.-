"""Tests for demand forecasting, cold start logic, weather fallback, and forecasts persistence."""

from datetime import date, timedelta
import sqlite3
import pytest

from app.db import get_db
from ml.predictor import predict_item_demand, predict_all_items


def test_predictor_weather_fallback_when_unavailable(app):
    """Test that when weather data is unavailable, predictor falls back to seasonal average with weather_source='fallback'."""
    with app.app_context():
        db = get_db()
        item = db.execute("SELECT id FROM items WHERE name = 'Croissant'").fetchone()
        item_id = item["id"]

        # Insert some sales history into daily_log so the item has data
        base_date = date(2026, 8, 1)
        for i in range(70):
            d_str = (base_date + timedelta(days=i)).isoformat()
            db.execute(
                "INSERT INTO daily_log (shop_id, item_id, date, sold) VALUES (1, ?, ?, ?)",
                (item_id, d_str, 30 + (i % 5)),
            )
        # Ensure no weather exists for target date
        target_date = "2026-12-15"
        db.execute("DELETE FROM weather_daily WHERE date = ?", (target_date,))
        db.commit()

        # Predict for target_date without network (conftest no_real_network ensures no HTTP call)
        res = predict_item_demand(item_id, target_date, shop_id=1)

        assert res["weather_source"] == "fallback"
        assert res["weather"]["is_rainy"] == 0
        assert res["weather"]["precip_mm"] == 0.0
        assert isinstance(res["weather"]["temp_mean_c"], float)
        assert res["predicted_qty"] >= 0


def test_predictor_cold_start_flag_under_60_days(app):
    """Test that an item with fewer than 60 days history triggers cold_start=True and uses 7-day moving average."""
    with app.app_context():
        db = get_db()
        item = db.execute("SELECT id FROM items WHERE name = 'Sourdough Loaf'").fetchone()
        item_id = item["id"]

        # Clear existing logs for this item
        db.execute("DELETE FROM daily_log WHERE item_id = ?", (item_id,))

        # Insert only 10 days of sales with known values (e.g. 20 sold each day)
        base_date = date(2026, 10, 1)
        for i in range(10):
            d_str = (base_date + timedelta(days=i)).isoformat()
            db.execute(
                "INSERT INTO daily_log (shop_id, item_id, date, sold) VALUES (1, ?, ?, ?)",
                (item_id, d_str, 20),
            )
        db.commit()

        target_date = "2026-10-12"
        res = predict_item_demand(item_id, target_date, shop_id=1)

        assert res["cold_start"] is True
        assert res["history_days"] == 10
        # 7-day average of 20 is 20.0
        assert res["predicted_qty"] == 20.0


def test_predictor_lag_features_stale_flag(app):
    """Test that lag_features_stale is True when target date is > 1 day after the last logged sales date."""
    with app.app_context():
        db = get_db()
        item = db.execute("SELECT id FROM items WHERE name = 'Cupcake'").fetchone()
        item_id = item["id"]

        db.execute("DELETE FROM daily_log WHERE item_id = ?", (item_id,))

        # Logged date is 2026-10-01
        db.execute(
            "INSERT INTO daily_log (shop_id, item_id, date, sold) VALUES (1, ?, '2026-10-01', 15)",
            (item_id,),
        )
        db.commit()

        # Target date is 2026-10-05 (>1 day later)
        res_stale = predict_item_demand(item_id, "2026-10-05", shop_id=1)
        assert res_stale["lag_features_stale"] is True

        # Target date is 2026-10-02 (exactly 1 day later)
        res_fresh = predict_item_demand(item_id, "2026-10-02", shop_id=1)
        assert res_fresh["lag_features_stale"] is False


def test_api_forecast_upserts_and_returns_fallback(client, app):
    """Test GET /api/forecast returns weather_source 'fallback' when weather call fails and persists into forecasts table."""
    with app.app_context():
        db = get_db()
        # Seed sufficient history for all items
        items = db.execute("SELECT id FROM items").fetchall()
        for it in items:
            for i in range(10):
                d_str = (date(2026, 9, 1) + timedelta(days=i)).isoformat()
                db.execute(
                    "INSERT OR IGNORE INTO daily_log (shop_id, item_id, date, sold) VALUES (1, ?, ?, ?)",
                    (it["id"], d_str, 20),
                )
        db.commit()

    target_date = "2026-12-20"
    res = client.get(f"/api/forecast?date={target_date}")
    assert res.status_code == 200
    data = res.get_json()
    assert data["date"] == target_date
    assert len(data["forecasts"]) >= 5

    for fc in data["forecasts"]:
        assert fc["weather_source"] == "fallback"
        assert "predicted_qty" in fc
        assert "cold_start" in fc
        assert "lag_features_stale" in fc
        assert "model_version" in fc

    # Check persistence in forecasts table
    with app.app_context():
        db = get_db()
        persisted = db.execute("SELECT * FROM forecasts WHERE date = ?", (target_date,)).fetchall()
        assert len(persisted) == len(data["forecasts"])
        for p in persisted:
            assert p["weather_source"] == "fallback"


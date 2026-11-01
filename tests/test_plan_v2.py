"""Tests for Phase 4B: Safety buffer, carry-over, bake quantity, what-if reasons, overstock flags, and 7-day outlook."""

import math
from datetime import date, timedelta
import io
import pytest
from app.db import get_db
from ml.predictor import (
    calculate_safety_buffer,
    get_carry_over,
    check_overstock,
    predict_item_demand,
    generate_what_if_reason,
    generate_seven_day_outlook,
)


def test_safety_buffer_80th_percentile_calculation(app):
    """Test safety buffer calculates the 80th percentile of absolute backtest errors over 28 days."""
    with app.app_context():
        db = get_db()
        item = db.execute("SELECT id FROM items WHERE name = 'Croissant'").fetchone()
        item_id = item["id"]

        # Clear existing logs and forecasts for clean test
        db.execute("DELETE FROM daily_log WHERE item_id = ?", (item_id,))
        db.execute("DELETE FROM forecasts WHERE item_id = ?", (item_id,))

        target_date = date(2026, 10, 31)
        # Create 28 days of daily_log and forecasts with known errors: 1, 2, 3, ... 28
        # The 80th percentile of [1..28] in numpy is ~ 22.6
        for i in range(28):
            d = target_date - timedelta(days=28 - i)
            d_str = d.isoformat()
            actual_sold = 30
            predicted = 30 + (i + 1) # absolute error is (i + 1)

            db.execute(
                "INSERT INTO daily_log (shop_id, item_id, date, sold) VALUES (1, ?, ?, ?)",
                (item_id, d_str, actual_sold),
            )
            db.execute(
                """
                INSERT INTO forecasts (shop_id, item_id, date, predicted_quantity, weather_source, model_version)
                VALUES (1, ?, ?, ?, 'test', 'TestModel')
                """,
                (item_id, d_str, predicted),
            )
        db.commit()

        # Compute buffer at 80th percentile (default)
        buffer_80 = calculate_safety_buffer(1, item_id, target_date, predicted_qty=50.0, percentile=80.0, db=db)
        # 80th percentile of range 1..28 is ~22.6
        assert 22.0 <= buffer_80 <= 23.5

        # Test configurability: 50th percentile (median of 1..28 is 14.5)
        buffer_50 = calculate_safety_buffer(1, item_id, target_date, predicted_qty=50.0, percentile=50.0, db=db)
        assert 14.0 <= buffer_50 <= 15.0


def test_safety_buffer_fallback_on_insufficient_history(app):
    """Test safety buffer falls back to 15% of forecast when history is short."""
    with app.app_context():
        db = get_db()
        item = db.execute("SELECT id FROM items WHERE name = 'Chocolate Muffin'").fetchone()
        item_id = item["id"]

        db.execute("DELETE FROM daily_log WHERE item_id = ?", (item_id,))
        db.execute("DELETE FROM forecasts WHERE item_id = ?", (item_id,))
        db.commit()

        target_date = date(2026, 10, 31)
        buf = calculate_safety_buffer(1, item_id, target_date, predicted_qty=40.0, db=db)
        # 15% of 40 is 6.0
        assert buf == 6.0


def test_carry_over_rule_by_shelf_life(app):
    """Test carry-over applies only when shelf_life_days > 1; otherwise 0."""
    with app.app_context():
        db = get_db()
        # Item 1: shelf_life_days = 1 (e.g. Baguette or Croissant)
        db.execute("UPDATE items SET shelf_life_days = 1 WHERE name = 'Croissant'")
        # Item 2: shelf_life_days = 3 (e.g. Sourdough Loaf)
        db.execute("UPDATE items SET shelf_life_days = 3 WHERE name = 'Sourdough Loaf'")

        croissant = db.execute("SELECT id FROM items WHERE name = 'Croissant'").fetchone()
        sourdough = db.execute("SELECT id FROM items WHERE name = 'Sourdough Loaf'").fetchone()

        target_date = date(2026, 11, 1)
        yesterday_str = (target_date - timedelta(days=1)).isoformat()

        # Log 12 leftovers for Croissant and 8 leftovers for Sourdough
        db.execute(
            """
            INSERT INTO daily_log (shop_id, item_id, date, sold, leftover)
            VALUES (1, ?, ?, 20, 12)
            ON CONFLICT(shop_id, item_id, date) DO UPDATE SET leftover = 12
            """,
            (croissant["id"], yesterday_str),
        )
        db.execute(
            """
            INSERT INTO daily_log (shop_id, item_id, date, sold, leftover)
            VALUES (1, ?, ?, 15, 8)
            ON CONFLICT(shop_id, item_id, date) DO UPDATE SET leftover = 8
            """,
            (sourdough["id"], yesterday_str),
        )
        db.commit()

        # Croissant shelf_life = 1 -> carry_over must be strictly 0
        carry_croissant, last_leftover_croissant = get_carry_over(1, croissant["id"], 1, target_date, db=db)
        assert carry_croissant == 0
        assert last_leftover_croissant == 12

        # Sourdough shelf_life = 3 -> carry_over must be 8
        carry_sourdough, last_leftover_sourdough = get_carry_over(1, sourdough["id"], 3, target_date, db=db)
        assert carry_sourdough == 8
        assert last_leftover_sourdough == 8


def test_bake_quantity_formula_and_bounds(app):
    """Test bake_qty = max(0, ceil(forecast + buffer - carry_over))."""
    with app.app_context():
        db = get_db()
        item = db.execute("SELECT id FROM items WHERE name = 'Sourdough Loaf'").fetchone()
        item_id = item["id"]

        # Ensure shelf life > 1
        db.execute("UPDATE items SET shelf_life_days = 3 WHERE id = ?", (item_id,))

        target_date = "2026-11-05"
        prev_date = "2026-11-04"

        # Log leftover = 5
        db.execute(
            "INSERT INTO daily_log (shop_id, item_id, date, sold, leftover) VALUES (1, ?, ?, 20, 5)"
            " ON CONFLICT(shop_id, item_id, date) DO UPDATE SET leftover = 5",
            (item_id, prev_date),
        )
        db.commit()

        res = predict_item_demand(item_id, target_date, shop_id=1)

        forecast = res["forecast"]
        buffer = res["buffer"]
        carry_over = res["carry_over"]
        bake_qty = res["bake_qty"]

        assert carry_over == 5
        expected_bake = max(0, math.ceil(forecast + buffer - carry_over))
        assert bake_qty == expected_bake

        # Also verify never below 0
        assert bake_qty >= 0

        # Verify lower and upper bounds
        assert res["lower"] == max(0.0, round(forecast - buffer, 1))
        assert res["upper"] == round(forecast + buffer, 1)


def test_overstock_flag_and_units():
    """Test overstock flag and excess units when leftover + carry_over > upper."""
    # Case 1: Overstocked
    # leftover = 20, carry_over = 20 -> total 40. upper = 30.0 -> overstock by 10
    overstock, units, warning = check_overstock(last_leftover=20, carry_over=20, upper=30.0)
    assert overstock is True
    assert units == 10
    assert "exceeds upper forecast" in warning

    # Case 2: Not overstocked
    # total 15 <= upper 25.0
    overstock_no, units_no, warning_no = check_overstock(last_leftover=5, carry_over=10, upper=25.0)
    assert overstock_no is False
    assert units_no == 0
    assert warning_no is None


def test_cold_start_reason_explanation(app):
    """Test cold-start item (< 60 days history) explains short history."""
    with app.app_context():
        db = get_db()
        item = db.execute("SELECT id FROM items WHERE name = 'Cinnamon Roll'").fetchone()
        item_id = item["id"]

        db.execute("DELETE FROM daily_log WHERE item_id = ?", (item_id,))
        for i in range(15):
            d_str = (date(2026, 10, 1) + timedelta(days=i)).isoformat()
            db.execute("INSERT INTO daily_log (shop_id, item_id, date, sold) VALUES (1, ?, ?, 10)", (item_id, d_str))
        db.commit()

        res = predict_item_demand(item_id, "2026-10-25", shop_id=1)
        assert res["cold_start"] is True
        assert "history is short" in res["reason"].lower()


def test_what_if_reason_generation():
    """Test counterfactual what-if reason produces expected plain language format."""
    class DummyModel:
        def predict(self, X):
            # Model heavily increases prediction on weekend and rain
            pred = 30.0
            if "is_weekend" in X and X["is_weekend"].iloc[0] == 1:
                pred += 10.0
            if "same_weekday_mean_4w" in X and X["same_weekday_mean_4w"].iloc[0] > 30:
                pred += 8.0
            return [pred]

    feature_cols = ["day_of_week", "is_weekend", "same_weekday_mean_4w", "temperature_c", "is_rainy"]
    feat_dict = {
        "day_of_week": 5, # Saturday
        "is_weekend": 1,
        "same_weekday_mean_4w": 38.0,
        "temperature_c": 18.0,
        "is_rainy": 0,
    }

    target_date = date(2026, 10, 31) # Saturday
    reason = generate_what_if_reason(
        active_model=DummyModel(),
        feature_cols=feature_cols,
        feat_dict=feat_dict,
        history_rows=[{"sold": 30.0} for _ in range(28)],
        target_date=target_date,
        predicted_qty=48.0,
        cold_start=False,
        shop_id=1,
    )

    assert "Higher than usual" in reason
    assert "weekend" in reason.lower()
    assert "strong last four saturdays" in reason.lower()


def test_seven_day_recursive_outlook(app):
    """Test 7-day outlook produces 7 days with widening buffer and lower confidence label."""
    with app.app_context():
        db = get_db()
        item = db.execute("SELECT id FROM items WHERE name = 'Croissant'").fetchone()
        item_id = item["id"]

        res = predict_item_demand(item_id, "2026-11-01", shop_id=1)
        outlook = res["seven_day_outlook"]

        assert len(outlook) == 7
        # Day 1 is Normal confidence
        assert outlook[0]["confidence"] == "Normal"
        assert outlook[0]["date"] == "2026-11-01"

        # Days 2 to 7 are Lower confidence
        for day in outlook[1:]:
            assert day["confidence"] == "Lower confidence"

        # Buffer widens across steps: last day buffer >= first day buffer
        assert outlook[-1]["buffer"] >= outlook[0]["buffer"]


def test_api_daily_log_saves_and_retrieves_leftover(client, app):
    """Test /api/daily-log accepts and persists leftover quantity."""
    with app.app_context():
        db = get_db()
        item = db.execute("SELECT id FROM items WHERE shop_id = 1").fetchone()
        item_id = item["id"]

    log_date = "2026-11-10"
    res = client.post(
        "/api/daily-log",
        json={
            "item_id": item_id,
            "date": log_date,
            "sold": 25,
            "baked": 30,
            "wasted": 2,
            "leftover": 3,
            "sold_out": False,
        },
    )
    assert res.status_code == 200
    data = res.get_json()
    assert data["leftover"] == 3

    # Check GET /api/daily-log
    get_res = client.get(f"/api/daily-log?item_id={item_id}&from={log_date}&to={log_date}")
    assert get_res.status_code == 200
    rows = get_res.get_json()
    assert len(rows) >= 1
    assert rows[0]["leftover"] == 3


def test_csv_import_with_leftover_column(client, app):
    """Test POST /api/import/csv imports leftover column into daily_log."""
    with app.app_context():
        db = get_db()
        item = db.execute("SELECT id, name FROM items WHERE shop_id = 1").fetchone()
        item_name = item["name"]

    csv_data = f"""date,item,sold,baked,wasted,leftover
2026-11-15,{item_name},40,45,1,4
"""
    data = {
        "file": (io.BytesIO(csv_data.encode("utf-8")), "import_leftovers.csv"),
        "col_date": "date",
        "col_item": "item",
        "col_sold": "sold",
        "col_baked": "baked",
        "col_wasted": "wasted",
        "col_leftover": "leftover",
    }
    res = client.post(
        "/api/import/csv",
        data=data,
        content_type="multipart/form-data",
    )
    assert res.status_code == 200
    res_data = res.get_json()
    assert res_data["rows_imported"] == 1

    with app.app_context():
        db = get_db()
        row = db.execute(
            "SELECT sold, baked, wasted, leftover FROM daily_log WHERE date = '2026-11-15' AND shop_id = 1",
        ).fetchone()
        assert row is not None
        assert row["sold"] == 40
        assert row["leftover"] == 4


def test_api_forecast_contains_phase_4b_fields(client):
    """Test GET /api/forecast returns all required Phase 4B fields."""
    res = client.get("/api/forecast?date=2026-11-01")
    assert res.status_code == 200
    data = res.get_json()
    forecasts = data["forecasts"]
    assert len(forecasts) > 0

    for f in forecasts:
        assert "forecast" in f
        assert "lower" in f
        assert "upper" in f
        assert "bake_qty" in f
        assert "buffer" in f
        assert "carry_over" in f
        assert "leftover" in f
        assert "reason" in f
        assert "overstock" in f
        assert "overstock_units" in f
        assert "seven_day_outlook" in f


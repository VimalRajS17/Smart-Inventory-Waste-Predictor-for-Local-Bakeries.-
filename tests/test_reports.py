"""Tests for the reports endpoints: forecast vs actual comparison, aggregations, and validations."""

from datetime import date, timedelta
from app.db import get_db


def test_reports_validation(client):
    """GET /api/reports/forecast-vs-actual validates 'days' and 'to' parameters."""
    # Bad days
    res = client.get("/api/reports/forecast-vs-actual?days=-5")
    assert res.status_code == 400
    assert res.get_json()["error"]["code"] == "invalid_input"

    # Zero days
    res_zero = client.get("/api/reports/forecast-vs-actual?days=0")
    assert res_zero.status_code == 400

    # Bad date
    res_date = client.get("/api/reports/forecast-vs-actual?to=bad-date")
    assert res_date.status_code == 400


def test_reports_empty_period(client, app):
    """Reports endpoint returns consistent 7-day structure when no sales/forecasts exist in range."""
    res = client.get("/api/reports/forecast-vs-actual?days=7&to=2030-01-07")
    assert res.status_code == 200
    data = res.get_json()

    assert data["period"]["days"] == 7
    assert data["period"]["end_date"] == "2030-01-07"
    assert data["summary"]["total_forecast"] == 0.0
    assert data["summary"]["total_sold"] == 0
    assert data["summary"]["mae"] is None
    assert len(data["daily"]) == 7
    assert len(data["records"]) == 0


def test_reports_forecast_vs_actual_joining(client, app):
    """Reports endpoint joins forecasts and daily_log accurately and computes MAE."""
    with app.app_context():
        db = get_db()
        item = db.execute("SELECT id FROM items WHERE name = 'Croissant'").fetchone()
        item_id = item["id"]

        test_date = "2026-10-25"

        # Clear existing records for that day and item
        db.execute("DELETE FROM forecasts WHERE date = ? AND item_id = ?", (test_date, item_id))
        db.execute("DELETE FROM daily_log WHERE date = ? AND item_id = ?", (test_date, item_id))

        # Insert forecast of 25.0
        db.execute(
            """
            INSERT INTO forecasts (shop_id, item_id, date, predicted_quantity, weather_source, model_version)
            VALUES (1, ?, ?, 25.0, 'forecast', 'TestModel')
            """,
            (item_id, test_date),
        )

        # Insert actual sold of 20, baked 28, wasted 3
        db.execute(
            """
            INSERT INTO daily_log (shop_id, item_id, date, sold, baked, wasted, sold_out)
            VALUES (1, ?, ?, 20, 28, 3, 0)
            """,
            (item_id, test_date),
        )
        db.commit()

    # Query reports covering test_date
    res = client.get(f"/api/reports/forecast-vs-actual?days=7&to={test_date}")
    assert res.status_code == 200
    data = res.get_json()

    assert data["summary"]["total_forecast"] >= 25.0
    assert data["summary"]["total_sold"] >= 20
    assert data["summary"]["total_baked"] >= 28
    assert data["summary"]["total_wasted"] >= 3

    # Find the record for Croissant on test_date
    rec = next(r for r in data["records"] if r["date"] == test_date and r["item_id"] == item_id)
    assert rec["forecast_qty"] == 25.0
    assert rec["actual_sold"] == 20
    assert rec["baked_qty"] == 28
    assert rec["wasted_qty"] == 3
    # difference = actual_sold - forecast_qty = 20 - 25.0 = -5.0
    assert rec["difference"] == -5.0


def test_reports_item_id_filtering(client, app):
    """GET /api/reports/forecast-vs-actual supports per-item filtering via item_id."""
    with app.app_context():
        db = get_db()
        croissant = db.execute("SELECT id, name FROM items WHERE name = 'Croissant'").fetchone()
        croissant_id = croissant["id"]

    # 1. Valid item_id
    res = client.get(f"/api/reports/forecast-vs-actual?item_id={croissant_id}&days=7")
    assert res.status_code == 200
    data = res.get_json()
    assert data["item"] is not None
    assert data["item"]["id"] == croissant_id
    assert data["item"]["name"] == "Croissant"
    for r in data["records"]:
        assert r["item_id"] == croissant_id

    # 2. Empty item_id query param: ?item_id=&days=7
    res_empty = client.get("/api/reports/forecast-vs-actual?item_id=&days=7")
    assert res_empty.status_code == 200
    data_empty = res_empty.get_json()
    assert data_empty["item"] is None

    # 3. Non-existent item_id returns 404
    res_404 = client.get("/api/reports/forecast-vs-actual?item_id=99999&days=7")
    assert res_404.status_code == 404
    assert res_404.get_json()["error"]["code"] == "not_found"

    # 4. Invalid item_id formats return 400
    res_invalid_str = client.get("/api/reports/forecast-vs-actual?item_id=invalid&days=7")
    assert res_invalid_str.status_code == 400
    assert res_invalid_str.get_json()["error"]["code"] == "invalid_input"

    res_zero = client.get("/api/reports/forecast-vs-actual?item_id=0&days=7")
    assert res_zero.status_code == 400
    assert res_zero.get_json()["error"]["code"] == "invalid_input"

    res_neg = client.get("/api/reports/forecast-vs-actual?item_id=-4&days=7")
    assert res_neg.status_code == 400
    assert res_neg.get_json()["error"]["code"] == "invalid_input"


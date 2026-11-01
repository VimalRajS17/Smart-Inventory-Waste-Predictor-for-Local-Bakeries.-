"""Tests for Phase 5: Forecast accuracy, 60-day evaluation simulation, waste reporting, and CSV exports."""

import csv
import io
import math
from datetime import date, timedelta
import pytest
from app.db import get_db
from ml.evaluate import compute_metrics
from ml.simulation import run_historical_simulation


# ---------------------------------------------------------------------------
# 1. Hand-Calculated Metric Formula Tests
# ---------------------------------------------------------------------------

def test_hand_calculated_mae_and_wape():
    """Verify MAE and WAPE match hand-calculated numbers on a small known example."""
    # Example 1:
    # Actuals: [20, 30, 50] -> sum = 100
    # Predictions: [24, 25, 45]
    # Absolute Errors: |20-24|=4, |30-25|=5, |50-45|=5 -> sum = 14
    # MAE = 14 / 3 = 4.67
    # WAPE = 14 / 100 = 0.14 = 14.0%
    y_true = [20.0, 30.0, 50.0]
    y_pred = [24.0, 25.0, 45.0]

    metrics = compute_metrics(y_true, y_pred)
    assert metrics["mae"] == 4.67
    assert metrics["wape"] == 0.14
    assert metrics["wape_pct"] == 14.0


def test_hand_calculated_baseline_wape():
    """Verify Model WAPE vs Baseline WAPE comparison on a small hand-calculated example."""
    # Actuals: [20, 30, 50] (sum = 100)
    # Model predictions: [24, 25, 45] (errors: 4, 5, 5 -> sum = 14 -> WAPE = 14.0%)
    # Baseline (Lag-7) actuals: [18, 35, 40]
    # Baseline errors: |20-18|=2, |30-35|=5, |50-40|=10 -> sum = 17
    # Baseline WAPE = 17 / 100 = 0.17 = 17.0%
    # Result: Model WAPE (14.0%) < Baseline WAPE (17.0%) -> beats_baseline = True
    actuals = [20.0, 30.0, 50.0]
    preds = [24.0, 25.0, 45.0]
    lag7 = [18.0, 35.0, 40.0]

    model_errs = [abs(a - p) for a, p in zip(actuals, preds)]
    base_errs = [abs(a - b) for a, b in zip(actuals, lag7)]
    sum_actuals = sum(actuals)

    model_wape = sum(model_errs) / sum_actuals
    base_wape = sum(base_errs) / sum_actuals

    assert model_wape == 0.14
    assert base_wape == 0.17
    assert model_wape < base_wape  # Model beats baseline


def test_hand_calculated_waste_and_financial_conversion():
    """Verify physical waste, unmet demand, and monetary conversions match exact hand calculations."""
    # Price = $3.00, Cost = $1.00
    price = 3.0
    cost = 1.0

    # Day 1: Baked = 25, Demand = 20
    # Sold = min(25, 20) = 20
    # Unsold Waste = max(0, 25 - 20) = 5 units
    # Unmet Demand = max(0, 20 - 25) = 0 units
    # Revenue = 20 * 3.00 = 60.00
    # Production Cost = 25 * 1.00 = 25.00
    # Waste Cost = 5 * 1.00 = 5.00
    # Lost Sales Cost = 0 * 3.00 = 0.00
    # Net Profit = 60.00 - 25.00 = 35.00
    b1, d1 = 25, 20
    s1 = min(b1, d1)
    w1 = max(0, b1 - d1)
    l1 = max(0, d1 - b1)
    p1 = s1 * price - b1 * cost
    assert s1 == 20
    assert w1 == 5
    assert l1 == 0
    assert w1 * cost == 5.0
    assert p1 == 35.0

    # Day 2: Baked = 15, Demand = 22
    # Sold = min(15, 22) = 15
    # Unsold Waste = 0 units
    # Unmet Demand = 22 - 15 = 7 units
    # Revenue = 15 * 3.00 = 45.00
    # Production Cost = 15 * 1.00 = 15.00
    # Lost Sales Value = 7 * 3.00 = 21.00
    # Net Profit = 45.00 - 15.00 = 30.00
    b2, d2 = 15, 22
    s2 = min(b2, d2)
    w2 = max(0, b2 - d2)
    l2 = max(0, d2 - b2)
    p2 = s2 * price - b2 * cost
    assert s2 == 15
    assert w2 == 0
    assert l2 == 7
    assert l2 * price == 21.0
    assert p2 == 30.0

    # Total 2-day net profit = 35.00 + 30.00 = 65.00
    assert p1 + p2 == 65.0


# ---------------------------------------------------------------------------
# 2. Accuracy Report Endpoint Tests
# ---------------------------------------------------------------------------

def test_api_reports_accuracy(client, app):
    """GET /api/reports/accuracy returns per-item MAE, WAPE, and baseline WAPE."""
    with app.app_context():
        db = get_db()
        item = db.execute("SELECT id FROM items WHERE name = 'Croissant'").fetchone()
        item_id = item["id"]

        # Insert 3 days of forecasts and actuals
        base_d = date(2026, 10, 20)
        # Dates: 2026-10-20, 2026-10-21, 2026-10-22
        # And their lag-7 dates: 2026-10-13, 2026-10-14, 2026-10-15
        for i, (act, pred, lag) in enumerate([(20, 24, 18), (30, 25, 35), (50, 45, 40)]):
            d_curr = (base_d + timedelta(days=i)).isoformat()
            d_lag7 = (base_d + timedelta(days=i) - timedelta(days=7)).isoformat()

            # Lag-7 sales in daily_log
            db.execute(
                "INSERT INTO daily_log (shop_id, item_id, date, sold) VALUES (1, ?, ?, ?) "
                "ON CONFLICT(shop_id, item_id, date) DO UPDATE SET sold = excluded.sold",
                (item_id, d_lag7, lag),
            )
            # Current day sales in daily_log
            db.execute(
                "INSERT INTO daily_log (shop_id, item_id, date, sold) VALUES (1, ?, ?, ?) "
                "ON CONFLICT(shop_id, item_id, date) DO UPDATE SET sold = excluded.sold",
                (item_id, d_curr, act),
            )
            # Forecast with actual_qty in forecasts
            db.execute(
                "INSERT INTO forecasts (shop_id, item_id, date, predicted_quantity, actual_qty) "
                "VALUES (1, ?, ?, ?, ?) "
                "ON CONFLICT(shop_id, item_id, date) DO UPDATE SET predicted_quantity = excluded.predicted_quantity, actual_qty = excluded.actual_qty",
                (item_id, d_curr, pred, act),
            )
        db.commit()

    res = client.get(f"/api/reports/accuracy?days=28&to=2026-10-25")
    assert res.status_code == 200
    data = res.get_json()

    assert data["status"] == "success"
    assert data["period"]["days"] == 28
    croissant = next((it for it in data["items"] if it["item_id"] == item_id), None)
    assert croissant is not None
    assert croissant["records_count"] >= 3
    # Check that MAE, WAPE, and baseline_wape are calculated numbers
    assert croissant["mae"] > 0
    assert croissant["wape"] > 0
    assert croissant["baseline_wape"] > 0


# ---------------------------------------------------------------------------
# 3. Waste Report Endpoint Tests
# ---------------------------------------------------------------------------

def test_api_reports_waste(client, app):
    """GET /api/reports/waste returns 4-week logged wasted and leftover totals and costs."""
    with app.app_context():
        db = get_db()
        item = db.execute("SELECT id FROM items WHERE name = 'Croissant'").fetchone()
        item_id = item["id"]

        test_date = "2026-10-24"
        db.execute(
            """
            INSERT INTO daily_log (shop_id, item_id, date, baked, sold, wasted, leftover)
            VALUES (1, ?, ?, 50, 40, 6, 4)
            ON CONFLICT(shop_id, item_id, date) DO UPDATE SET
                baked = excluded.baked, sold = excluded.sold,
                wasted = excluded.wasted, leftover = excluded.leftover
            """,
            (item_id, test_date),
        )
        db.commit()

    res = client.get(f"/api/reports/waste?days=28&to={test_date}")
    assert res.status_code == 200
    data = res.get_json()

    assert data["status"] == "success"
    assert data["summary"]["total_baked"] >= 50
    assert data["summary"]["total_wasted"] >= 6
    assert data["summary"]["total_leftover"] >= 4
    assert data["summary"]["total_loss_units"] >= 10
    assert data["summary"]["total_waste_cost"] > 0.0


# ---------------------------------------------------------------------------
# 4. Simulation Endpoint & Past Data Label Tests
# ---------------------------------------------------------------------------

def test_api_reports_simulation_label(client, app):
    """GET /api/reports/simulation returns 60-day replay results labeled 'Simulation on past data'."""
    res = client.get("/api/reports/simulation?days=60")
    assert res.status_code == 200
    data = res.get_json()

    # The simulation label MUST be present in the response
    assert data["simulation_label"] == "Simulation on past data"
    assert "policies" in data
    assert len(data["policies"]) >= 3

    policy_names = [p["policy_name"] for p in data["policies"]]
    assert "same_as_last_week" in policy_names
    assert "forecast_only" in policy_names
    assert "forecast_buffer" in policy_names

    # Check baseline savings vs baseline is 0.0
    base_pol = next(p for p in data["policies"] if p["policy_name"] == "same_as_last_week")
    assert base_pol["savings_vs_baseline"] == 0.0


# ---------------------------------------------------------------------------
# 5. CSV Export Endpoints Tests
# ---------------------------------------------------------------------------

def test_export_daily_log_csv(client, app):
    """GET /api/export/daily-log returns RFC 4180 CSV with daily_log columns."""
    with app.app_context():
        db = get_db()
        item = db.execute("SELECT id FROM items LIMIT 1").fetchone()
        db.execute(
            """
            INSERT INTO daily_log (shop_id, item_id, date, sold, baked, wasted, leftover, sold_out, reason)
            VALUES (1, ?, '2026-10-25', 20, 25, 2, 3, 0, 'Sunny day')
            ON CONFLICT(shop_id, item_id, date) DO UPDATE SET sold = excluded.sold
            """,
            (item["id"],),
        )
        db.commit()

    res = client.get("/api/export/daily-log")
    assert res.status_code == 200
    assert "text/csv" in res.content_type
    assert "attachment; filename=" in res.headers.get("Content-Disposition", "")

    csv_text = res.get_data(as_text=True)
    reader = csv.reader(io.StringIO(csv_text))
    header = next(reader)
    expected_cols = ["date", "item_name", "sold", "baked", "wasted", "leftover", "sold_out", "reason"]
    assert header == expected_cols

    # Verify rows exist
    rows = list(reader)
    assert len(rows) > 0


def test_export_forecasts_csv(client, app):
    """GET /api/export/forecasts returns RFC 4180 CSV with forecasts columns."""
    # Ensure at least one forecast exists
    with app.app_context():
        db = get_db()
        item = db.execute("SELECT id FROM items LIMIT 1").fetchone()
        db.execute(
            """
            INSERT INTO forecasts (shop_id, item_id, date, predicted_quantity, actual_qty, model_version)
            VALUES (1, ?, '2026-10-25', 18.0, 16.0, 'HistGradientBoosting')
            ON CONFLICT(shop_id, item_id, date) DO UPDATE SET actual_qty = excluded.actual_qty
            """,
            (item["id"],),
        )
        db.commit()

    res = client.get("/api/export/forecasts")
    assert res.status_code == 200
    assert "text/csv" in res.content_type
    assert "attachment; filename=" in res.headers.get("Content-Disposition", "")

    csv_text = res.get_data(as_text=True)
    reader = csv.reader(io.StringIO(csv_text))
    header = next(reader)
    expected_cols = [
        "date", "item_name", "predicted_quantity", "actual_qty",
        "weather_source", "cold_start", "model_version", "created_at"
    ]
    assert header == expected_cols

    rows = list(reader)
    assert len(rows) > 0


# ---------------------------------------------------------------------------
# 6. CLI Command & Persistence Tests
# ---------------------------------------------------------------------------

def test_evaluate_cli_persists_in_evaluation_results(runner, app):
    """CLI 'flask evaluate --shop-id 1' executes 60-day replay and stores results in evaluation_results."""
    result = runner.invoke(args=["evaluate", "--shop-id", "1"])
    assert result.exit_code == 0
    assert "PHASE 5: 60-DAY HISTORICAL REPLAY SIMULATION" in result.output
    assert "SIMULATION LABEL: Simulation on past data" in result.output
    assert "1. Same as Last Week" in result.output
    assert "2. Forecast Only" in result.output
    assert "3. Forecast + Safety Buffer" in result.output

    with app.app_context():
        db = get_db()
        rows = db.execute("SELECT * FROM evaluation_results WHERE shop_id = 1").fetchall()
        assert len(rows) > 0
        # Check shop aggregate records (item_id IS NULL)
        agg_rows = [r for r in rows if r["item_id"] is None]
        assert len(agg_rows) >= 3


# ---------------------------------------------------------------------------
# 7. Reports Screen UI Verification
# ---------------------------------------------------------------------------

def test_reports_screen_renders_simulation_label_and_elements(client):
    """GET /reports renders the page with export buttons and 'Simulation on past data' label."""
    res = client.get("/reports")
    assert res.status_code == 200
    html = res.get_data(as_text=True)

    # Verify simulation label is present in HTML
    assert "Simulation on past data" in html

    # Verify export buttons exist
    assert "/api/export/daily-log" in html
    assert "/api/export/forecasts" in html

    # Verify accuracy, simulation, and waste container divs exist
    assert 'id="reports-accuracy-container"' in html
    assert 'id="reports-simulation-container"' in html
    assert 'id="reports-waste-container"' in html

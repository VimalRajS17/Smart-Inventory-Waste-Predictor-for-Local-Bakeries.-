"""Tests for Phase 1.5: 730-day dynamic dataset, seed-demo clearing, weather chunking, and reset-demo."""

from datetime import date, timedelta
import os
import pandas as pd
import pytest

from app.db import get_db, reset_demo, seed_demo
from data.generate_sample_data import generate_sample_data
from ml.predictor import predict_item_demand
from ml.train_model import load_from_db


def test_generate_sample_data_ends_yesterday_and_spans_730_days(tmp_path):
    """generate_sample_data generates exactly 730 days ending yesterday with 5 items."""
    csv_file = str(tmp_path / "sales_history.csv")
    df = generate_sample_data(csv_path=csv_file, num_days=730)

    assert os.path.exists(csv_file)
    assert len(df) == 730 * 5

    today = date.today()
    yesterday = today - timedelta(days=1)
    start_expected = today - timedelta(days=730)

    min_date = date.fromisoformat(df["date"].min())
    max_date = date.fromisoformat(df["date"].max())

    assert max_date == yesterday
    assert min_date == start_expected
    assert (max_date - min_date).days + 1 == 730

    expected_cols = {"date", "item", "quantity_sold", "day_of_week", "is_weekend", "temperature_c", "is_rainy", "is_holiday"}
    assert expected_cols.issubset(set(df.columns))

    # Verify per item count
    counts = df["item"].value_counts()
    assert len(counts) == 5
    for item_name, count in counts.items():
        assert count == 730


def test_seed_demo_clears_old_demo_rows(app):
    """seed_demo clears old demo rows before inserting new records."""
    with app.app_context():
        db = get_db()
        shop = db.execute("SELECT id FROM shops WHERE name = 'Demo Bakery'").fetchone()
        shop_id = shop["id"]

        # Insert a dummy old record with a legacy date
        legacy_date = "2020-01-01"
        item = db.execute("SELECT id FROM items WHERE shop_id = ? LIMIT 1", (shop_id,)).fetchone()
        db.execute(
            "INSERT INTO daily_log (shop_id, item_id, date, sold) VALUES (?, ?, ?, 99)",
            (shop_id, item["id"], legacy_date),
        )
        db.commit()

        # Check it exists
        old_row = db.execute(
            "SELECT id FROM daily_log WHERE shop_id = ? AND date = ?",
            (shop_id, legacy_date),
        ).fetchone()
        assert old_row is not None

        # Run seed_demo
        seed_demo()

        # Check old record was purged
        purged = db.execute(
            "SELECT id FROM daily_log WHERE shop_id = ? AND date = ?",
            (shop_id, legacy_date),
        ).fetchone()
        assert purged is None


def test_fetch_weather_chunking_and_skipping(app, monkeypatch):
    """fetch-weather command chunks archive requests into <=90 days and skips cached dates."""
    from tests.conftest import MockResponse

    with app.app_context():
        db = get_db()
        shop = db.execute("SELECT id, lat, lon FROM shops WHERE name = 'Demo Bakery'").fetchone()
        shop_id = shop["id"]

        today = date.today()
        # Seed 90 days of weather
        c_rows = []
        for d in range(1, 91):
            d_str = (today - timedelta(days=d)).isoformat()
            c_rows.append((shop_id, d_str, 20.0, 20.0, 0.0, 0, "archive"))

        db.executemany(
            """
            INSERT INTO weather_daily (shop_id, date, temperature_c, temp_mean_c, precip_mm, is_rainy, source)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(shop_id, date) DO UPDATE SET temp_mean_c = excluded.temp_mean_c
            """,
            c_rows,
        )
        db.commit()

        api_calls = []

        def mock_get(url, params=None, timeout=None):
            api_calls.append({"url": url, "params": params})
            time_series = ["2026-01-01"]
            return MockResponse({
                "daily": {
                    "time": time_series,
                    "temperature_2m_max": [18.0],
                    "temperature_2m_min": [10.0],
                    "temperature_2m_mean": [14.0],
                    "precipitation_sum": [0.0],
                }
            })

        monkeypatch.setattr("requests.get", mock_get)

        runner = app.test_cli_runner()
        # Run fetch-weather for 90 days - all 90 days are already cached!
        res = runner.invoke(args=["fetch-weather", "--history-days", "90", "--forecast-days", "0"])
        assert res.exit_code == 0
        assert "cached, skipping" in res.output
        # Since archive days are cached and forecast-days=0, no archive API calls should have been made
        archive_calls = [c for c in api_calls if "archive" in c["url"]]
        assert len(archive_calls) == 0


def test_reset_demo_and_tomorrow_forecast(app, monkeypatch):
    """reset-demo sets up fresh 730-day history; /api/forecast for tomorrow has fresh lag and non-fallback weather."""
    from tests.conftest import MockResponse

    # Mock weather API for forecast
    mock_forecast = {
        "daily": {
            "time": [(date.today() + timedelta(days=1)).isoformat()],
            "temperature_2m_max": [16.0],
            "temperature_2m_min": [9.0],
            "temperature_2m_mean": [12.5],
            "precipitation_sum": [0.0],
        }
    }
    monkeypatch.setattr("requests.get", lambda url, params=None, timeout=None: MockResponse(mock_forecast))

    runner = app.test_cli_runner()
    res = runner.invoke(args=["reset-demo", "--history-days", "730"])
    assert res.exit_code == 0
    assert "Demo reset complete" in res.output

    with app.app_context():
        # Check DB rows per item > 500
        df = load_from_db()
        assert len(df) == 730 * 5
        counts = df["item"].value_counts()
        for item_name, count in counts.items():
            assert count >= 500

        # Query forecast for tomorrow
        tomorrow_str = (date.today() + timedelta(days=1)).isoformat()
        db = get_db()
        item = db.execute("SELECT id FROM items WHERE name = 'Croissant'").fetchone()

        pred_res = predict_item_demand(item["id"], tomorrow_str, shop_id=1)
        assert pred_res["lag_features_stale"] is False
        assert pred_res["weather_source"] == "forecast"
        assert pred_res["weather_source"] != "fallback"

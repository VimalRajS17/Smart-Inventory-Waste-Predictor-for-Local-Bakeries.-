"""Tests for Open-Meteo weather service: parsing, thresholding, caching, refresh, and resilience."""

from datetime import date, datetime, timedelta, timezone
import pytest
import requests

from app.db import get_db
from app.services.weather import (
    fetch_forecast,
    fetch_history,
    get_weather_for_date,
    save_weather,
    _normalize_weather_response,
)
from tests.conftest import MockResponse


def test_weather_parsing_and_thresholding(monkeypatch):
    """Test parsing of Open-Meteo daily response, rain threshold calculation, and mean computation."""
    monkeypatch.setenv("RAIN_THRESHOLD_MM", "1.5")

    raw_response = {
        "daily": {
            "time": ["2026-10-30", "2026-10-31"],
            "temperature_2m_max": [20.0, 18.0],
            "temperature_2m_min": [10.0, 8.0],
            "temperature_2m_mean": [15.0, None],  # Second day missing mean
            "precipitation_sum": [1.0, 2.5],      # 1.0 < 1.5 (dry), 2.5 >= 1.5 (rainy)
        }
    }

    rows = _normalize_weather_response(raw_response)
    assert len(rows) == 2

    # Day 1: provided mean, dry
    assert rows[0]["date"] == "2026-10-30"
    assert rows[0]["temp_mean_c"] == 15.0
    assert rows[0]["temp_max_c"] == 20.0
    assert rows[0]["temp_min_c"] == 10.0
    assert rows[0]["precip_mm"] == 1.0
    assert rows[0]["is_rainy"] == 0

    # Day 2: computed mean from (18.0 + 8.0) / 2 = 13.0, rainy
    assert rows[1]["date"] == "2026-10-31"
    assert rows[1]["temp_mean_c"] == 13.0
    assert rows[1]["precip_mm"] == 2.5
    assert rows[1]["is_rainy"] == 1


def test_fetch_forecast_with_mocked_network(monkeypatch):
    """Test fetch_forecast successfully queries endpoint and parses data."""
    mock_data = {
        "daily": {
            "time": ["2026-10-30"],
            "temperature_2m_max": [16.0],
            "temperature_2m_min": [8.0],
            "temperature_2m_mean": [12.0],
            "precipitation_sum": [0.0],
        }
    }

    def mock_get(url, params=None, timeout=None):
        assert "api.open-meteo.com/v1/forecast" in url
        assert params["latitude"] == 51.5074
        assert params["forecast_days"] == 7
        return MockResponse(mock_data)

    monkeypatch.setattr(requests, "get", mock_get)

    result = fetch_forecast(51.5074, -0.1278, days=7)
    assert len(result) == 1
    assert result[0]["date"] == "2026-10-30"
    assert result[0]["temp_mean_c"] == 12.0
    assert result[0]["is_rainy"] == 0


def test_fetch_history_with_mocked_network(monkeypatch):
    """Test fetch_history queries archive endpoint."""
    mock_data = {
        "daily": {
            "time": ["2026-10-01"],
            "temperature_2m_max": [19.0],
            "temperature_2m_min": [11.0],
            "temperature_2m_mean": [15.0],
            "precipitation_sum": [4.0],
        }
    }

    def mock_get(url, params=None, timeout=None):
        assert "archive-api.open-meteo.com/v1/archive" in url
        assert params["start_date"] == "2026-10-01"
        assert params["end_date"] == "2026-10-01"
        return MockResponse(mock_data)

    monkeypatch.setattr(requests, "get", mock_get)

    result = fetch_history(51.5074, -0.1278, "2026-10-01", "2026-10-01")
    assert len(result) == 1
    assert result[0]["temp_mean_c"] == 15.0
    assert result[0]["is_rainy"] == 1


def test_weather_cache_hit(app):
    """Test get_weather_for_date returns cached row without calling network."""
    with app.app_context():
        db = get_db()
        shop_id = db.execute("SELECT id FROM shops LIMIT 1").fetchone()["id"]

        # Insert fresh forecast row
        now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
        target_date = "2026-11-01"
        db.execute(
            """
            INSERT INTO weather_daily (shop_id, date, temperature_c, temp_mean_c, temp_max_c, temp_min_c, precip_mm, is_rainy, source, fetched_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (shop_id, target_date, 14.5, 14.5, 18.0, 11.0, 0.2, 0, "forecast", now_str),
        )
        db.commit()

        # Call get_weather_for_date (no network mock provided, no_real_network would fail if invoked)
        row = get_weather_for_date(shop_id, target_date)
        assert row is not None
        assert row["date"] == target_date
        assert row["temp_mean_c"] == 14.5
        assert row["source"] == "forecast"


def test_weather_cache_miss_and_fetch(app, monkeypatch):
    """Test get_weather_for_date on cache miss queries Open-Meteo, saves row, and returns."""
    target_date = "2026-11-05"

    mock_data = {
        "daily": {
            "time": [target_date],
            "temperature_2m_max": [15.0],
            "temperature_2m_min": [7.0],
            "temperature_2m_mean": [11.0],
            "precipitation_sum": [3.2],
        }
    }

    called = []
    def mock_get(url, params=None, timeout=None):
        called.append(url)
        return MockResponse(mock_data)

    monkeypatch.setattr(requests, "get", mock_get)

    with app.app_context():
        db = get_db()
        shop_id = db.execute("SELECT id FROM shops LIMIT 1").fetchone()["id"]

        result = get_weather_for_date(shop_id, target_date)
        assert len(called) == 1
        assert result is not None
        assert result["date"] == target_date
        assert result["temp_mean_c"] == 11.0
        assert result["is_rainy"] == 1

        # Check DB has record
        db_row = db.execute("SELECT * FROM weather_daily WHERE shop_id = ? AND date = ?", (shop_id, target_date)).fetchone()
        assert db_row is not None
        assert db_row["temp_mean_c"] == 11.0


def test_weather_stale_forecast_refresh(app, monkeypatch):
    """Test that a forecast fetched > 6 hours ago triggers a refresh."""
    target_date = (date.today() + timedelta(days=2)).isoformat()

    with app.app_context():
        db = get_db()
        shop_id = db.execute("SELECT id FROM shops LIMIT 1").fetchone()["id"]

        # Insert stale forecast fetched 8 hours ago
        stale_time = (datetime.now(timezone.utc) - timedelta(hours=8)).strftime("%Y-%m-%d %H:%M:%S")
        db.execute(
            """
            INSERT INTO weather_daily (shop_id, date, temperature_c, temp_mean_c, temp_max_c, temp_min_c, precip_mm, is_rainy, source, fetched_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (shop_id, target_date, 10.0, 10.0, 14.0, 6.0, 0.0, 0, "forecast", stale_time),
        )
        db.commit()

        # Mock fresh weather response
        mock_data = {
            "daily": {
                "time": [target_date],
                "temperature_2m_max": [17.0],
                "temperature_2m_min": [9.0],
                "temperature_2m_mean": [13.0],
                "precipitation_sum": [0.0],
            }
        }
        refresh_called = []
        def mock_get(url, params=None, timeout=None):
            refresh_called.append(url)
            return MockResponse(mock_data)

        monkeypatch.setattr(requests, "get", mock_get)

        refreshed = get_weather_for_date(shop_id, target_date)
        assert len(refresh_called) == 1
        assert refreshed["temp_mean_c"] == 13.0


def test_weather_failure_fallback_returns_none(app, monkeypatch):
    """Test that on network failure for missing date, get_weather_for_date logs warning and returns None without raising."""
    def mock_fail(url, params=None, timeout=None):
        raise requests.ConnectionError("Connection timed out (simulated)")

    monkeypatch.setattr(requests, "get", mock_fail)

    with app.app_context():
        db = get_db()
        shop_id = db.execute("SELECT id FROM shops LIMIT 1").fetchone()["id"]

        # Date not in DB, network fails
        res = get_weather_for_date(shop_id, "2029-05-10")
        assert res is None


def test_weather_retries_with_backoff(monkeypatch):
    """Test retry mechanism on 500 server error before succeeding."""
    calls = []

    def mock_retry_get(url, params=None, timeout=None):
        calls.append(len(calls) + 1)
        if len(calls) < 2:
            return MockResponse({}, status_code=500, text="Internal Server Error")
        return MockResponse({
            "daily": {
                "time": ["2026-11-01"],
                "temperature_2m_max": [15.0],
                "temperature_2m_min": [5.0],
                "temperature_2m_mean": [10.0],
                "precipitation_sum": [0.0],
            }
        })

    # Speed up backoff sleep for tests
    monkeypatch.setattr("time.sleep", lambda s: None)
    monkeypatch.setattr(requests, "get", mock_retry_get)

    result = fetch_forecast(51.5074, -0.1278, days=1)
    assert len(calls) == 2  # 1 failure + 1 successful retry
    assert len(result) == 1
    assert result[0]["temp_mean_c"] == 10.0


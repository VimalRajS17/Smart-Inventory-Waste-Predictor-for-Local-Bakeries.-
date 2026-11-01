"""Open-Meteo weather service for historical data and forecasts."""

import logging
import os
import time
from datetime import date, datetime, timedelta, timezone

import click
from flask.cli import with_appcontext
import requests

from app.db import get_db

logger = logging.getLogger(__name__)

FORECAST_API_URL = "https://api.open-meteo.com/v1/forecast"
ARCHIVE_API_URL = "https://archive-api.open-meteo.com/v1/archive"
REQUEST_TIMEOUT = 10
MAX_RETRIES = 2


def _get_rain_threshold() -> float:
    """Return rain threshold in mm from environment (default 1.0)."""
    try:
        return float(os.getenv("RAIN_THRESHOLD_MM", "1.0"))
    except (ValueError, TypeError):
        return 1.0


def _request_with_retries(url: str, params: dict, timeout: int = REQUEST_TIMEOUT, max_retries: int = MAX_RETRIES) -> dict:
    """Execute HTTP GET with timeout and exponential backoff retries."""
    last_exc = None
    for attempt in range(max_retries + 1):
        try:
            resp = requests.get(url, params=params, timeout=timeout)
            if resp.status_code == 200:
                return resp.json()
            elif resp.status_code in (429, 500, 502, 503, 504):
                last_exc = requests.HTTPError(
                    f"HTTP {resp.status_code} from {url}: {resp.text[:100]}", response=resp
                )
            else:
                resp.raise_for_status()
        except requests.RequestException as exc:
            last_exc = exc

        if attempt < max_retries:
            delay = 1.0 * (2 ** attempt)
            time.sleep(delay)

    raise last_exc or requests.RequestException(f"Failed to fetch data from {url}")


def _normalize_weather_response(data: dict) -> list[dict]:
    """Normalize Open-Meteo daily arrays into standard dict rows."""
    daily = data.get("daily", {})
    dates = daily.get("time", [])
    max_temps = daily.get("temperature_2m_max", [])
    min_temps = daily.get("temperature_2m_min", [])
    mean_temps = daily.get("temperature_2m_mean", [])
    precips = daily.get("precipitation_sum", [])

    threshold = _get_rain_threshold()
    rows = []
    for i, date_str in enumerate(dates):
        max_t = max_temps[i] if i < len(max_temps) and max_temps[i] is not None else None
        min_t = min_temps[i] if i < len(min_temps) and min_temps[i] is not None else None
        mean_t = mean_temps[i] if i < len(mean_temps) and mean_temps[i] is not None else None

        # Compute mean temperature if not returned by endpoint
        if mean_t is not None:
            temp_mean = round(float(mean_t), 1)
        elif max_t is not None and min_t is not None:
            temp_mean = round((float(max_t) + float(min_t)) / 2.0, 1)
        else:
            temp_mean = None

        temp_max = round(float(max_t), 1) if max_t is not None else None
        temp_min = round(float(min_t), 1) if min_t is not None else None
        precip = round(float(precips[i]), 1) if i < len(precips) and precips[i] is not None else 0.0
        is_rainy = 1 if precip >= threshold else 0

        rows.append({
            "date": date_str,
            "temp_mean_c": temp_mean,
            "temp_max_c": temp_max,
            "temp_min_c": temp_min,
            "precip_mm": precip,
            "is_rainy": is_rainy,
        })
    return rows


def fetch_forecast(lat: float, lon: float, days: int = 7) -> list[dict]:
    """Fetch forecast weather data from Open-Meteo Forecast API."""
    params = {
        "latitude": lat,
        "longitude": lon,
        "daily": ["temperature_2m_max", "temperature_2m_min", "temperature_2m_mean", "precipitation_sum"],
        "forecast_days": days,
        "timezone": "auto",
    }
    data = _request_with_retries(FORECAST_API_URL, params=params)
    return _normalize_weather_response(data)


def fetch_history(lat: float, lon: float, start_date: str, end_date: str) -> list[dict]:
    """Fetch historical weather data from Open-Meteo Archive API."""
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date,
        "end_date": end_date,
        "daily": ["temperature_2m_max", "temperature_2m_min", "temperature_2m_mean", "precipitation_sum"],
        "timezone": "auto",
    }
    data = _request_with_retries(ARCHIVE_API_URL, params=params)
    return _normalize_weather_response(data)


def save_weather(shop_id: int, rows: list[dict], source: str):
    """Upsert weather rows into weather_daily table with source and fetched_at."""
    if not rows:
        return
    db = get_db()
    now_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")

    upsert_data = [
        (
            shop_id,
            r["date"],
            r["temp_mean_c"],  # temperature_c
            r["temp_mean_c"],
            r["temp_max_c"],
            r["temp_min_c"],
            r["precip_mm"],
            r["is_rainy"],
            source,
            now_utc,
        )
        for r in rows
    ]

    db.executemany(
        """
        INSERT INTO weather_daily (
            shop_id, date, temperature_c, temp_mean_c, temp_max_c, temp_min_c,
            precip_mm, is_rainy, source, fetched_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(shop_id, date) DO UPDATE SET
            temperature_c = excluded.temperature_c,
            temp_mean_c = excluded.temp_mean_c,
            temp_max_c = excluded.temp_max_c,
            temp_min_c = excluded.temp_min_c,
            precip_mm = excluded.precip_mm,
            is_rainy = excluded.is_rainy,
            source = excluded.source,
            fetched_at = excluded.fetched_at
        """,
        upsert_data,
    )
    db.commit()


def _is_forecast_stale(fetched_at_val, max_age_hours: int = 6) -> bool:
    """Check whether a forecast row was fetched more than max_age_hours ago."""
    if not fetched_at_val:
        return True
    try:
        if isinstance(fetched_at_val, str):
            cleaned = fetched_at_val.replace("T", " ")
            dt = datetime.fromisoformat(cleaned)
        elif isinstance(fetched_at_val, datetime):
            dt = fetched_at_val
        else:
            return True

        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return (datetime.now(timezone.utc) - dt) > timedelta(hours=max_age_hours)
    except Exception:
        return True


def get_weather_for_date(shop_id: int, target_date: str) -> dict | None:
    """Retrieve weather for shop and date from cache, refreshing if missing or stale (>6h forecast).
    
    Returns None on any network or parsing failure, logs a warning, and never raises to caller.
    """
    db = get_db()
    cached = db.execute(
        """
        SELECT * FROM weather_daily
        WHERE shop_id = ? AND date = ?
        """,
        (shop_id, target_date),
    ).fetchone()

    # If cached archive row, historical data is final
    if cached and cached["source"] == "archive":
        return dict(cached)

    # If cached forecast row fetched within 6 hours, return it
    if cached and cached["source"] == "forecast" and not _is_forecast_stale(cached["fetched_at"], max_age_hours=6):
        return dict(cached)

    # Missing or stale forecast row: try to refresh
    try:
        shop = db.execute(
            "SELECT lat, lon FROM shops WHERE id = ?", (shop_id,)
        ).fetchone()
        if not shop:
            logger.warning("Shop ID %s not found when refreshing weather.", shop_id)
            return dict(cached) if cached else None

        lat, lon = float(shop["lat"]), float(shop["lon"])
        today_str = date.today().isoformat()

        if target_date < today_str:
            rows = fetch_history(lat, lon, target_date, target_date)
            source = "archive"
        else:
            rows = fetch_forecast(lat, lon, days=7)
            source = "forecast"

        save_weather(shop_id, rows, source)

        refreshed = db.execute(
            """
            SELECT * FROM weather_daily
            WHERE shop_id = ? AND date = ?
            """,
            (shop_id, target_date),
        ).fetchone()
        return dict(refreshed) if refreshed else None

    except Exception as exc:
        logger.warning(
            "Failed to fetch weather for shop_id=%s date=%s: %s",
            shop_id, target_date, exc, exc_info=True
        )
        return dict(cached) if cached else None


def sync_weather_for_date_range(shop_id: int, start_date_str: str, end_date_str: str) -> dict:
    """Synchronize weather data (archive or forecast) for a shop across a date range.
    
    Checks cached dates, splits missing dates into 90-day archive chunks, and fetches
    from Open-Meteo. Never raises exceptions to caller; returns a summary dict.
    """
    db = get_db()
    shop = db.execute("SELECT id, name, lat, lon FROM shops WHERE id = ?", (shop_id,)).fetchone()
    if not shop:
        return {"status": "error", "message": f"Shop {shop_id} not found", "days_fetched": 0}

    try:
        start_date = date.fromisoformat(start_date_str)
        end_date = date.fromisoformat(end_date_str)
    except (ValueError, TypeError) as e:
        return {"status": "error", "message": f"Invalid date format: {e}", "days_fetched": 0}

    if start_date > end_date:
        start_date, end_date = end_date, start_date

    lat, lon = float(shop["lat"]), float(shop["lon"])
    today = date.today()
    days_fetched = 0

    try:
        # 1. Handle historical dates (< today)
        hist_start = start_date
        hist_end = min(end_date, today - timedelta(days=1))

        if hist_start <= hist_end:
            cached_rows = db.execute(
                """
                SELECT date FROM weather_daily
                WHERE shop_id = ? AND date >= ? AND date <= ?
                  AND (temp_mean_c IS NOT NULL OR temperature_c IS NOT NULL)
                """,
                (shop_id, hist_start.isoformat(), hist_end.isoformat()),
            ).fetchall()
            cached_dates = set(r["date"] for r in cached_rows)

            chunk_size = 90
            curr_start = hist_start
            chunks = []
            while curr_start <= hist_end:
                curr_end = min(curr_start + timedelta(days=chunk_size - 1), hist_end)
                chunks.append((curr_start, curr_end))
                curr_start = curr_end + timedelta(days=1)

            for idx, (c_start, c_end) in enumerate(chunks):
                c_dates = [
                    (c_start + timedelta(days=d)).isoformat()
                    for d in range((c_end - c_start).days + 1)
                ]
                missing = [d for d in c_dates if d not in cached_dates]
                if not missing:
                    continue

                try:
                    chunk_rows = fetch_history(lat, lon, c_start.isoformat(), c_end.isoformat())
                    save_weather(shop_id, chunk_rows, "archive")
                    days_fetched += len(chunk_rows)
                    for r in chunk_rows:
                        cached_dates.add(r["date"])
                except Exception as e:
                    logger.warning("Failed to fetch archive weather chunk %s to %s: %s", c_start, c_end, e)

                if idx < len(chunks) - 1:
                    time.sleep(0.2)

        # 2. Handle forecast/today dates (>= today)
        if end_date >= today:
            fc_days = min(16, (end_date - today).days + 1)
            fc_days = max(fc_days, 7)
            try:
                forecast_rows = fetch_forecast(lat, lon, days=fc_days)
                save_weather(shop_id, forecast_rows, "forecast")
                days_fetched += len(forecast_rows)
            except Exception as e:
                logger.warning("Failed to fetch forecast weather: %s", e)

        return {
            "status": "ok",
            "start_date": start_date_str,
            "end_date": end_date_str,
            "days_fetched": days_fetched,
        }
    except Exception as exc:
        logger.warning("Error synchronizing weather for range %s to %s: %s", start_date_str, end_date_str, exc)
        return {
            "status": "partial_or_error",
            "message": str(exc),
            "days_fetched": days_fetched,
        }



# ---------------------------------------------------------------------------
# Flask CLI Command
# ---------------------------------------------------------------------------

@click.command("fetch-weather")
@click.option("--history-days", default=30, type=int, help="Number of past days to fetch from archive API (up to 730).")
@click.option("--forecast-days", default=7, type=int, help="Number of forecast days to fetch from forecast API.")
@with_appcontext
def fetch_weather_command(history_days: int, forecast_days: int):
    """Fetch archive and forecast weather for registered shops and cache in database."""
    db = get_db()
    shops = db.execute("SELECT id, name, lat, lon FROM shops").fetchall()
    if not shops:
        click.echo("No shops found in database. Run 'flask --app app seed-demo' first.")
        return

    today = date.today()
    history_days = max(0, min(730, history_days))
    forecast_days = max(0, min(16, forecast_days))

    for shop in shops:
        shop_id = shop["id"]
        shop_name = shop["name"]
        lat, lon = float(shop["lat"]), float(shop["lon"])
        click.echo(f"Fetching weather for shop '{shop_name}' ({lat}, {lon})...")

        # 1. Fetch history in 90-day chunks, skipping cached days
        history_count = 0
        if history_days > 0:
            start_date = today - timedelta(days=history_days)
            end_date = today - timedelta(days=1)

            # Query existing dates in range
            cached_rows = db.execute(
                """
                SELECT date FROM weather_daily
                WHERE shop_id = ? AND date >= ? AND date <= ?
                  AND (temp_mean_c IS NOT NULL OR temperature_c IS NOT NULL)
                """,
                (shop_id, start_date.isoformat(), end_date.isoformat()),
            ).fetchall()
            cached_dates = set(r["date"] for r in cached_rows)

            # Build 90-day chunks
            chunk_size = 90
            curr_start = start_date
            chunks = []
            while curr_start <= end_date:
                curr_end = min(curr_start + timedelta(days=chunk_size - 1), end_date)
                chunks.append((curr_start, curr_end))
                curr_start = curr_end + timedelta(days=1)

            for idx, (c_start, c_end) in enumerate(chunks):
                # Check missing dates in chunk
                c_dates = [
                    (c_start + timedelta(days=d)).isoformat()
                    for d in range((c_end - c_start).days + 1)
                ]
                missing = [d for d in c_dates if d not in cached_dates]

                if not missing:
                    click.echo(f"  Archive chunk {c_start} to {c_end}: all {len(c_dates)} days cached, skipping.")
                    continue

                click.echo(f"  Fetching archive chunk {c_start} to {c_end} ({len(missing)} days uncached)...")
                try:
                    chunk_rows = fetch_history(lat, lon, c_start.isoformat(), c_end.isoformat())
                    save_weather(shop_id, chunk_rows, "archive")
                    history_count += len(chunk_rows)
                    for r in chunk_rows:
                        cached_dates.add(r["date"])
                except Exception as e:
                    click.echo(f"  Warning: failed to fetch archive chunk {c_start} to {c_end}: {e}")

                if idx < len(chunks) - 1:
                    time.sleep(0.5)

        # 2. Fetch forecast if forecast_days > 0
        forecast_count = 0
        if forecast_days > 0:
            try:
                forecast_rows = fetch_forecast(lat, lon, days=forecast_days)
                save_weather(shop_id, forecast_rows, "forecast")
                forecast_count = len(forecast_rows)
            except Exception as e:
                click.echo(f"  Warning: failed to fetch forecast weather: {e}")

        click.echo(
            f"Successfully updated weather for '{shop_name}': {history_count} archive days fetched, {forecast_count} forecast days fetched."
        )


"""Data extraction and preparation for Loaf Ledger ML pipeline."""

import os
import sqlite3
import pandas as pd
from app.db import get_db

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_training_data(shop_id: int, item_id: int | None = None, db=None) -> pd.DataFrame:
    """Load historical sales joined with weather data for a shop from SQLite."""
    if db is None:
        db = get_db()

    query = """
    SELECT
        d.date,
        d.shop_id,
        d.item_id,
        i.name AS item,
        d.sold AS quantity_sold,
        COALESCE(w.temp_mean_c, w.temperature_c, 20.0) AS temperature,
        COALESCE(w.is_rainy, 0) AS rain,
        COALESCE(w.is_holiday, 0) AS holiday
    FROM daily_log d
    JOIN items i ON d.item_id = i.id AND d.shop_id = i.shop_id
    LEFT JOIN weather_daily w ON d.shop_id = w.shop_id AND d.date = w.date
    WHERE d.shop_id = ?
      AND d.sold IS NOT NULL
    """
    params = [shop_id]
    if item_id is not None:
        query += " AND d.item_id = ?"
        params.append(item_id)

    query += " ORDER BY d.item_id ASC, d.date ASC"

    rows = db.execute(query, params).fetchall()
    if not rows:
        return pd.DataFrame(columns=[
            "date", "shop_id", "item_id", "item", "quantity_sold",
            "temperature", "rain", "holiday"
        ])

    data = [dict(r) for r in rows]
    df = pd.DataFrame(data)
    df["date"] = pd.to_datetime(df["date"])
    df["quantity_sold"] = df["quantity_sold"].astype(float)
    df["temperature"] = df["temperature"].astype(float)
    df["rain"] = df["rain"].astype(int)
    df["holiday"] = df["holiday"].astype(int)
    return df


def load_from_csv(csv_path: str = None) -> pd.DataFrame:
    """Load historical sales dataframe from CSV (backwards compatibility)."""
    if csv_path is None:
        csv_path = os.path.join(PROJECT_ROOT, "data", "sales_history.csv")
    df = pd.read_csv(csv_path)
    df["date"] = pd.to_datetime(df["date"])
    if "temperature_c" in df.columns and "temperature" not in df.columns:
        df["temperature"] = df["temperature_c"]
    if "is_rainy" in df.columns and "rain" not in df.columns:
        df["rain"] = df["is_rainy"]
    if "is_holiday" in df.columns and "holiday" not in df.columns:
        df["holiday"] = df["is_holiday"]
    return df


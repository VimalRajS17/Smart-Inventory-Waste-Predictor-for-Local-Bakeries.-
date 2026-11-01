"""
Trains and compares demand-forecasting models for Loaf Ledger,
then saves the best model + a metrics report.

Supports loading data from CSV or from the SQLite database.
"""

import argparse
import json
import os
import sqlite3

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "sales_history.csv")
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")
DB_PATH = os.path.join(PROJECT_ROOT, os.getenv("DATABASE_PATH", "bakery.db"))
TEST_FRACTION = 0.15  # last 15% of days per item held out for testing

os.makedirs(MODEL_DIR, exist_ok=True)


def load_from_db(db_path: str = None) -> pd.DataFrame:
    """Build training dataframe from daily_log joined with weather_daily."""
    if db_path is None:
        try:
            from flask import current_app
            db_path = current_app.config.get("DATABASE")
        except Exception:
            db_path = None

        if db_path is None:
            db_env = os.getenv("DATABASE_PATH", "bakery.db")
            db_path = db_env if os.path.isabs(db_env) else os.path.join(PROJECT_ROOT, db_env)

    conn = sqlite3.connect(db_path)
    query = """
    SELECT
        d.date,
        i.name AS item,
        d.sold AS quantity_sold,
        w.temperature_c,
        w.is_rainy,
        w.is_holiday
    FROM daily_log d
    JOIN items i ON d.item_id = i.id
    JOIN weather_daily w ON d.shop_id = w.shop_id AND d.date = w.date
    WHERE d.sold IS NOT NULL
      AND w.temperature_c IS NOT NULL
    ORDER BY d.date ASC
    """
    df = pd.read_sql_query(query, conn)
    conn.close()

    df["date"] = pd.to_datetime(df["date"])
    df["day_of_week"] = df["date"].dt.dayofweek
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)

    print(f"Loaded {len(df)} rows from database (daily_log joined with weather_daily)")
    if not df.empty and "item" in df.columns:
        for it_name, count in df["item"].value_counts().items():
            print(f"  {it_name}: {count} rows")
    return df


def load_from_csv(csv_path: str = DATA_PATH) -> pd.DataFrame:
    """Load training dataframe from raw CSV."""
    df = pd.read_csv(csv_path)
    df["date"] = pd.to_datetime(df["date"])
    print(f"Loaded {len(df)} rows from CSV: {csv_path}")
    return df


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add lag and rolling-average features per item (sorted by date)."""
    df = df.sort_values(["item", "date"]).reset_index(drop=True)
    df["date"] = pd.to_datetime(df["date"])

    df["lag_1"] = df.groupby("item")["quantity_sold"].shift(1)
    df["lag_7"] = df.groupby("item")["quantity_sold"].shift(7)
    df["rolling_mean_7"] = (
        df.groupby("item")["quantity_sold"]
        .shift(1)
        .rolling(7)
        .mean()
        .reset_index(level=0, drop=True)
    )
    df["month"] = df["date"].dt.month
    return df.dropna().reset_index(drop=True)


def moving_average_predict(df: pd.DataFrame) -> np.ndarray:
    """Naive baseline: predict today's demand as the 7-day rolling mean."""
    return df["rolling_mean_7"].values


def evaluate(y_true, y_pred, name: str) -> dict:
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mask = y_true != 0
    mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    print(f"{name:20s} | MAE: {mae:6.2f} | RMSE: {rmse:6.2f} | MAPE: {mape:6.2f}%")
    return {"model": name, "mae": round(mae, 2), "rmse": round(rmse, 2), "mape": round(mape, 2)}


def train(source: str = "csv"):
    if source == "db":
        raw = load_from_db()
    else:
        raw = load_from_csv()

    if raw.empty:
        raise ValueError(f"No training data found from source '{source}'.")

    df = add_features(raw)

    feature_cols = [
        "day_of_week", "is_weekend", "temperature_c", "is_rainy",
        "is_holiday", "month", "lag_1", "lag_7", "rolling_mean_7",
    ]

    # Chronological split per item (no data leakage)
    train_frames, test_frames = [], []
    for item, group in df.groupby("item"):
        split_idx = int(len(group) * (1 - TEST_FRACTION))
        train_frames.append(group.iloc[:split_idx])
        test_frames.append(group.iloc[split_idx:])
    train_df = pd.concat(train_frames).reset_index(drop=True)
    test_df = pd.concat(test_frames).reset_index(drop=True)

    # One-hot encode item
    train_encoded = pd.get_dummies(train_df, columns=["item"])
    test_encoded = pd.get_dummies(test_df, columns=["item"])
    test_encoded = test_encoded.reindex(columns=train_encoded.columns, fill_value=0)

    item_dummy_cols = [c for c in train_encoded.columns if c.startswith("item_")]
    X_cols = feature_cols + item_dummy_cols

    X_train, y_train = train_encoded[X_cols], train_encoded["quantity_sold"]
    X_test, y_test = test_encoded[X_cols], test_encoded["quantity_sold"]

    results = []

    # 1. Moving average baseline
    ma_pred = moving_average_predict(test_df)
    results.append(evaluate(y_test.values, ma_pred, "Moving Average"))

    # 2. Linear Regression
    lin_model = LinearRegression()
    lin_model.fit(X_train, y_train)
    lin_pred = lin_model.predict(X_test)
    results.append(evaluate(y_test.values, lin_pred, "Linear Regression"))

    # 3. Random Forest
    rf_model = RandomForestRegressor(n_estimators=200, max_depth=8, random_state=42)
    rf_model.fit(X_train, y_train)
    rf_pred = rf_model.predict(X_test)
    results.append(evaluate(y_test.values, rf_pred, "Random Forest"))

    best = min(results[1:], key=lambda r: r["mae"])
    best_model = rf_model if best["model"] == "Random Forest" else lin_model

    joblib.dump(best_model, os.path.join(MODEL_DIR, "best_model.joblib"))
    joblib.dump(X_cols, os.path.join(MODEL_DIR, "feature_columns.joblib"))

    with open(os.path.join(MODEL_DIR, "metrics.json"), "w") as f:
        json.dump({"results": results, "best_model": best["model"]}, f, indent=2)

    print(f"\nBest model: {best['model']} — saved to models/best_model.joblib")


def main():
    parser = argparse.ArgumentParser(description="Train demand-forecasting models for Loaf Ledger.")
    parser.add_argument(
        "--source",
        choices=["csv", "db"],
        default="csv",
        help="Source of training data: 'csv' (default) or 'db' (database tables daily_log and weather_daily)",
    )
    args = parser.parse_args()
    train(source=args.source)


if __name__ == "__main__":
    main()


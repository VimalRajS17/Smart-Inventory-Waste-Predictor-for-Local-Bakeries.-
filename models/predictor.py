"""Loads the trained model and generates a demand prediction for a given
item and date, using recent sales history for lag/rolling-average features."""

import os

import joblib
import pandas as pd

MODEL_DIR = os.path.dirname(__file__)
DATA_PATH = os.path.join(os.path.dirname(MODEL_DIR), "data", "sales_history.csv")

_model = None
_feature_cols = None


def _load():
    global _model, _feature_cols
    if _model is None:
        _model = joblib.load(os.path.join(MODEL_DIR, "best_model.joblib"))
        _feature_cols = joblib.load(os.path.join(MODEL_DIR, "feature_columns.joblib"))
    return _model, _feature_cols


def predict_demand(item: str, target_date: str, temperature_c: float = 25.0,
                    is_rainy: int = 0, is_holiday: int = 0) -> float:
    """Predict quantity to be sold for `item` on `target_date`.

    Uses the historical CSV to compute recent lag/rolling features for that
    item. In a production system, this would pull from the live `sales`
    table in the database instead of the static CSV.
    """
    model, feature_cols = _load()

    history = pd.read_csv(DATA_PATH)
    history["date"] = pd.to_datetime(history["date"])
    item_history = history[history["item"] == item].sort_values("date")

    if item_history.empty:
        raise ValueError(f"No history found for item: {item}")

    target = pd.to_datetime(target_date)
    day_of_week = target.dayofweek
    is_weekend = int(day_of_week >= 5)
    month = target.month

    lag_1 = item_history["quantity_sold"].iloc[-1]
    lag_7 = item_history["quantity_sold"].iloc[-7] if len(item_history) >= 7 else lag_1
    rolling_mean_7 = item_history["quantity_sold"].tail(7).mean()

    row = {col: 0 for col in feature_cols}
    row.update({
        "day_of_week": day_of_week,
        "is_weekend": is_weekend,
        "temperature_c": temperature_c,
        "is_rainy": is_rainy,
        "is_holiday": is_holiday,
        "month": month,
        "lag_1": lag_1,
        "lag_7": lag_7,
        "rolling_mean_7": rolling_mean_7,
    })
    item_col = f"item_{item}"
    if item_col in row:
        row[item_col] = 1

    X = pd.DataFrame([row])[feature_cols]
    prediction = model.predict(X)[0]
    return max(0, round(float(prediction), 1))

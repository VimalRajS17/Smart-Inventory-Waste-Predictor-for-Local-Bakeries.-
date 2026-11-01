"""Feature engineering for bakery item demand forecasting.

Constructs calendar, weather, lag, and rolling statistics per item with strict
time-series non-leakage (features for day t use only data prior to day t).
"""

from datetime import date
import numpy as np
import pandas as pd

FEATURE_COLUMNS = [
    "day_of_week",
    "month",
    "weekend",
    "holiday",
    "temperature",
    "rain",
    "lag_1",
    "lag_7",
    "lag_14",
    "rolling_mean_7",
    "rolling_mean_28",
    "same_weekday_mean_4w",
]


def build_features(df: pd.DataFrame, dropna: bool = True) -> pd.DataFrame:
    """Generate features for all items in dataframe.
    
    Ensures zero data leakage: for row t, all lag and rolling features are computed
    strictly using observations from days prior to day t.
    """
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    group_col = "item_id" if "item_id" in df.columns else "item"
    df = df.sort_values([group_col, "date"]).reset_index(drop=True)

    # 1. Calendar Features
    df["day_of_week"] = df["date"].dt.dayofweek
    df["month"] = df["date"].dt.month
    df["weekend"] = (df["day_of_week"] >= 5).astype(int)
    df["is_weekend"] = df["weekend"]

    # 2. Weather & Holiday Features
    if "temperature" not in df.columns:
        if "temp_mean_c" in df.columns:
            df["temperature"] = df["temp_mean_c"]
        elif "temperature_c" in df.columns:
            df["temperature"] = df["temperature_c"]
        else:
            df["temperature"] = 20.0

    if "rain" not in df.columns:
        if "is_rainy" in df.columns:
            df["rain"] = df["is_rainy"]
        elif "precip_mm" in df.columns:
            df["rain"] = (df["precip_mm"] > 0).astype(int)
        else:
            df["rain"] = 0

    if "holiday" not in df.columns:
        if "is_holiday" in df.columns:
            df["holiday"] = df["is_holiday"]
        else:
            df["holiday"] = 0

    df["temperature"] = df["temperature"].astype(float)
    df["rain"] = df["rain"].astype(int)
    df["holiday"] = df["holiday"].astype(int)

    # Aliases
    df["temperature_c"] = df["temperature"]
    df["is_rainy"] = df["rain"]
    df["is_holiday"] = df["holiday"]

    # 3. Lags & Rolling Means (strictly shifted to day t-1 or earlier)
    # lag_1: day t-1
    df["lag_1"] = df.groupby(group_col)["quantity_sold"].shift(1)
    # lag_7: day t-7 (same day of week last week)
    df["lag_7"] = df.groupby(group_col)["quantity_sold"].shift(7)
    # lag_14: day t-14 (same day of week 2 weeks ago)
    df["lag_14"] = df.groupby(group_col)["quantity_sold"].shift(14)

    # rolling_mean_7: rolling average of prior 7 days (t-7 to t-1)
    df["rolling_mean_7"] = (
        df.groupby(group_col)["quantity_sold"]
        .shift(1)
        .rolling(7, min_periods=1)
        .mean()
        .reset_index(level=0, drop=True)
    )

    # rolling_mean_28: rolling average of prior 28 days (t-28 to t-1)
    df["rolling_mean_28"] = (
        df.groupby(group_col)["quantity_sold"]
        .shift(1)
        .rolling(28, min_periods=1)
        .mean()
        .reset_index(level=0, drop=True)
    )

    # same_weekday_mean_4w: average of same day of week over the last 4 weeks (t-7, t-14, t-21, t-28)
    lag_21 = df.groupby(group_col)["quantity_sold"].shift(21)
    lag_28 = df.groupby(group_col)["quantity_sold"].shift(28)
    same_weekday_df = pd.concat([df["lag_7"], df["lag_14"], lag_21, lag_28], axis=1)
    df["same_weekday_mean_4w"] = same_weekday_df.mean(axis=1)

    if dropna:
        # Require at least lag_1 and lag_7 (dropping early rows without minimal lag history)
        df = df.dropna(subset=["lag_1", "lag_7", "rolling_mean_7"]).reset_index(drop=True)
        # Fill remaining warm-up NaNs in lag_14, rolling_mean_28, same_weekday_mean_4w with rolling_mean_7
        df["lag_14"] = df["lag_14"].fillna(df["lag_7"])
        df["rolling_mean_28"] = df["rolling_mean_28"].fillna(df["rolling_mean_7"])
        df["same_weekday_mean_4w"] = df["same_weekday_mean_4w"].fillna(df["lag_7"])

    return df


def build_inference_features(
    history_sales: list[float],
    target_date: date,
    temperature: float,
    is_rainy: int,
    is_holiday: int,
) -> dict:
    """Build feature vector for live prediction of day t using historical sales up to day t-1."""
    n = len(history_sales)
    if n == 0:
        lag_1 = 0.0
        lag_7 = 0.0
        lag_14 = 0.0
        rolling_7 = 0.0
        rolling_28 = 0.0
        same_weekday_4w = 0.0
    else:
        lag_1 = float(history_sales[-1])
        lag_7 = float(history_sales[-7]) if n >= 7 else lag_1
        lag_14 = float(history_sales[-14]) if n >= 14 else lag_7

        recent_7 = history_sales[-7:]
        rolling_7 = float(sum(recent_7) / len(recent_7))

        recent_28 = history_sales[-28:]
        rolling_28 = float(sum(recent_28) / len(recent_28))

        # 4 weeks of same weekday: [-7, -14, -21, -28]
        sw_vals = []
        for offset in (7, 14, 21, 28):
            if n >= offset:
                sw_vals.append(float(history_sales[-offset]))
        same_weekday_4w = float(sum(sw_vals) / len(sw_vals)) if sw_vals else lag_7

    day_of_week = target_date.weekday()
    month = target_date.month
    weekend = 1 if day_of_week >= 5 else 0

    return {
        "day_of_week": day_of_week,
        "month": month,
        "weekend": weekend,
        "holiday": int(is_holiday),
        "temperature": float(temperature),
        "rain": int(is_rainy),
        "lag_1": lag_1,
        "lag_7": lag_7,
        "lag_14": lag_14,
        "rolling_mean_7": rolling_7,
        "rolling_mean_28": rolling_28,
        "same_weekday_mean_4w": same_weekday_4w,
    }


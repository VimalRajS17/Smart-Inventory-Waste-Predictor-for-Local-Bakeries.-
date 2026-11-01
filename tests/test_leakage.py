"""Test to verify absence of data leakage in time-series feature engineering."""

from datetime import date, timedelta
import pandas as pd
from ml.train_model import add_features


def test_no_feature_leakage_from_current_or_future_days():
    """Verify that features for day t do not change when sales on day t or later are modified."""
    base_date = date(2026, 1, 1)
    days = 20

    # Baseline series: constant or predictable sales
    records_1 = []
    for i in range(days):
        d_str = (base_date + timedelta(days=i)).isoformat()
        records_1.append({
            "item": "Croissant",
            "date": d_str,
            "quantity_sold": 20 + i,
            "temperature_c": 15.0,
            "is_rainy": 0,
            "is_holiday": 0,
        })
    df_raw_1 = pd.DataFrame(records_1)
    features_1 = add_features(df_raw_1)

    # Pick day t (e.g. index 12)
    target_day = (base_date + timedelta(days=12)).isoformat()
    row_t_1 = features_1[features_1["date"] == target_day].iloc[0]

    lag_1_val = row_t_1["lag_1"]
    lag_7_val = row_t_1["lag_7"]
    rolling_7_val = row_t_1["rolling_mean_7"]

    # Modified series: sales on day t and future days (t+1, t+2...) are drastically altered
    records_2 = []
    for i in range(days):
        d_str = (base_date + timedelta(days=i)).isoformat()
        if i >= 12:
            # Drastically change sales on day t and subsequent days
            qty = 9999 + i * 50
        else:
            qty = 20 + i  # identical history up to t-1

        records_2.append({
            "item": "Croissant",
            "date": d_str,
            "quantity_sold": qty,
            "temperature_c": 15.0,
            "is_rainy": 0,
            "is_holiday": 0,
        })
    df_raw_2 = pd.DataFrame(records_2)
    features_2 = add_features(df_raw_2)

    row_t_2 = features_2[features_2["date"] == target_day].iloc[0]

    # Features for day t must be completely unaffected by day t or future sales
    assert row_t_2["lag_1"] == lag_1_val, f"Leakage detected in lag_1: {row_t_2['lag_1']} != {lag_1_val}"
    assert row_t_2["lag_7"] == lag_7_val, f"Leakage detected in lag_7: {row_t_2['lag_7']} != {lag_7_val}"
    assert row_t_2["rolling_mean_7"] == rolling_7_val, f"Leakage detected in rolling_mean_7: {row_t_2['rolling_mean_7']} != {rolling_7_val}"


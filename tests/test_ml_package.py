"""Tests for ml/ package: prepare, features, train, evaluate, registry, CLI, and /api/forecast integration."""

from datetime import date, timedelta
import os
import numpy as np
import pandas as pd
import pytest

from ml.evaluate import compute_metrics, evaluate_item_rolling_origin
from ml.features import FEATURE_COLUMNS, build_features, build_inference_features
from ml.prepare import load_training_data
from ml.registry import get_active_model, promote_and_register_item_model
from ml.train import MODEL_TYPES, create_model, train_shop_models


# ---------------------------------------------------------------------------
# 1. Feature Engineering & Strict Leakage Proof
# ---------------------------------------------------------------------------

def test_feature_engineering_strict_zero_leakage():
    """Prove that features for day t use only data from strictly before day t (t-1 or earlier).
    
    Mutating sales on day t or any future day (t+1, t+2...) must NOT change any features for day t.
    """
    base_date = date(2026, 1, 1)
    num_days = 40

    records_orig = []
    for i in range(num_days):
        d_str = (base_date + timedelta(days=i)).isoformat()
        records_orig.append({
            "item_id": 1,
            "item": "Croissant",
            "date": d_str,
            "quantity_sold": 20.0 + (i % 7) * 2.0,
            "temperature": 15.0 + (i % 5),
            "rain": i % 3 == 0,
            "holiday": 0,
        })
    df_orig = pd.DataFrame(records_orig)
    feat_orig = build_features(df_orig, dropna=False)

    # Focus on day t (e.g. index 32)
    target_idx = 32
    target_date_str = (base_date + timedelta(days=target_idx)).isoformat()
    row_orig_t = feat_orig[feat_orig["date"] == target_date_str].iloc[0]

    # Captured features for day t before mutation
    orig_lags = {
        col: row_orig_t[col]
        for col in ["lag_1", "lag_7", "lag_14", "rolling_mean_7", "rolling_mean_28", "same_weekday_mean_4w"]
    }

    # Now create mutated dataset where sales on day t and ALL future days are radically altered
    records_mutated = []
    for i in range(num_days):
        d_str = (base_date + timedelta(days=i)).isoformat()
        if i >= target_idx:
            # Drastically alter sales from day t onwards
            qty = 99999.0 + i * 1000.0
        else:
            qty = 20.0 + (i % 7) * 2.0

        records_mutated.append({
            "item_id": 1,
            "item": "Croissant",
            "date": d_str,
            "quantity_sold": qty,
            "temperature": 15.0 + (i % 5),
            "rain": i % 3 == 0,
            "holiday": 0,
        })
    df_mutated = pd.DataFrame(records_mutated)
    feat_mutated = build_features(df_mutated, dropna=False)
    row_mutated_t = feat_mutated[feat_mutated["date"] == target_date_str].iloc[0]

    # Assert EVERY lag and rolling feature for day t remains 100% identical
    for col, orig_val in orig_lags.items():
        mutated_val = row_mutated_t[col]
        assert orig_val == mutated_val, (
            f"LEAKAGE DETECTED in '{col}' for day t: original={orig_val}, mutated={mutated_val}"
        )


def test_same_weekday_mean_4w_exact_value():
    """Verify that same_weekday_mean_4w is the exact mean of lags 7, 14, 21, and 28."""
    base_date = date(2026, 1, 1)
    records = []
    for i in range(35):
        d_str = (base_date + timedelta(days=i)).isoformat()
        # Sales equal to the day index
        records.append({
            "item_id": 1,
            "item": "Croissant",
            "date": d_str,
            "quantity_sold": float(i),
            "temperature": 20.0,
            "rain": 0,
            "holiday": 0,
        })
    df = pd.DataFrame(records)
    feat = build_features(df, dropna=False)

    # At day 30:
    # lag_7 = day 23 = 23
    # lag_14 = day 16 = 16
    # lag_21 = day 9 = 9
    # lag_28 = day 2 = 2
    # Expected same_weekday_mean_4w = (23 + 16 + 9 + 2) / 4.0 = 50 / 4.0 = 12.5
    row_30 = feat.iloc[30]
    assert row_30["lag_7"] == 23.0
    assert row_30["lag_14"] == 16.0
    assert row_30["same_weekday_mean_4w"] == 12.5


# ---------------------------------------------------------------------------
# 2. Candidate Models & Metric Calculations
# ---------------------------------------------------------------------------

def test_all_five_candidate_models_instantiate_and_predict():
    """Verify all 5 required models fit and return non-negative predictions."""
    X = pd.DataFrame({
        "day_of_week": [0, 1, 2, 3, 4, 5, 6] * 5,
        "month": [1] * 35,
        "weekend": [0, 0, 0, 0, 0, 1, 1] * 5,
        "holiday": [0] * 35,
        "temperature": [20.0] * 35,
        "rain": [0] * 35,
        "lag_1": [25.0] * 35,
        "lag_7": [24.0] * 35,
        "lag_14": [23.0] * 35,
        "rolling_mean_7": [25.0] * 35,
        "rolling_mean_28": [24.0] * 35,
        "same_weekday_mean_4w": [24.0] * 35,
    })
    y = np.array([25.0 + (i % 7) for i in range(35)], dtype=float)

    for m_type in MODEL_TYPES:
        model = create_model(m_type)
        model.fit(X, y)
        preds = model.predict(X[:5])
        assert len(preds) == 5
        assert all(p >= 0 for p in preds)


def test_metrics_calculation_wape():
    """Verify compute_metrics properly calculates MAE, RMSE, and WAPE."""
    y_true = np.array([100.0, 200.0, 300.0])
    y_pred = np.array([110.0, 190.0, 330.0])
    # abs_errors = [10, 10, 30] -> sum = 50
    # actual_sum = 600
    # WAPE = 50 / 600 = 0.0833 (8.33%)
    # MAE = 50 / 3 = 16.67
    m = compute_metrics(y_true, y_pred)
    assert m["mae"] == 16.67
    assert m["wape"] == 0.0833
    assert m["wape_pct"] == 8.33


# ---------------------------------------------------------------------------
# 3. Rolling-Origin Backtesting & Promotion Logic
# ---------------------------------------------------------------------------

def test_rolling_origin_four_folds():
    """Verify 4-fold rolling origin backtest on time series."""
    base_date = date(2024, 1, 1)
    records = []
    # 120 days of data
    for i in range(120):
        d_str = (base_date + timedelta(days=i)).isoformat()
        records.append({
            "item_id": 1,
            "item": "Croissant",
            "date": d_str,
            "quantity_sold": 30.0 + (i % 7) * 3.0,
            "temperature": 18.0,
            "rain": 0,
            "holiday": 0,
        })
    df_item = pd.DataFrame(records)

    eval_res = evaluate_item_rolling_origin(df_item, num_folds=4, fold_days=7)
    assert eval_res["cold_start"] is False
    assert len(eval_res["model_results"]) == len(MODEL_TYPES)

    # Verify each model has 4 fold metrics
    for m in MODEL_TYPES:
        fold_metrics = eval_res["model_results"][m]["fold_metrics"]
        assert len(fold_metrics) == 4
        for f in fold_metrics:
            assert f["mae"] >= 0.0
            assert f["rmse"] >= 0.0
            assert f["wape"] >= 0.0


def test_cold_start_under_60_days():
    """Items with fewer than 60 days of history always use moving average and flag cold_start."""
    base_date = date(2026, 1, 1)
    records = []
    # Only 40 days of history
    for i in range(40):
        d_str = (base_date + timedelta(days=i)).isoformat()
        records.append({
            "item_id": 2,
            "item": "NewItem",
            "date": d_str,
            "quantity_sold": 15.0,
            "temperature": 20.0,
            "rain": 0,
            "holiday": 0,
        })
    df_item = pd.DataFrame(records)

    eval_res = evaluate_item_rolling_origin(df_item, num_folds=4, fold_days=7)
    assert eval_res["cold_start"] is True
    assert eval_res["promoted_model"] == "moving_average"
    assert eval_res["beats_baseline"] is False


def test_promotion_and_model_registry(app, client):
    """Model is promoted, saved to models/<shop_id>/<item_id>/v<N>.joblib, and recorded in DB."""
    base_date = date(2024, 1, 1)
    records = []
    # 150 days of history with weekly patterns
    for i in range(150):
        d_str = (base_date + timedelta(days=i)).isoformat()
        t = 15.0 + (i % 10)
        r = int(i % 5 == 0)
        records.append({
            "item_id": 1,
            "item": "Croissant",
            "date": d_str,
            "quantity_sold": 30.0 + (i % 7) * 3.0 + t * 2.0 - r * 5.0,
            "temperature": t,
            "rain": r,
            "holiday": 0,
        })
    df_item = pd.DataFrame(records)

    with app.app_context():
        from app.db import get_db
        db = get_db()
        eval_res = evaluate_item_rolling_origin(df_item, num_folds=4, fold_days=7)
        reg = promote_and_register_item_model(shop_id=1, item_id=1, eval_result=eval_res, df_item=df_item, db=db)

        assert reg["version"] == 1
        assert reg["active"] == 1

        # Check database row
        row = db.execute("SELECT * FROM model_registry WHERE shop_id = 1 AND item_id = 1 AND active = 1").fetchone()
        assert row is not None
        assert row["model_type"] == reg["model_type"]

        if reg["path"]:
            # Verify file exists on disk
            assert os.path.exists(os.path.join(os.getcwd(), reg["path"]))
            assert "models/1/1/v1.joblib" in reg["path"]

        # Second training increments version to v2
        reg2 = promote_and_register_item_model(shop_id=1, item_id=1, eval_result=eval_res, df_item=df_item, db=db)
        assert reg2["version"] == 2
        assert reg2["active"] == 1

        # Old v1 is deactivated
        v1_row = db.execute("SELECT active FROM model_registry WHERE shop_id = 1 AND item_id = 1 AND version = 1").fetchone()
        assert v1_row["active"] == 0


# ---------------------------------------------------------------------------
# 4. CLI Command & /api/forecast Integration
# ---------------------------------------------------------------------------

def test_flask_train_cli_command(runner, app):
    """Running 'flask train --shop-id 1' executes pipeline, prints tables, and populates registry."""
    with app.app_context():
        from app.db import seed_demo
        seed_demo()

    result = runner.invoke(args=["train", "--shop-id", "1"])
    assert result.exit_code == 0
    # Verify tables and summary in CLI output
    assert "ROLLING-ORIGIN BACKTEST COMPARISON" in result.output
    assert "OVERALL SUMMARY ACROSS ALL ITEMS" in result.output
    assert "Promotion Summary for" in result.output

    # Verify rows in model_registry table
    with app.app_context():
        from app.db import get_db
        db = get_db()
        rows = db.execute("SELECT * FROM model_registry WHERE shop_id = 1 AND active = 1").fetchall()
        assert len(rows) > 0


def test_api_forecast_uses_active_registered_model(client, app, runner):
    """GET /api/forecast dynamically uses the active registered model version."""
    with app.app_context():
        from app.db import seed_demo
        seed_demo()

    # Run train CLI to populate model_registry with active models
    runner.invoke(args=["train", "--shop-id", "1"])

    # Query /api/forecast for tomorrow
    tomorrow = (date.today() + timedelta(days=1)).isoformat()
    resp = client.get(f"/api/forecast?date={tomorrow}")
    assert resp.status_code == 200
    data = resp.get_json()

    # Check that model_version reflects the registered active model
    forecast_items = data.get("forecasts", [])
    assert len(forecast_items) > 0
    for item_fc in forecast_items:
        assert item_fc["model_version"] is not None
        # e.g. "Random Forest v1" or "Hist Gradient Boosting v1" or "Moving Average v1"
        assert "v1" in item_fc["model_version"] or "Moving Average" in item_fc["model_version"]

    # Verify model_version is also recorded in forecasts table
    with app.app_context():
        from app.db import get_db
        db = get_db()
        fc_rows = db.execute("SELECT model_version FROM forecasts WHERE shop_id = 1 AND date = ?", (tomorrow,)).fetchall()
        assert len(fc_rows) > 0
        for r in fc_rows:
            assert r["model_version"] is not None


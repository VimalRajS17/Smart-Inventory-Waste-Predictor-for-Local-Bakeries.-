"""Loads trained models, constructs features from SQLite tables, and predicts item demand."""

from datetime import date, datetime, timedelta
import json
import math
import os
import holidays
import joblib
import numpy as np
import pandas as pd
from app.db import get_db
from ml.features import FEATURE_COLUMNS, build_inference_features
from ml.registry import get_active_model

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")

_model = None
_feature_cols = None
_model_version = None


def _load():
    """Load model artifact, feature column list, and model version."""
    global _model, _feature_cols, _model_version
    if _model is None:
        model_path = os.path.join(MODEL_DIR, "best_model.joblib")
        cols_path = os.path.join(MODEL_DIR, "feature_columns.joblib")
        metrics_path = os.path.join(MODEL_DIR, "metrics.json")

        if os.path.exists(model_path):
            _model = joblib.load(model_path)
        if os.path.exists(cols_path):
            _feature_cols = joblib.load(cols_path)
        if os.path.exists(metrics_path):
            try:
                with open(metrics_path, "r", encoding="utf-8") as f:
                    metrics_data = json.load(f)
                    _model_version = metrics_data.get("best_model", "Random Forest")
            except Exception:
                _model_version = "Random Forest"
        else:
            _model_version = "Random Forest"

    return _model, _feature_cols, _model_version


def get_seasonal_avg_temp(shop_id: int, item_id: int, month: int, db=None) -> float:
    """Compute the item's historical average temperature for a specific month, falling back if unavailable."""
    if db is None:
        db = get_db()
    month_str = f"{month:02d}"

    row = db.execute(
        """
        SELECT AVG(w.temp_mean_c) AS avg_temp
        FROM daily_log d
        JOIN weather_daily w ON d.shop_id = w.shop_id AND d.date = w.date
        WHERE d.shop_id = ? AND d.item_id = ? AND strftime('%m', d.date) = ?
          AND w.temp_mean_c IS NOT NULL
        """,
        (shop_id, item_id, month_str),
    ).fetchone()

    if row and row["avg_temp"] is not None:
        return round(float(row["avg_temp"]), 1)

    # Fallback across the shop for that month
    row_shop = db.execute(
        """
        SELECT AVG(temp_mean_c) AS avg_temp
        FROM weather_daily
        WHERE shop_id = ? AND strftime('%m', date) = ? AND temp_mean_c IS NOT NULL
        """,
        (shop_id, month_str),
    ).fetchone()

    if row_shop and row_shop["avg_temp"] is not None:
        return round(float(row_shop["avg_temp"]), 1)

    return 20.0


def calculate_safety_buffer(
    shop_id: int,
    item_id: int,
    target_date: date,
    predicted_qty: float,
    percentile: float = 80.0,
    db=None,
) -> float:
    """Compute the safety buffer as the N-th percentile (default 80th) of absolute backtest errors over the last 28 days."""
    if db is None:
        db = get_db()

    target_str = target_date.isoformat()
    start_28d = (target_date - timedelta(days=28)).isoformat()

    # 1. First attempt: check real recorded errors (forecasts vs daily_log) over last 28 days
    f_rows = db.execute(
        """
        SELECT abs(f.predicted_quantity - d.sold) AS abs_err
        FROM daily_log d
        JOIN forecasts f ON d.shop_id = f.shop_id AND d.item_id = f.item_id AND d.date = f.date
        WHERE d.shop_id = ? AND d.item_id = ? AND d.date < ? AND d.date >= ?
          AND d.sold IS NOT NULL AND f.predicted_quantity IS NOT NULL
        ORDER BY d.date ASC
        """,
        (shop_id, item_id, target_str, start_28d),
    ).fetchall()

    abs_errors = [float(r["abs_err"]) for r in f_rows if r["abs_err"] is not None]

    # 2. Second attempt: if fewer than 7 recorded forecast errors, evaluate backtest residuals from daily_log
    if len(abs_errors) < 7:
        log_rows = db.execute(
            """
            SELECT date, sold
            FROM daily_log
            WHERE shop_id = ? AND item_id = ? AND date < ? AND sold IS NOT NULL
            ORDER BY date ASC
            """,
            (shop_id, item_id, target_str),
        ).fetchall()

        if len(log_rows) >= 14:
            sales = [float(r["sold"]) for r in log_rows]
            calc_errors = []
            start_idx = max(7, len(sales) - 28)
            for i in range(start_idx, len(sales)):
                prior_7 = sales[i - 7 : i]
                ma7 = sum(prior_7) / 7.0
                calc_errors.append(abs(sales[i] - ma7))
            if len(calc_errors) >= 7:
                abs_errors = calc_errors

    if abs_errors:
        buffer_val = float(np.percentile(abs_errors, percentile))
        return max(1.0, round(buffer_val, 1))

    # Fallback to 15% of forecast if insufficient history exists
    return max(1.0, round(predicted_qty * 0.15, 1)) if predicted_qty > 0 else 1.0


def get_carry_over(
    shop_id: int,
    item_id: int,
    shelf_life_days: int,
    target_date: date,
    db=None,
) -> tuple[int, int]:
    """Retrieve carry-over: the last logged leftover if shelf_life_days > 1, otherwise 0.
    
    Returns (carry_over, last_leftover).
    """
    if db is None:
        db = get_db()

    target_str = target_date.isoformat()
    row = db.execute(
        """
        SELECT leftover, date
        FROM daily_log
        WHERE shop_id = ? AND item_id = ? AND date < ? AND leftover IS NOT NULL
        ORDER BY date DESC LIMIT 1
        """,
        (shop_id, item_id, target_str),
    ).fetchone()

    if row and row["leftover"] is not None:
        last_leftover = int(row["leftover"])
    else:
        last_leftover = 0

    carry_over = last_leftover if shelf_life_days > 1 else 0
    return carry_over, last_leftover


def check_overstock(last_leftover: int, carry_over: int, upper: float) -> tuple[bool, int, str | None]:
    """Check if leftover plus carry-over exceeds the upper forecast range."""
    total_leftover = last_leftover + carry_over
    if total_leftover > upper:
        overstock = True
        overstock_units = int(round(total_leftover - upper))
        warning = (
            f"Overstock warning: {total_leftover} leftover units exceeds upper forecast "
            f"({int(round(upper))}) by {overstock_units} units."
        )
        return overstock, overstock_units, warning
    return False, 0, None


def generate_what_if_reason(
    active_model,
    feature_cols: list[str],
    feat_dict: dict,
    history_rows: list,
    target_date: date,
    predicted_qty: float,
    cold_start: bool,
    shop_id: int,
    db=None,
) -> str:
    """Generate counterfactual explanation by replacing each feature with its recent average."""
    if cold_start:
        return "History is short: using 7-day average sales as baseline."

    day_name = target_date.strftime("%A")

    if active_model is None or not feature_cols or not hasattr(active_model, "predict"):
        if target_date.weekday() >= 5:
            return f"Higher than usual: weekend, strong last four {day_name}s."
        return f"Typical demand: standard {day_name} pattern."

    recent_28_rows = history_rows[-28:] if history_rows else []
    recent_sales = [float(r["sold"]) for r in recent_28_rows]
    mean_sales_28d = float(np.mean(recent_sales)) if recent_sales else predicted_qty

    feat_avgs = {
        "day_of_week": 3.0,
        "month": float(target_date.month),
        "is_weekend": 2.0 / 7.0,
        "is_holiday": 0.0,
        "temperature_c": float(feat_dict.get("temperature_c", 20.0)),
        "is_rainy": 0.15,
        "lag_1": mean_sales_28d,
        "lag_7": mean_sales_28d,
        "lag_14": mean_sales_28d,
        "rolling_mean_7": mean_sales_28d,
        "rolling_mean_28": mean_sales_28d,
        "same_weekday_mean_4w": mean_sales_28d,
    }

    base_X = pd.DataFrame([feat_dict])[feature_cols]
    base_pred = predicted_qty

    effects = []
    for col in feature_cols:
        if col.startswith("item_"):
            continue
        avg_val = feat_avgs.get(col, 0.0)
        X_cf = base_X.copy()
        X_cf[col] = avg_val
        try:
            pred_cf = float(active_model.predict(X_cf)[0])
            delta = base_pred - pred_cf
            effects.append((col, delta, abs(delta)))
        except Exception:
            continue

    effects.sort(key=lambda x: x[2], reverse=True)
    meaningful = [e for e in effects if e[2] >= 0.1]

    if not meaningful:
        if target_date.weekday() >= 5:
            return f"Higher than usual: weekend, steady recent sales."
        return f"Typical demand: normal {day_name} pattern."

    top_effects = meaningful[:2]

    def _phrase(col: str, delta: float) -> str:
        is_up = delta > 0
        if col in ("is_weekend", "day_of_week"):
            if target_date.weekday() >= 5:
                return "weekend" if is_up else "weekday pattern"
            return "weekday pattern" if is_up else "mid-week pattern"
        elif col == "same_weekday_mean_4w":
            return f"strong last four {day_name}s" if is_up else f"softer last four {day_name}s"
        elif col == "temperature_c":
            return "warm temperature" if is_up else "cold temperature"
        elif col == "is_rainy":
            return "dry weather" if is_up else "rain"
        elif col == "is_holiday":
            return "holiday" if is_up else "non-holiday"
        elif col == "rolling_mean_7":
            return "higher 7-day trend" if is_up else "lower 7-day trend"
        elif col == "rolling_mean_28":
            return "upward monthly trend" if is_up else "slower monthly trend"
        elif col == "lag_1":
            return "strong yesterday sales" if is_up else "slower yesterday sales"
        elif col == "lag_7":
            return f"strong sales last {day_name}" if is_up else f"lower sales last {day_name}"
        else:
            return col.replace("_", " ")

    phrases = []
    for col, delta, _ in top_effects:
        p = _phrase(col, delta)
        if p not in phrases:
            phrases.append(p)

    net_delta = sum(e[1] for e in top_effects)
    if net_delta > 0.5 or base_pred > mean_sales_28d + 1.0:
        headline = "Higher than usual"
    elif net_delta < -0.5 or base_pred < mean_sales_28d - 1.0:
        headline = "Lower than usual"
    else:
        headline = "Typical demand"

    return f"{headline}: {', '.join(phrases)}."


def generate_seven_day_outlook(
    active_reg,
    history_sales: list[float],
    target_date: date,
    base_predicted_qty: float,
    base_buffer: float,
    shop_id: int,
    item_id: int,
    db=None,
) -> list[dict]:
    """Generate recursive 7-day outlook with widening confidence intervals."""
    if db is None:
        db = get_db()
    outlook = []
    simulated_sales = list(history_sales)

    for h in range(7):
        curr_date = target_date + timedelta(days=h)
        curr_date_str = curr_date.isoformat()

        if h == 0:
            pred_h = base_predicted_qty
            buf_h = base_buffer
        else:
            w = db.execute(
                "SELECT temp_mean_c, temperature_c, is_rainy FROM weather_daily WHERE shop_id = ? AND date = ?",
                (shop_id, curr_date_str),
            ).fetchone()
            if w and (w["temp_mean_c"] is not None or w["temperature_c"] is not None):
                t_val = float(w["temp_mean_c"] if w["temp_mean_c"] is not None else w["temperature_c"])
                rain_val = int(w["is_rainy"] or 0)
            else:
                t_val = get_seasonal_avg_temp(shop_id, item_id, curr_date.month, db)
                rain_val = 0

            try:
                cal = holidays.country_holidays(os.getenv("HOLIDAY_COUNTRY", "UK"))
                hol_val = int(curr_date in cal)
            except Exception:
                hol_val = 0

            if active_reg and active_reg.get("model") and active_reg["model_type"] != "moving_average":
                feat_dict = build_inference_features(
                    history_sales=simulated_sales,
                    target_date=curr_date,
                    temperature=t_val,
                    is_rainy=rain_val,
                    is_holiday=hol_val,
                )
                cols = active_reg.get("feature_cols") or FEATURE_COLUMNS
                X = pd.DataFrame([feat_dict])[cols]
                raw_pred = active_reg["model"].predict(X)[0]
                pred_h = max(0.0, round(float(raw_pred), 1))
            else:
                recent = simulated_sales[-7:]
                pred_h = round(float(sum(recent) / len(recent)), 1) if recent else base_predicted_qty

            widening_factor = math.sqrt(1.0 + 0.25 * h)
            buf_h = max(1.0, round(base_buffer * widening_factor, 1))

        simulated_sales.append(pred_h)

        low_h = max(0.0, round(pred_h - buf_h, 1))
        up_h = round(pred_h + buf_h, 1)

        outlook.append({
            "date": curr_date_str,
            "day_name": curr_date.strftime("%a"),
            "forecast": pred_h,
            "lower": low_h,
            "upper": up_h,
            "buffer": buf_h,
            "confidence": "Normal" if h == 0 else "Lower confidence",
        })

    return outlook


def predict_item_demand(item_id_or_name: int | str, target_date_str: str, shop_id: int = 1) -> dict:
    """Predict demand for an item on target_date_str using daily_log and weather_daily.
    
    Upserts the forecast into the forecasts table and returns a comprehensive metadata dict
    including forecast, lower, upper, bake_qty, reason, overstock check, and 7-day outlook.
    """
    db = get_db()
    model, feature_cols, model_version = _load()

    # 1. Resolve item
    if isinstance(item_id_or_name, int) or (isinstance(item_id_or_name, str) and item_id_or_name.isdigit()):
        item_id = int(item_id_or_name)
        item = db.execute(
            "SELECT id, shop_id, name, current_stock, shelf_life_days FROM items WHERE id = ?",
            (item_id,),
        ).fetchone()
    else:
        item = db.execute(
            "SELECT id, shop_id, name, current_stock, shelf_life_days FROM items WHERE name = ? AND shop_id = ?",
            (str(item_id_or_name), shop_id),
        ).fetchone()
        if not item:
            item = db.execute(
                "SELECT id, shop_id, name, current_stock, shelf_life_days FROM items WHERE name = ?",
                (str(item_id_or_name),),
            ).fetchone()

    if not item:
        raise ValueError(f"Item '{item_id_or_name}' not found.")

    item_id = item["id"]
    item_name = item["name"]
    shop_id = item["shop_id"]
    current_stock = int(item["current_stock"]) if item["current_stock"] is not None else 0
    shelf_life_days = int(item["shelf_life_days"]) if item["shelf_life_days"] is not None else 1
    target_date = date.fromisoformat(target_date_str)

    # 2. Retrieve history from daily_log
    history_rows = db.execute(
        """
        SELECT date, sold
        FROM daily_log
        WHERE shop_id = ? AND item_id = ? AND sold IS NOT NULL
        ORDER BY date ASC
        """,
        (shop_id, item_id),
    ).fetchall()

    history_days = len(history_rows)
    if history_days == 0:
        last_logged_date = None
        lag_features_stale = True
    else:
        last_logged_str = history_rows[-1]["date"]
        last_logged_date = date.fromisoformat(last_logged_str)
        if target_date > date.today():
            lag_features_stale = (date.today() - last_logged_date).days > 1
        else:
            lag_features_stale = (target_date - last_logged_date).days > 1

    cold_start = history_days < 60

    # 3. Retrieve weather or apply seasonal fallback
    w_row = db.execute(
        """
        SELECT date, temp_mean_c, temperature_c, precip_mm, is_rainy, source
        FROM weather_daily
        WHERE shop_id = ? AND date = ?
        """,
        (shop_id, target_date_str),
    ).fetchone()

    if w_row and (w_row["temp_mean_c"] is not None or w_row["temperature_c"] is not None):
        temp_mean_c = float(w_row["temp_mean_c"] if w_row["temp_mean_c"] is not None else w_row["temperature_c"])
        precip_mm = float(w_row["precip_mm"] or 0.0)
        is_rainy = int(w_row["is_rainy"] or 0)
        weather_source = w_row["source"] or "forecast"
    else:
        temp_mean_c = get_seasonal_avg_temp(shop_id, item_id, target_date.month, db)
        precip_mm = 0.0
        is_rainy = 0
        weather_source = "fallback"

    # 4. Holiday calculation via holidays package
    country = os.getenv("HOLIDAY_COUNTRY", "UK")
    try:
        cal = holidays.country_holidays(country)
        is_holiday = int(target_date in cal)
    except Exception:
        is_holiday = 0

    # 5. Prediction calculation via model_registry or moving average
    active_reg = get_active_model(shop_id, item_id, db=db)
    recent_sales = [float(r["sold"]) for r in history_rows[-7:]]
    ma_val = round(float(sum(recent_sales) / len(recent_sales)), 1) if recent_sales else 0.0

    inference_feat_dict = {}
    active_model_obj = None
    active_cols = None

    if cold_start:
        predicted_qty = ma_val
        if active_reg and active_reg.get("version"):
            model_version = f"Moving Average v{active_reg['version']}"
        else:
            model_version = "7-day Moving Average"
    elif active_reg is not None and active_reg.get("model") is not None and active_reg["model_type"] != "moving_average":
        sales_floats = [float(r["sold"]) for r in history_rows]
        inference_feat_dict = build_inference_features(
            history_sales=sales_floats,
            target_date=target_date,
            temperature=temp_mean_c,
            is_rainy=is_rainy,
            is_holiday=is_holiday,
        )
        active_cols = active_reg.get("feature_cols") or FEATURE_COLUMNS
        X = pd.DataFrame([inference_feat_dict])[active_cols]
        raw_pred = active_reg["model"].predict(X)[0]
        predicted_qty = max(0.0, round(float(raw_pred), 1))
        model_version = f"{active_reg['model_type'].replace('_', ' ').title()} v{active_reg['version']}"
        active_model_obj = active_reg["model"]
    elif active_reg is not None and active_reg["model_type"] == "moving_average":
        predicted_qty = ma_val
        model_version = f"Moving Average v{active_reg['version']}"
    else:
        # Fallback to legacy model or moving average if registry is unseeded
        if model is not None and feature_cols is not None:
            day_of_week = target_date.weekday()
            is_weekend = int(day_of_week >= 5)
            month = target_date.month

            lag_1 = float(history_rows[-1]["sold"]) if history_rows else 0.0
            lag_7 = float(history_rows[-7]["sold"]) if history_days >= 7 else lag_1
            rolling_mean_7 = ma_val

            row_features = {col: 0 for col in feature_cols}
            row_features.update({
                "day_of_week": day_of_week,
                "is_weekend": is_weekend,
                "temperature_c": temp_mean_c,
                "is_rainy": is_rainy,
                "is_holiday": is_holiday,
                "month": month,
                "lag_1": lag_1,
                "lag_7": lag_7,
                "rolling_mean_7": rolling_mean_7,
            })
            item_col = f"item_{item_name}"
            if item_col in row_features:
                row_features[item_col] = 1

            inference_feat_dict = row_features
            active_cols = feature_cols
            active_model_obj = model

            X = pd.DataFrame([row_features])[feature_cols]
            raw_pred = model.predict(X)[0]
            predicted_qty = max(0.0, round(float(raw_pred), 1))
        else:
            predicted_qty = ma_val
            model_version = "7-day Moving Average"

    # 6. Upsert into forecasts table
    db.execute(
        """
        INSERT INTO forecasts (
            shop_id, item_id, date, predicted_quantity, actual_qty, weather_source,
            cold_start, lag_features_stale, model_version
        ) VALUES (
            ?, ?, ?, ?,
            (SELECT sold FROM daily_log WHERE shop_id = ? AND item_id = ? AND date = ?),
            ?, ?, ?, ?
        )
        ON CONFLICT(shop_id, item_id, date) DO UPDATE SET
            predicted_quantity = excluded.predicted_quantity,
            actual_qty = COALESCE(excluded.actual_qty, forecasts.actual_qty),
            weather_source = excluded.weather_source,
            cold_start = excluded.cold_start,
            lag_features_stale = excluded.lag_features_stale,
            model_version = excluded.model_version
        """,
        (
            shop_id,
            item_id,
            target_date_str,
            predicted_qty,
            shop_id,
            item_id,
            target_date_str,
            weather_source,
            int(cold_start),
            int(lag_features_stale),
            model_version,
        ),
    )
    db.commit()

    # 7. Safety Buffer (80th percentile of absolute backtest errors over last 28 days)
    buffer = calculate_safety_buffer(shop_id, item_id, target_date, predicted_qty, db=db)

    # 8. Carry-Over (last logged leftover if shelf_life_days > 1, otherwise 0)
    carry_over, last_leftover = get_carry_over(shop_id, item_id, shelf_life_days, target_date, db=db)

    # 9. Range and Bake Quantity
    # lower and upper
    lower = max(0.0, round(predicted_qty - buffer, 1))
    upper = round(predicted_qty + buffer, 1)

    # Bake quantity: forecast plus safety buffer minus carry-over, rounded up, never below 0
    bake_qty = max(0, int(math.ceil(predicted_qty + buffer - carry_over)))

    # 10. Overstock Flag: if leftover plus carry-over exceeds upper forecast
    overstock, overstock_units, overstock_warning = check_overstock(last_leftover, carry_over, upper)

    # 11. What-If Reason
    reason = generate_what_if_reason(
        active_model=active_model_obj,
        feature_cols=active_cols or [],
        feat_dict=inference_feat_dict,
        history_rows=history_rows,
        target_date=target_date,
        predicted_qty=predicted_qty,
        cold_start=cold_start,
        shop_id=shop_id,
        db=db,
    )

    # 12. Warnings
    warnings = []
    if cold_start:
        warnings.append("New item: Under 60 days of sales history recorded. Using 7-day rolling average.")
    if lag_features_stale:
        warnings.append("Recent sales not logged: The last logged sales entry is more than 1 day old.")
    if weather_source == "fallback":
        warnings.append("Weather offline: Live forecast unavailable; using seasonal temperature average.")
    if overstock and overstock_warning:
        warnings.append(overstock_warning)

    # 13. Recursive 7-Day Outlook
    sales_floats = [float(r["sold"]) for r in history_rows]
    outlook = generate_seven_day_outlook(
        active_reg=active_reg,
        history_sales=sales_floats,
        target_date=target_date,
        base_predicted_qty=predicted_qty,
        base_buffer=buffer,
        shop_id=shop_id,
        item_id=item_id,
        db=db,
    )

    # 14. Contextual Bandit Adjustment (Phase 4D - Controlled by RL_ENABLED)
    from ml.rl import is_rl_enabled, get_active_policy, discretize_state, MULTIPLIERS
    rl_active = False
    rl_multiplier = 1.0
    rl_notice = None
    rl_bake_qty = None

    if is_rl_enabled():
        bandit, policy_meta = get_active_policy(shop_id, item_id, db=db)
        if bandit and policy_meta:
            obs = policy_meta.get("observations_count", 0)
            beat_buf = bool(policy_meta.get("beat_buffer_rule", 0))
            if obs >= 30 and beat_buf:
                state_str = discretize_state(
                    day_of_week=target_date.weekday(),
                    is_holiday=is_holiday,
                    is_rainy=is_rainy,
                    temperature_c=temp_mean_c,
                    forecast_qty=predicted_qty,
                    rolling_mean_7=ma_val,
                    carry_over=carry_over,
                )
                act_idx = bandit.select_action(state_str, is_eval=True)
                rl_multiplier = MULTIPLIERS[act_idx]
                rl_active = True
                rl_bake_qty = max(0, int(math.ceil(rl_multiplier * predicted_qty - carry_over)))
                rl_notice = f"Reinforcement Learning Policy: applied {rl_multiplier:.1f}x multiplier ({obs} observations)."

    return {
        "item_id": item_id,
        "name": item_name,
        "current_stock": current_stock,
        "shelf_life_days": shelf_life_days,
        "forecast": predicted_qty,
        "predicted_qty": predicted_qty,
        "buffer": buffer,
        "carry_over": carry_over,
        "leftover": last_leftover,
        "lower": lower,
        "upper": upper,
        "range_min": lower,
        "range_max": upper,
        "range_str": f"{int(round(lower))}–{int(round(upper))}",
        "bake_qty": rl_bake_qty if rl_active else bake_qty,
        "suggested_bake_qty": rl_bake_qty if rl_active else bake_qty,
        "rl_active": rl_active,
        "rl_multiplier": rl_multiplier,
        "rl_notice": rl_notice,
        "rl_bake_qty": rl_bake_qty,
        "reason": reason,
        "overstock": overstock,
        "overstock_units": overstock_units,
        "overstock_warning": overstock_warning,
        "warnings": warnings,
        "weather": {
            "date": target_date_str,
            "temp_mean_c": temp_mean_c,
            "precip_mm": precip_mm,
            "is_rainy": is_rainy,
        },
        "weather_source": weather_source,
        "history_days": history_days,
        "cold_start": cold_start,
        "lag_features_stale": lag_features_stale,
        "model_version": model_version,
        "seven_day_outlook": outlook,
    }


def predict_all_items(target_date_str: str, shop_id: int = 1) -> list[dict]:
    """Generate and persist forecasts for all items in a given shop."""
    db = get_db()
    items = db.execute(
        "SELECT id, name FROM items WHERE shop_id = ? ORDER BY id ASC",
        (shop_id,),
    ).fetchall()

    results = []
    for item in items:
        res = predict_item_demand(item["id"], target_date_str, shop_id=shop_id)
        results.append(res)
    return results


def predict_demand(item: str, target_date: str, temperature_c: float = 25.0,
                   is_rainy: int = 0, is_holiday: int = 0) -> float:
    """Legacy backward-compatible wrapper returning only predicted quantity."""
    res = predict_item_demand(item, target_date)
    return res["predicted_qty"]

"""Rolling-origin backtesting, metric evaluation, and model comparison reporting."""

import numpy as np
import pandas as pd
from ml.features import FEATURE_COLUMNS, build_features
from ml.train import MODEL_TYPES, create_model


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Compute MAE, RMSE, and WAPE (sum of absolute errors / sum of actuals)."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.maximum(0.0, np.asarray(y_pred, dtype=float))
    n = len(y_true)
    if n == 0:
        return {"mae": 0.0, "rmse": 0.0, "wape": 0.0, "wape_pct": 0.0}

    abs_errors = np.abs(y_true - y_pred)
    mae = float(np.mean(abs_errors))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    actual_sum = float(np.sum(y_true))
    wape = float(np.sum(abs_errors) / actual_sum) if actual_sum > 0 else 0.0

    return {
        "mae": round(mae, 2),
        "rmse": round(rmse, 2),
        "wape": round(wape, 4),
        "wape_pct": round(wape * 100, 2),
    }


def evaluate_item_rolling_origin(
    df_item: pd.DataFrame,
    num_folds: int = 4,
    fold_days: int = 7,
) -> dict:
    """Execute rolling-origin backtest across at least num_folds on an item's time series.
    
    Returns evaluation dictionary containing per-fold metrics, aggregate metrics,
    and promotion decision.
    """
    if df_item is None or len(df_item) == 0:
        return {
            "item_name": "Item",
            "item_id": None,
            "raw_history_days": 0,
            "cold_start": True,
            "beats_baseline": False,
            "promoted_model": "moving_average",
            "model_results": {},
            "folds": [],
            "reason": "No historical sales data found. Using moving average fallback.",
        }

    df_feat = build_features(df_item, dropna=True)
    n_rows = len(df_feat)
    raw_history_days = len(df_item)
    cold_start = raw_history_days < 60

    item_name = df_item["item"].iloc[0] if (len(df_item) > 0 and "item" in df_item.columns) else "Item"
    item_id = int(df_item["item_id"].iloc[0]) if (len(df_item) > 0 and "item_id" in df_item.columns) else None

    # Handle cold start or insufficient data for 4-fold backtest
    total_test_horizon = num_folds * fold_days
    min_required_train = 30
    if cold_start or n_rows < (total_test_horizon + min_required_train):
        return {
            "item_name": item_name,
            "item_id": item_id,
            "raw_history_days": raw_history_days,
            "cold_start": True,
            "beats_baseline": False,
            "promoted_model": "moving_average",
            "model_results": {},
            "folds": [],
            "reason": f"Under 60 days of history ({raw_history_days} days). Using moving average fallback.",
        }

    # Build fold cutoffs
    # Fold k has test indices [start_idx, end_idx)
    fold_splits = []
    for k in range(num_folds):
        # Earliest fold is 0, latest fold is num_folds - 1
        days_from_end = (num_folds - k) * fold_days
        days_to_end = (num_folds - k - 1) * fold_days

        test_start_idx = n_rows - days_from_end
        test_end_idx = n_rows - days_to_end if days_to_end > 0 else n_rows
        train_end_idx = test_start_idx

        fold_splits.append((train_end_idx, test_start_idx, test_end_idx))

    # Evaluate each model across each fold
    results_by_model = {m: {"fold_metrics": [], "mean_mae": 0.0, "mean_rmse": 0.0, "mean_wape_pct": 0.0} for m in MODEL_TYPES}

    X_all = df_feat[FEATURE_COLUMNS]
    y_all = df_feat["quantity_sold"].to_numpy(dtype=float)

    for fold_idx, (train_end, test_start, test_end) in enumerate(fold_splits):
        X_train = X_all.iloc[:train_end]
        y_train = y_all[:train_end]
        X_test = X_all.iloc[test_start:test_end]
        y_test = y_all[test_start:test_end]

        for m_name in MODEL_TYPES:
            model = create_model(m_name)
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            metrics = compute_metrics(y_test, preds)
            results_by_model[m_name]["fold_metrics"].append(metrics)

    # Compute mean across folds
    for m_name, res in results_by_model.items():
        res["mean_mae"] = round(float(np.mean([f["mae"] for f in res["fold_metrics"]])), 2)
        res["mean_rmse"] = round(float(np.mean([f["rmse"] for f in res["fold_metrics"]])), 2)
        res["mean_wape_pct"] = round(float(np.mean([f["wape_pct"] for f in res["fold_metrics"]])), 2)
        res["mean_wape_raw"] = float(np.mean([f["wape"] for f in res["fold_metrics"]]))

    # Promotion Rule:
    # A model is promoted only if it beats the naive baseline in the latest fold (Fold 4).
    # Otherwise use the moving average.
    latest_fold_idx = num_folds - 1
    naive_latest_wape = results_by_model["naive"]["fold_metrics"][latest_fold_idx]["wape"]

    candidate_ml_models = ["random_forest", "hist_gradient_boosting", "linear_regression"]
    eligible_models = []

    for m_name in candidate_ml_models:
        latest_wape = results_by_model[m_name]["fold_metrics"][latest_fold_idx]["wape"]
        beats_naive_latest = latest_wape < naive_latest_wape
        results_by_model[m_name]["beats_naive_latest"] = beats_naive_latest
        if beats_naive_latest:
            eligible_models.append(m_name)

    results_by_model["naive"]["beats_naive_latest"] = False
    results_by_model["moving_average"]["beats_naive_latest"] = (
        results_by_model["moving_average"]["fold_metrics"][latest_fold_idx]["wape"] < naive_latest_wape
    )

    if eligible_models:
        # Pick model with lowest latest fold WAPE (or mean WAPE)
        best_ml = min(eligible_models, key=lambda m: (
            results_by_model[m]["fold_metrics"][latest_fold_idx]["wape"],
            results_by_model[m]["mean_wape_raw"]
        ))
        promoted_model = best_ml
        beats_baseline = True
    else:
        promoted_model = "moving_average"
        beats_baseline = False

    return {
        "item_name": item_name,
        "item_id": item_id,
        "raw_history_days": raw_history_days,
        "cold_start": False,
        "beats_baseline": beats_baseline,
        "promoted_model": promoted_model,
        "model_results": results_by_model,
        "num_folds": num_folds,
        "fold_days": fold_days,
    }


def print_comparison_tables(evaluations: list[dict]):
    """Print formatted per-item and overall backtest comparison tables to stdout."""
    print("\n" + "=" * 96)
    print("LOAF LEDGER ML PIPELINE: ROLLING-ORIGIN BACKTEST COMPARISON (4 FOLDS)")
    print("=" * 96)

    overall_metrics = {m: {"mae": [], "rmse": [], "wape_pct": []} for m in MODEL_TYPES}

    for ev in evaluations:
        item_name = ev.get("item_name", "Unknown Item")
        print(f"\nItem: {item_name.upper()} ({ev.get('raw_history_days', 0)} days of history)")
        print("-" * 96)

        if ev.get("cold_start"):
            print(f"  [Cold Start] {ev.get('reason')}")
            print(f"  Promoted Model: 7-day Moving Average (Active = 1, Beats Baseline = 0)")
            continue

        header = f"{'Model':<24} | {'Fold 1':<8} | {'Fold 2':<8} | {'Fold 3':<8} | {'Fold 4':<8} | {'MAE':<6} | {'RMSE':<6} | {'Mean WAPE':<10} | {'Latest Win?':<11}"
        print(header)
        print("-" * 96)

        results = ev["model_results"]
        for m_name in MODEL_TYPES:
            r = results[m_name]
            f_wapes = [f"{f['wape_pct']:.1f}%" for f in r["fold_metrics"]]
            f1, f2, f3, f4 = (f_wapes + ["-", "-", "-", "-"])[:4]

            beats_str = "YES" if r.get("beats_naive_latest") else "NO"
            if m_name == "naive":
                beats_str = "BASELINE"

            is_promoted = " *" if m_name == ev["promoted_model"] else ""
            display_name = f"{m_name.replace('_', ' ').title()}{is_promoted}"

            row_str = f"{display_name:<24} | {f1:<8} | {f2:<8} | {f3:<8} | {f4:<8} | {r['mean_mae']:<6.2f} | {r['mean_rmse']:<6.2f} | {r['mean_wape_pct']:>6.1f}%   | {beats_str:<11}"
            print(row_str)

            overall_metrics[m_name]["mae"].append(r["mean_mae"])
            overall_metrics[m_name]["rmse"].append(r["mean_rmse"])
            overall_metrics[m_name]["wape_pct"].append(r["mean_wape_pct"])

        print(f"  --> Promoted: {ev['promoted_model'].replace('_', ' ').title()} (Beats Naive Baseline: {ev['beats_baseline']})")

    # Overall Summary Table
    print("\n" + "=" * 96)
    print("OVERALL SUMMARY ACROSS ALL ITEMS")
    print("=" * 96)
    print(f"{'Model':<28} | {'Average MAE':<14} | {'Average RMSE':<14} | {'Average WAPE':<14}")
    print("-" * 96)

    for m_name in MODEL_TYPES:
        maes = overall_metrics[m_name]["mae"]
        rmses = overall_metrics[m_name]["rmse"]
        wapes = overall_metrics[m_name]["wape_pct"]
        if maes:
            avg_mae = np.mean(maes)
            avg_rmse = np.mean(rmses)
            avg_wape = np.mean(wapes)
            print(f"{m_name.replace('_', ' ').title():<28} | {avg_mae:<14.2f} | {avg_rmse:<14.2f} | {avg_wape:>7.1f}%")
        else:
            print(f"{m_name.replace('_', ' ').title():<28} | {'N/A':<14} | {'N/A':<14} | {'N/A':<14}")
    print("=" * 96 + "\n")


"""Model definitions and estimators for bakery demand forecasting."""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression


class NaiveBaseline(BaseEstimator, RegressorMixin):
    """Naive baseline: predicts today's demand as the same weekday last week (lag_7)."""

    def fit(self, X, y=None):
        return self

    def predict(self, X):
        if isinstance(X, pd.DataFrame):
            if "lag_7" in X.columns:
                return X["lag_7"].to_numpy(dtype=float)
            return X.iloc[:, 0].to_numpy(dtype=float)
        # Assuming lag_7 is at index 7 if array
        return np.asarray(X)[:, 7].astype(float)


class MovingAverageBaseline(BaseEstimator, RegressorMixin):
    """Moving average baseline: predicts today's demand as the 7-day rolling mean."""

    def fit(self, X, y=None):
        return self

    def predict(self, X):
        if isinstance(X, pd.DataFrame):
            if "rolling_mean_7" in X.columns:
                return X["rolling_mean_7"].to_numpy(dtype=float)
            return X.iloc[:, 0].to_numpy(dtype=float)
        # Assuming rolling_mean_7 is at index 9 if array
        return np.asarray(X)[:, 9].astype(float)


MODEL_TYPES = [
    "naive",
    "moving_average",
    "linear_regression",
    "random_forest",
    "hist_gradient_boosting",
]


def create_model(model_type: str):
    """Create a fresh instance of the specified model type."""
    t = model_type.lower()
    if t == "naive":
        return NaiveBaseline()
    elif t == "moving_average":
        return MovingAverageBaseline()
    elif t == "linear_regression":
        return LinearRegression()
    elif t == "random_forest":
        return RandomForestRegressor(n_estimators=100, random_state=42, min_samples_leaf=2)
    elif t == "hist_gradient_boosting":
        return HistGradientBoostingRegressor(random_state=42, max_iter=100)
    else:
        raise ValueError(f"Unknown model type '{model_type}'. Supported: {MODEL_TYPES}")


def get_candidate_models() -> dict:
    """Return dictionary of freshly initialized candidate models."""
    return {name: create_model(name) for name in MODEL_TYPES}


def train_shop_models(shop_id: int, db=None) -> list[dict]:
    """Train candidate models, run rolling-origin 4-fold backtest, promote winners, and update registry."""
    from app.db import get_db
    from ml.prepare import load_training_data
    from ml.evaluate import evaluate_item_rolling_origin, print_comparison_tables
    from ml.registry import promote_and_register_item_model

    if db is None:
        db = get_db()

    items = db.execute("SELECT id, name FROM items WHERE shop_id = ? ORDER BY id ASC", (shop_id,)).fetchall()
    if not items:
        return []

    evaluations = []
    for it in items:
        item_id = it["id"]
        df_item = load_training_data(shop_id, item_id=item_id, db=db)
        eval_res = evaluate_item_rolling_origin(df_item, num_folds=4, fold_days=7)
        if eval_res.get("item_name") == "Item":
            eval_res["item_name"] = it["name"]
        eval_res["item_id"] = item_id
        reg_record = promote_and_register_item_model(shop_id, item_id, eval_res, df_item, db=db)
        eval_res["registry_record"] = reg_record
        evaluations.append(eval_res)

    print_comparison_tables(evaluations)
    return evaluations


def count_new_logs_since_training(shop_id: int, db) -> tuple[int, str | None]:
    """Count how many daily_log rows with sold quantity exist since the shop's last training.
    
    Returns (new_log_count, last_trained_iso_string).
    """
    last_row = db.execute(
        "SELECT MAX(trained_at) AS last_trained FROM model_registry WHERE shop_id = ?",
        (shop_id,),
    ).fetchone()
    last_trained = last_row["last_trained"] if last_row and last_row["last_trained"] else None

    if last_trained is None:
        # Shop has never been trained
        row = db.execute(
            "SELECT COUNT(*) AS c FROM daily_log WHERE shop_id = ? AND sold IS NOT NULL",
            (shop_id,),
        ).fetchone()
        return (row["c"] if row else 0), None

    last_clean = last_trained.replace("T", " ")[:19]
    # Count rows added or updated after last_trained timestamp or with dates after last_trained date
    row = db.execute(
        """
        SELECT COUNT(*) AS c
        FROM daily_log
        WHERE shop_id = ? AND sold IS NOT NULL
          AND (
            datetime(replace(created_at, 'T', ' ')) >= datetime(?)
            OR date > substr(?, 1, 10)
          )
        """,
        (shop_id, last_clean, last_clean),
    ).fetchone()
    count = row["c"] if row else 0
    return count, last_trained


def retrain_all_eligible_shops(db=None, force: bool = False, min_new_rows: int = 30) -> list[dict]:
    """Train every shop with at least min_new_rows new log rows since its last training."""
    from app.db import get_db

    if db is None:
        db = get_db()

    shops = db.execute("SELECT id, name FROM shops ORDER BY id ASC").fetchall()
    results = []

    for s in shops:
        shop_id = s["id"]
        shop_name = s["name"]

        new_count, last_trained = count_new_logs_since_training(shop_id, db)
        is_eligible = force or (new_count >= min_new_rows)

        shop_summary = {
            "shop_id": shop_id,
            "shop_name": shop_name,
            "new_logs_count": new_count,
            "last_trained": last_trained,
            "retrained": False,
            "evaluations": [],
        }

        if is_eligible:
            evaluations = train_shop_models(shop_id, db=db)
            shop_summary["retrained"] = True
            shop_summary["evaluations"] = evaluations

        results.append(shop_summary)

    return results


import click
from flask.cli import with_appcontext


@click.command("train")
@click.option("--shop-id", default=None, type=int, help="Shop ID to train models for (defaults to demo shop).")
@with_appcontext
def train_command(shop_id: int | None):
    """Train candidate models, run rolling-origin backtest, and update model registry."""
    from app.db import get_db
    db = get_db()

    if shop_id is None:
        shop = db.execute("SELECT id, name FROM shops ORDER BY id ASC LIMIT 1").fetchone()
        if not shop:
            click.echo("No shops found in database. Please run 'seed-demo' or complete onboarding first.")
            return
        target_shop_id = shop["id"]
        shop_name = shop["name"]
    else:
        shop = db.execute("SELECT id, name FROM shops WHERE id = ?", (shop_id,)).fetchone()
        if not shop:
            click.echo(f"Shop with ID {shop_id} not found.")
            return
        target_shop_id = shop["id"]
        shop_name = shop["name"]

    click.echo(f"\n========================================================")
    click.echo(f"Training models for '{shop_name}' (Shop ID: {target_shop_id})...")
    click.echo(f"========================================================")

    evaluations = train_shop_models(target_shop_id, db=db)
    if not evaluations:
        click.echo(f"No items or daily sales found to train for shop '{shop_name}'.")
        return

    click.echo(f"Promotion Summary for '{shop_name}':")
    for ev in evaluations:
        it_name = ev.get("item_name")
        rec = ev.get("registry_record", {})
        m_type = rec.get("model_type", "").replace("_", " ").title()
        ver = rec.get("version", 1)
        path = rec.get("path") or "[In-Memory / Moving Average]"
        beats = "Yes (Promoted)" if rec.get("beats_baseline") else "No (Kept Previous Active)"
        if ev.get("cold_start"):
            beats = "Cold start (< 60d)"
        click.echo(f"  • {it_name}: v{ver} ({m_type}) -> {path} [Active: {rec.get('active', 0)}, Status: {beats}]")
    click.echo("\nModel training and registry update complete.")


@click.command("retrain-all")
@click.option("--force", is_flag=True, default=False, help="Force retrain all shops regardless of new log rows threshold.")
@click.option("--min-new-rows", default=30, type=int, help="Minimum new rows required to trigger retraining (default: 30).")
@with_appcontext
def retrain_all_command(force: bool, min_new_rows: int):
    """Train every bakery shop with at least 30 new log rows since its last training."""
    from app.db import get_db
    db = get_db()

    click.echo("\n========================================================")
    click.echo(f"Starting batch retraining scan (threshold: {min_new_rows} new rows, force={force})...")
    click.echo("========================================================")

    results = retrain_all_eligible_shops(db=db, force=force, min_new_rows=min_new_rows)
    retrained_count = 0

    for r in results:
        s_name = r["shop_name"]
        s_id = r["shop_id"]
        c = r["new_logs_count"]
        if r["retrained"]:
            retrained_count += 1
            click.echo(f"\n✓ Shop '{s_name}' (ID: {s_id}): Retrained ({c} new log rows).")
            for ev in r["evaluations"]:
                rec = ev.get("registry_record", {})
                beats = "Promoted" if rec.get("beats_baseline") else "Kept Previous Active"
                click.echo(f"   • {ev.get('item_name')}: v{rec.get('version')} ({rec.get('model_type')}) [Status: {beats}]")
        else:
            click.echo(f"• Shop '{s_name}' (ID: {s_id}): Skipped ({c} new log rows < {min_new_rows} required).")

    click.echo(f"\nBatch retraining complete: {retrained_count} of {len(results)} shops retrained.\n")


@click.command("rollback")
@click.option("--shop-id", required=True, type=int, help="Shop ID")
@click.option("--item-id", required=True, type=int, help="Item ID")
@click.option("--version", default=None, type=int, help="Target version to roll back to (defaults to previous version).")
@with_appcontext
def rollback_command(shop_id: int, item_id: int, version: int | None):
    """Roll back the active model for an item to a previous version."""
    from app.db import get_db
    from ml.registry import rollback_item_model

    db = get_db()
    try:
        res = rollback_item_model(shop_id=shop_id, item_id=item_id, target_version=version, db=db)
        click.echo(
            f"Successfully rolled back shop {shop_id}, item {item_id}: "
            f"v{res['previous_version']} deactivated -> v{res['active_version']} ({res['model_type']}) is now active."
        )
    except Exception as exc:
        click.echo(f"Rollback failed: {exc}", err=True)


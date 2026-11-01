"""Model registry, artifact versioning, and persistence for Loaf Ledger."""

from datetime import datetime, timezone
import os
import joblib
import pandas as pd
from app.db import get_db
from ml.features import FEATURE_COLUMNS, build_features
from ml.train import create_model

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")


def promote_and_register_item_model(
    shop_id: int,
    item_id: int,
    eval_result: dict,
    df_item: pd.DataFrame,
    db=None,
) -> dict:
    """Promote the winning model, train it on full history, save artifact, and register in database."""
    if db is None:
        db = get_db()

    promoted_type = eval_result["promoted_model"]
    beats_baseline = 1 if eval_result["beats_baseline"] else 0
    cold_start = 1 if eval_result["cold_start"] else 0

    # Determine next version number for (shop_id, item_id)
    v_row = db.execute(
        "SELECT MAX(version) AS max_v FROM model_registry WHERE shop_id = ? AND item_id = ?",
        (shop_id, item_id),
    ).fetchone()
    next_version = (v_row["max_v"] or 0) + 1 if v_row else 1

    model_rel_path = None
    mae = None
    rmse = None
    wape = None

    if promoted_type in ("random_forest", "hist_gradient_boosting", "linear_regression") and not cold_start:
        # 1. Fit winning ML model on all available history
        df_feat = build_features(df_item, dropna=True)
        X_all = df_feat[FEATURE_COLUMNS]
        y_all = df_feat["quantity_sold"].to_numpy(dtype=float)

        fitted_model = create_model(promoted_type)
        fitted_model.fit(X_all, y_all)

        # 2. Save artifact to models/<shop_id>/<item_id>/v<N>.joblib
        item_dir = os.path.join(MODELS_DIR, str(shop_id), str(item_id))
        os.makedirs(item_dir, exist_ok=True)
        filename = f"v{next_version}.joblib"
        artifact_abs_path = os.path.join(item_dir, filename)
        model_rel_path = os.path.relpath(artifact_abs_path, PROJECT_ROOT).replace("\\", "/")

        m_res = eval_result["model_results"].get(promoted_type, {})
        mae = m_res.get("mean_mae")
        rmse = m_res.get("mean_rmse")
        wape = m_res.get("mean_wape_raw")

        payload = {
            "model": fitted_model,
            "feature_cols": FEATURE_COLUMNS,
            "model_type": promoted_type,
            "version": next_version,
            "shop_id": shop_id,
            "item_id": item_id,
            "metrics": {"mae": mae, "rmse": rmse, "wape": wape},
            "trained_at": datetime.now(timezone.utc).isoformat(),
        }
        joblib.dump(payload, artifact_abs_path)
    else:
        # Moving average fallback or cold start
        promoted_type = "moving_average"
        m_res = eval_result.get("model_results", {}).get("moving_average", {})
        mae = m_res.get("mean_mae")
        rmse = m_res.get("mean_rmse")
        wape = m_res.get("mean_wape_raw")

    # 3. Check existing active model for promotion safety guard
    existing_active = db.execute(
        "SELECT id, version FROM model_registry WHERE shop_id = ? AND item_id = ? AND active = 1",
        (shop_id, item_id),
    ).fetchone()

    # Promotion Rule:
    # A new model is activated only if it beats the baseline.
    # If it fails to beat the baseline, the existing active model remains active!
    if beats_baseline:
        should_activate = 1
    else:
        # Fallback to activating only if no model was previously active for this item
        should_activate = 1 if existing_active is None else 0

    if should_activate:
        db.execute(
            "UPDATE model_registry SET active = 0 WHERE shop_id = ? AND item_id = ?",
            (shop_id, item_id),
        )

    # 4. Insert new registry record
    cur = db.execute(
        """
        INSERT INTO model_registry (
            shop_id, item_id, model_type, version, path,
            mae, rmse, wape, beats_baseline, active
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            shop_id,
            item_id,
            promoted_type,
            next_version,
            model_rel_path,
            mae,
            rmse,
            wape,
            beats_baseline,
            should_activate,
        ),
    )
    db.commit()

    return {
        "id": cur.lastrowid,
        "shop_id": shop_id,
        "item_id": item_id,
        "model_type": promoted_type,
        "version": next_version,
        "path": model_rel_path,
        "mae": mae,
        "rmse": rmse,
        "wape": wape,
        "beats_baseline": beats_baseline,
        "active": should_activate,
    }


def rollback_item_model(
    shop_id: int,
    item_id: int,
    target_version: int | None = None,
    db=None,
) -> dict:
    """Roll back the active model for an item to a previous version."""
    if db is None:
        db = get_db()

    # Find currently active model
    current_active = db.execute(
        """
        SELECT id, version, model_type, path
        FROM model_registry
        WHERE shop_id = ? AND item_id = ? AND active = 1
        ORDER BY version DESC
        LIMIT 1
        """,
        (shop_id, item_id),
    ).fetchone()

    if not current_active:
        raise ValueError(f"No active model found to roll back for shop {shop_id}, item {item_id}.")

    current_version = current_active["version"]

    if target_version is not None:
        target_row = db.execute(
            """
            SELECT id, version, model_type, path
            FROM model_registry
            WHERE shop_id = ? AND item_id = ? AND version = ?
            """,
            (shop_id, item_id, target_version),
        ).fetchone()
        if not target_row:
            raise ValueError(f"Version v{target_version} not found for shop {shop_id}, item {item_id}.")
    else:
        # Find highest previous version
        target_row = db.execute(
            """
            SELECT id, version, model_type, path
            FROM model_registry
            WHERE shop_id = ? AND item_id = ? AND version < ?
            ORDER BY version DESC
            LIMIT 1
            """,
            (shop_id, item_id, current_version),
        ).fetchone()
        if not target_row:
            raise ValueError(f"No previous model version found to roll back to for shop {shop_id}, item {item_id}.")

    # Deactivate current active model
    db.execute(
        "UPDATE model_registry SET active = 0 WHERE shop_id = ? AND item_id = ?",
        (shop_id, item_id),
    )
    # Activate target version
    db.execute(
        "UPDATE model_registry SET active = 1 WHERE id = ?",
        (target_row["id"],),
    )
    db.commit()

    return {
        "shop_id": shop_id,
        "item_id": item_id,
        "previous_version": current_version,
        "active_version": target_row["version"],
        "model_type": target_row["model_type"],
        "path": target_row["path"],
    }


def get_active_model(shop_id: int, item_id: int, db=None) -> dict | None:
    """Retrieve active model information and load serialized estimator from disk if applicable."""
    if db is None:
        db = get_db()

    row = db.execute(
        """
        SELECT id, shop_id, item_id, model_type, version, path, mae, rmse, wape, beats_baseline, active
        FROM model_registry
        WHERE shop_id = ? AND item_id = ? AND active = 1
        ORDER BY version DESC
        LIMIT 1
        """,
        (shop_id, item_id),
    ).fetchone()

    if not row:
        return None

    record = dict(row)
    model_obj = None
    feature_cols = FEATURE_COLUMNS

    if record.get("path"):
        full_path = os.path.join(PROJECT_ROOT, record["path"])
        if os.path.exists(full_path):
            try:
                artifact = joblib.load(full_path)
                model_obj = artifact.get("model")
                feature_cols = artifact.get("feature_cols", FEATURE_COLUMNS)
            except Exception:
                model_obj = None

    record["model"] = model_obj
    record["feature_cols"] = feature_cols
    return record


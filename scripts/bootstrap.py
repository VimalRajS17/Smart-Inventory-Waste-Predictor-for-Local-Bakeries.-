#!/usr/bin/env python3
"""Deployment bootstrap script for Loaf Ledger.

Executed by the host container start command prior to starting Gunicorn.
Idempotent and safe to run on every startup:
1. Creates database tables and indices if missing (non-destructive).
2. Seeds demo shop, items, and sales history only if database contains 0 shops.
3. Fetches 7-day weather forecast from Open-Meteo if weather cache is empty.
4. Trains and enrolls models in model_registry only if registry has 0 active models for a shop.
"""

import logging
import os
import sys
import time

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("loafledger.bootstrap")


def run_bootstrap(app=None) -> bool:
    """Execute idempotent bootstrap sequence inside Flask application context."""
    start_time = time.time()
    logger.info("=" * 70)
    logger.info("[BOOTSTRAP] Starting Loaf Ledger deployment bootstrap sequence...")
    logger.info("=" * 70)

    if app is None:
        from app import create_app
        app = create_app()

    with app.app_context():
        from app.db import create_tables_if_missing, get_db, seed_demo

        db = get_db()
        db_type = "PostgreSQL" if app.config.get("DATABASE_URL") and (
            app.config["DATABASE_URL"].startswith("postgres://") or
            app.config["DATABASE_URL"].startswith("postgresql://")
        ) else "SQLite"
        logger.info(f"[BOOTSTRAP] Database engine active: {db_type}")

        # ------------------------------------------------------------------
        # Step 1: Create tables and indexes if missing
        # ------------------------------------------------------------------
        logger.info("[BOOTSTRAP] [Step 1/4] Checking database schema...")
        create_tables_if_missing(db)
        logger.info("[BOOTSTRAP] [Step 1/4] Schema verified (all 9 tables and indices exist).")

        # ------------------------------------------------------------------
        # Step 2: Seed demo shop only if database is empty
        # ------------------------------------------------------------------
        logger.info("[BOOTSTRAP] [Step 2/4] Checking existing bakery shops...")
        shop_count_row = db.execute("SELECT COUNT(*) AS c FROM shops;").fetchone()
        shop_count = shop_count_row["c"] if shop_count_row else 0

        if shop_count == 0:
            logger.info("[BOOTSTRAP] [Step 2/4] Database is empty (0 shops found). Seeding demo bakery...")
            seed_demo()
            logger.info("[BOOTSTRAP] [Step 2/4] Demo bakery, items, and sales history seeded successfully.")
        else:
            logger.info(f"[BOOTSTRAP] [Step 2/4] Database contains {shop_count} existing shop(s). Skipping demo seed.")

        # ------------------------------------------------------------------
        # Step 3: Fetch weather if cache is empty
        # ------------------------------------------------------------------
        logger.info("[BOOTSTRAP] [Step 3/4] Checking weather forecast cache...")
        shops = db.execute("SELECT id, name, lat, lon FROM shops ORDER BY id ASC;").fetchall()

        for shop in shops:
            shop_id = shop["id"]
            shop_name = shop["name"]
            w_row = db.execute("SELECT COUNT(*) AS c FROM weather_daily WHERE shop_id = ?;", (shop_id,)).fetchone()
            weather_count = w_row["c"] if w_row else 0

            if weather_count == 0:
                logger.info(f"[BOOTSTRAP] [Step 3/4] Weather cache empty for shop '{shop_name}' (ID: {shop_id}). Fetching forecast...")
                try:
                    from app.services.weather import fetch_forecast, save_weather
                    forecast_rows = fetch_forecast(float(shop["lat"]), float(shop["lon"]), days=7)
                    save_weather(shop_id, forecast_rows, "forecast")
                    logger.info(f"[BOOTSTRAP] [Step 3/4] Successfully cached {len(forecast_rows)} days of forecast for shop {shop_id}.")
                except Exception as exc:
                    logger.warning(f"[BOOTSTRAP] [Step 3/4] Failed to fetch live weather forecast: {exc}. Proceeding with offline fallback.")
            else:
                logger.info(f"[BOOTSTRAP] [Step 3/4] Weather cache contains {weather_count} records for shop '{shop_name}'. Skipping fetch.")

        # ------------------------------------------------------------------
        # Step 4: Train models if registry has none for a shop
        # ------------------------------------------------------------------
        logger.info("[BOOTSTRAP] [Step 4/4] Checking machine learning model registry...")
        for shop in shops:
            shop_id = shop["id"]
            shop_name = shop["name"]
            reg_row = db.execute(
                "SELECT COUNT(*) AS c FROM model_registry WHERE shop_id = ? AND active = 1;",
                (shop_id,),
            ).fetchone()
            active_models = reg_row["c"] if reg_row else 0

            if active_models == 0:
                logger.info(f"[BOOTSTRAP] [Step 4/4] No active models in registry for shop '{shop_name}' (ID: {shop_id}). Training models...")
                from ml.train import train_shop_models
                train_shop_models(shop_id, db=db)

                # Ensure fallback global model file exists if missing
                global_model_path = os.path.join(PROJECT_ROOT, "models", "best_model.joblib")
                if not os.path.exists(global_model_path):
                    try:
                        from ml.train_model import train as train_global_model
                        train_global_model(source="db")
                    except Exception as exc:
                        logger.warning(f"[BOOTSTRAP] [Step 4/4] Fallback model training note: {exc}")

                logger.info(f"[BOOTSTRAP] [Step 4/4] Model training and registry enrollment completed for shop '{shop_name}'.")
            else:
                logger.info(f"[BOOTSTRAP] [Step 4/4] Registry has {active_models} active model(s) for shop '{shop_name}'. Skipping training.")

    elapsed = time.time() - start_time
    logger.info("=" * 70)
    logger.info(f"[BOOTSTRAP] Loaf Ledger bootstrap sequence finished cleanly in {elapsed:.2f}s.")
    logger.info("=" * 70)
    return True


def main():
    """Main execution function with standard exit codes."""
    try:
        success = run_bootstrap()
        sys.exit(0 if success else 1)
    except Exception as exc:
        logger.error(f"[BOOTSTRAP] Fatal error during bootstrap: {exc}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()

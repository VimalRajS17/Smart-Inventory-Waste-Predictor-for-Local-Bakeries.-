"""Background scheduler integration for nightly bakery model retraining."""

import logging
import os
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger

logger = logging.getLogger("loafledger.scheduler")
_scheduler = None


def run_nightly_retrain(app):
    """Task executed by the nightly scheduler inside the Flask application context."""
    with app.app_context():
        logger.info("Executing scheduled nightly model retraining scan...")
        from ml.train import retrain_all_eligible_shops
        from app.db import get_db

        try:
            db = get_db()
            results = retrain_all_eligible_shops(db=db, force=False, min_new_rows=30)
            retrained = sum(1 for r in results if r["retrained"])
            logger.info(f"Nightly retraining finished: {retrained} of {len(results)} shops retrained.")
        except Exception as exc:
            logger.error(f"Error during nightly retraining: {exc}", exc_info=True)


def init_scheduler(app):
    """Initialize APScheduler if enabled by RETRAIN_SCHEDULE_ENABLED environment variable."""
    global _scheduler
    enabled_env = os.getenv("RETRAIN_SCHEDULE_ENABLED", "0").lower()
    is_enabled = enabled_env in ("1", "true", "yes")

    # In development mode with Werkzeug reloader, avoid duplicate timers in main process
    if os.environ.get("WERKZEUG_RUN_MAIN") == "false":
        return None

    if not is_enabled:
        return None

    if _scheduler is not None and _scheduler.running:
        return _scheduler

    try:
        cron_hour = int(os.getenv("RETRAIN_CRON_HOUR", "3"))
        cron_minute = int(os.getenv("RETRAIN_CRON_MINUTE", "0"))

        _scheduler = BackgroundScheduler(daemon=True)
        _scheduler.add_job(
            func=run_nightly_retrain,
            trigger=CronTrigger(hour=cron_hour, minute=cron_minute),
            args=[app],
            id="nightly_model_retrain",
            replace_existing=True,
            name="Nightly Bakery Model Retraining",
        )
        _scheduler.start()
        logger.info(f"Nightly retraining scheduler started (Cron: {cron_hour:02d}:{cron_minute:02d} daily).")
        return _scheduler
    except Exception as exc:
        logger.error(f"Failed to start APScheduler: {exc}")
        return None


def shutdown_scheduler():
    """Shutdown background scheduler if active."""
    global _scheduler
    if _scheduler is not None and _scheduler.running:
        _scheduler.shutdown(wait=False)
        _scheduler = None


def get_scheduler_status():
    """Return status string of the background scheduler."""
    global _scheduler
    if _scheduler is not None and _scheduler.running:
        return "running"
    enabled_env = os.getenv("RETRAIN_SCHEDULE_ENABLED", "0").lower()
    if enabled_env in ("1", "true", "yes"):
        return "configured"
    return "disabled"



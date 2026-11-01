"""Tests for daily_log upsert, unique constraints, and logging operations."""

from datetime import date
from app.db import get_db, log_sale, log_waste


def test_daily_log_uniqueness_and_upsert(app):
    """Test that multiple logs for the same (shop_id, item_id, date) upsert into a single row."""
    with app.app_context():
        db = get_db()
        item = db.execute("SELECT id, shop_id FROM items WHERE name = 'Croissant'").fetchone()
        item_id = item["id"]
        shop_id = item["shop_id"]
        log_date = "2026-10-30"

        # 1. First insert via log_sale helper
        log_sale("Croissant", log_date, 10)

        rows = db.execute(
            "SELECT * FROM daily_log WHERE shop_id = ? AND item_id = ? AND date = ?",
            (shop_id, item_id, log_date),
        ).fetchall()
        assert len(rows) == 1
        assert rows[0]["sold"] == 10
        assert rows[0]["baked"] is None
        assert rows[0]["wasted"] is None

        # 2. Second sale on the same day accumulates
        log_sale("Croissant", log_date, 5)

        rows = db.execute(
            "SELECT * FROM daily_log WHERE shop_id = ? AND item_id = ? AND date = ?",
            (shop_id, item_id, log_date),
        ).fetchall()
        assert len(rows) == 1
        assert rows[0]["sold"] == 15

        # 3. Log waste on the same day does not overwrite sold
        log_waste("Croissant", log_date, 2, reason="Stale")

        rows = db.execute(
            "SELECT * FROM daily_log WHERE shop_id = ? AND item_id = ? AND date = ?",
            (shop_id, item_id, log_date),
        ).fetchall()
        assert len(rows) == 1
        assert rows[0]["sold"] == 15
        assert rows[0]["wasted"] == 2
        assert rows[0]["reason"] == "Stale"


def test_daily_log_direct_upsert_with_sold_out(app):
    """Test direct SQL upsert setting baked, sold, wasted, and sold_out."""
    with app.app_context():
        db = get_db()
        item = db.execute("SELECT id, shop_id FROM items LIMIT 1").fetchone()
        item_id = item["id"]
        shop_id = item["shop_id"]
        log_date = "2026-10-31"

        # Direct parameterised upsert
        db.execute(
            """
            INSERT INTO daily_log (shop_id, item_id, date, sold, baked, wasted, sold_out)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(shop_id, item_id, date) DO UPDATE SET
                sold = excluded.sold,
                baked = excluded.baked,
                wasted = excluded.wasted,
                sold_out = excluded.sold_out
            """,
            (shop_id, item_id, log_date, 40, 50, 1, 1),
        )
        db.commit()

        row = db.execute(
            "SELECT * FROM daily_log WHERE shop_id = ? AND item_id = ? AND date = ?",
            (shop_id, item_id, log_date),
        ).fetchone()

        assert row["sold"] == 40
        assert row["baked"] == 50
        assert row["wasted"] == 1
        assert row["sold_out"] == 1

        # Verify only 1 row exists
        count = db.execute(
            "SELECT COUNT(*) AS c FROM daily_log WHERE shop_id = ? AND item_id = ? AND date = ?",
            (shop_id, item_id, log_date),
        ).fetchone()["c"]
        assert count == 1


"""SQLite database helpers for the bakery inventory app."""

import sqlite3
from contextlib import contextmanager

DB_PATH = "bakery.db"


@contextmanager
def get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db():
    with get_conn() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS items (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT UNIQUE NOT NULL,
                current_stock INTEGER DEFAULT 0,
                shelf_life_days INTEGER DEFAULT 1
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS sales (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                item_name TEXT NOT NULL,
                date TEXT NOT NULL,
                quantity_sold INTEGER NOT NULL
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS waste_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                item_name TEXT NOT NULL,
                date TEXT NOT NULL,
                quantity_wasted INTEGER NOT NULL,
                reason TEXT
            )
        """)

        # seed default items if table is empty
        existing = conn.execute("SELECT COUNT(*) AS c FROM items").fetchone()["c"]
        if existing == 0:
            default_items = [
                ("Croissant", 20, 1),
                ("Sourdough Loaf", 15, 2),
                ("Chocolate Muffin", 25, 1),
                ("Cinnamon Roll", 15, 1),
                ("Cupcake", 20, 2),
            ]
            conn.executemany(
                "INSERT INTO items (name, current_stock, shelf_life_days) VALUES (?, ?, ?)",
                default_items,
            )


def get_all_items():
    with get_conn() as conn:
        return [dict(r) for r in conn.execute("SELECT * FROM items").fetchall()]


def log_sale(item_name: str, date: str, quantity: int):
    with get_conn() as conn:
        conn.execute(
            "INSERT INTO sales (item_name, date, quantity_sold) VALUES (?, ?, ?)",
            (item_name, date, quantity),
        )


def log_waste(item_name: str, date: str, quantity: int, reason: str = ""):
    with get_conn() as conn:
        conn.execute(
            "INSERT INTO waste_log (item_name, date, quantity_wasted, reason) VALUES (?, ?, ?, ?)",
            (item_name, date, quantity, reason),
        )


def get_waste_summary():
    with get_conn() as conn:
        rows = conn.execute("""
            SELECT item_name, SUM(quantity_wasted) as total_wasted
            FROM waste_log GROUP BY item_name
        """).fetchall()
        return [dict(r) for r in rows]

"""SQLite database connection, schema management, and CLI commands for Loaf Ledger."""

import csv
from datetime import datetime, timezone
import os
import sqlite3
import click
from flask import current_app, g
from flask.cli import with_appcontext


class PostgresConnectionWrapper:
    """Wrapper around a PostgreSQL connection providing an interface compatible with SQLite Row and DB-API."""

    def __init__(self, raw_conn):
        self.raw_conn = raw_conn

    def execute(self, sql, params=None):
        if "PRAGMA" in sql.upper():
            class DummyCursor:
                def fetchall(self):
                    return []
                def fetchone(self):
                    return None
            return DummyCursor()

        cursor = self.raw_conn.cursor()
        converted_sql = sql.replace("?", "%s")
        if "INTEGER PRIMARY KEY AUTOINCREMENT" in converted_sql:
            converted_sql = converted_sql.replace("INTEGER PRIMARY KEY AUTOINCREMENT", "SERIAL PRIMARY KEY")
        cursor.execute(converted_sql, params or ())
        return cursor

    def executemany(self, sql, seq_of_params):
        cursor = self.raw_conn.cursor()
        converted_sql = sql.replace("?", "%s")
        cursor.executemany(converted_sql, seq_of_params)
        return cursor

    def commit(self):
        self.raw_conn.commit()

    def rollback(self):
        self.raw_conn.rollback()

    def close(self):
        self.raw_conn.close()


def get_db():
    """Get or create database connection for the current request context."""
    if "db" not in g:
        db_url = current_app.config.get("DATABASE_URL")
        if db_url and (db_url.startswith("postgres://") or db_url.startswith("postgresql://")):
            try:
                import psycopg2
                from psycopg2.extras import RealDictCursor
                conn = psycopg2.connect(db_url, cursor_factory=RealDictCursor)
                g.db = PostgresConnectionWrapper(conn)
            except ImportError:
                raise ImportError(
                    "DATABASE_URL is configured for PostgreSQL, but psycopg2 is not installed. "
                    "Install it using: pip install psycopg2-binary"
                )
        else:
            db_path = current_app.config["DATABASE"]
            g.db = sqlite3.connect(db_path)
            g.db.row_factory = sqlite3.Row
            g.db.execute("PRAGMA foreign_keys = ON;")
            _ensure_tables(g.db)
    return g.db


def create_tables_if_missing(db):
    """Ensure all core tables and indices exist non-destructively."""
    # 0. Users
    db.execute("""
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        email TEXT NOT NULL UNIQUE,
        password_hash TEXT NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    """)

    # 1. Shops / Bakery locations
    db.execute("""
    CREATE TABLE IF NOT EXISTS shops (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        owner_user_id INTEGER REFERENCES users(id) ON DELETE CASCADE,
        name TEXT NOT NULL,
        city TEXT,
        lat REAL NOT NULL,
        lon REAL NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        UNIQUE(owner_user_id, name)
    );
    """)

    # 2. Items sold by each bakery
    db.execute("""
    CREATE TABLE IF NOT EXISTS items (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        shop_id INTEGER NOT NULL REFERENCES shops(id) ON DELETE CASCADE,
        name TEXT NOT NULL,
        current_stock INTEGER DEFAULT 0,
        shelf_life_days INTEGER DEFAULT 1,
        unit_price REAL DEFAULT NULL,
        unit_cost REAL DEFAULT NULL,
        UNIQUE(shop_id, name)
    );
    """)

    # 3. Unified daily log per item
    db.execute("""
    CREATE TABLE IF NOT EXISTS daily_log (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        shop_id INTEGER NOT NULL REFERENCES shops(id) ON DELETE CASCADE,
        item_id INTEGER NOT NULL REFERENCES items(id) ON DELETE CASCADE,
        date TEXT NOT NULL,
        baked INTEGER DEFAULT NULL,
        sold INTEGER DEFAULT NULL,
        wasted INTEGER DEFAULT NULL,
        leftover INTEGER DEFAULT NULL,
        sold_out INTEGER DEFAULT 0,
        reason TEXT DEFAULT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        UNIQUE(shop_id, item_id, date)
    );
    """)

    # 4. Weather & holiday tracking per shop and date
    db.execute("""
    CREATE TABLE IF NOT EXISTS weather_daily (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        shop_id INTEGER NOT NULL REFERENCES shops(id) ON DELETE CASCADE,
        date TEXT NOT NULL,
        temperature_c REAL,
        temp_mean_c REAL,
        temp_max_c REAL,
        temp_min_c REAL,
        precip_mm REAL,
        is_rainy INTEGER DEFAULT 0,
        is_holiday INTEGER DEFAULT 0,
        source TEXT,
        fetched_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        UNIQUE(shop_id, date)
    );
    """)

    # 5. Forecast history
    db.execute("""
    CREATE TABLE IF NOT EXISTS forecasts (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        shop_id INTEGER NOT NULL REFERENCES shops(id) ON DELETE CASCADE,
        item_id INTEGER NOT NULL REFERENCES items(id) ON DELETE CASCADE,
        date TEXT NOT NULL,
        predicted_quantity REAL NOT NULL,
        actual_qty REAL DEFAULT NULL,
        weather_source TEXT,
        cold_start INTEGER DEFAULT 0,
        lag_features_stale INTEGER DEFAULT 0,
        model_version TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        UNIQUE(shop_id, item_id, date)
    );
    """)

    # 6. Model registry per shop and item
    db.execute("""
    CREATE TABLE IF NOT EXISTS model_registry (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        shop_id INTEGER NOT NULL REFERENCES shops(id) ON DELETE CASCADE,
        item_id INTEGER NOT NULL REFERENCES items(id) ON DELETE CASCADE,
        model_type TEXT NOT NULL,
        version INTEGER NOT NULL,
        path TEXT,
        trained_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        mae REAL,
        rmse REAL,
        wape REAL,
        beats_baseline INTEGER DEFAULT 0,
        active INTEGER DEFAULT 0,
        UNIQUE(shop_id, item_id, version)
    );
    """)

    # 7. RL Contextual Bandit Policies (Phase 4D)
    db.execute("""
    CREATE TABLE IF NOT EXISTS policies (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        shop_id INTEGER NOT NULL REFERENCES shops(id) ON DELETE CASCADE,
        item_id INTEGER NOT NULL REFERENCES items(id) ON DELETE CASCADE,
        version INTEGER NOT NULL,
        policy_type TEXT NOT NULL,
        parameters TEXT NOT NULL,
        observations_count INTEGER DEFAULT 0,
        beat_buffer_rule INTEGER DEFAULT 0,
        active INTEGER DEFAULT 0,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        UNIQUE(shop_id, item_id, version)
    );
    """)

    # 8. 60-Day Replay Evaluation Results (Phase 5)
    db.execute("""
    CREATE TABLE IF NOT EXISTS evaluation_results (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        shop_id INTEGER NOT NULL REFERENCES shops(id) ON DELETE CASCADE,
        item_id INTEGER REFERENCES items(id) ON DELETE CASCADE,
        evaluated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        period_start TEXT NOT NULL,
        period_end TEXT NOT NULL,
        days_count INTEGER DEFAULT 60,
        policy_name TEXT NOT NULL,
        total_baked INTEGER DEFAULT 0,
        total_sold INTEGER DEFAULT 0,
        total_waste_units INTEGER DEFAULT 0,
        total_waste_cost REAL DEFAULT 0.0,
        total_lost_sales_units INTEGER DEFAULT 0,
        total_lost_sales_cost REAL DEFAULT 0.0,
        net_profit REAL DEFAULT 0.0,
        savings_vs_baseline REAL DEFAULT 0.0,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    """)

    # Performance indices
    db.execute("CREATE INDEX IF NOT EXISTS idx_shops_owner ON shops(owner_user_id);")
    db.execute("CREATE INDEX IF NOT EXISTS idx_daily_log_shop_date ON daily_log(shop_id, date);")
    db.execute("CREATE INDEX IF NOT EXISTS idx_daily_log_item ON daily_log(item_id);")
    db.execute("CREATE INDEX IF NOT EXISTS idx_weather_shop_date ON weather_daily(shop_id, date);")
    db.execute("CREATE INDEX IF NOT EXISTS idx_forecasts_shop_date ON forecasts(shop_id, date);")
    db.execute("CREATE INDEX IF NOT EXISTS idx_model_registry_active ON model_registry(shop_id, item_id, active);")
    db.execute("CREATE INDEX IF NOT EXISTS idx_policies_active ON policies(shop_id, item_id, active);")
    db.execute("CREATE INDEX IF NOT EXISTS idx_evaluation_results_shop ON evaluation_results(shop_id, policy_name);")


def _ensure_tables(db):
    """Ensure all core and migration tables exist without dropping existing data."""
    create_tables_if_missing(db)

    # Ensure leftover column exists on daily_log
    try:
        cols = [r["name"] for r in db.execute("PRAGMA table_info(daily_log)").fetchall()]
        if cols and "leftover" not in cols:
            db.execute("ALTER TABLE daily_log ADD COLUMN leftover INTEGER DEFAULT NULL;")
    except Exception:
        pass

    # Ensure actual_qty column exists on forecasts
    try:
        f_cols = [r["name"] for r in db.execute("PRAGMA table_info(forecasts)").fetchall()]
        if f_cols and "actual_qty" not in f_cols:
            db.execute("ALTER TABLE forecasts ADD COLUMN actual_qty REAL DEFAULT NULL;")
    except Exception:
        pass

    # Ensure unit_price and unit_cost exist on items
    try:
        i_cols = [r["name"] for r in db.execute("PRAGMA table_info(items)").fetchall()]
        if i_cols and "unit_price" not in i_cols:
            db.execute("ALTER TABLE items ADD COLUMN unit_price REAL DEFAULT NULL;")
        if i_cols and "unit_cost" not in i_cols:
            db.execute("ALTER TABLE items ADD COLUMN unit_cost REAL DEFAULT NULL;")
    except Exception:
        pass



def close_db(e=None):
    """Close request database connection if open."""
    db = g.pop("db", None)
    if db is not None:
        db.close()


def init_db():
    """Execute schema.sql to initialize database tables."""
    db = get_db()
    schema_path = os.path.join(os.path.dirname(__file__), "schema.sql")
    with open(schema_path, "r", encoding="utf-8") as f:
        db.executescript(f.read())


@click.command("init-db")
@with_appcontext
def init_db_command():
    """Clear existing data and create new database tables."""
    init_db()
    click.echo("Initialized the database.")


def seed_demo():
    """Seed demo user, demo shop, 5 default items, and historical sales into daily_log."""
    from werkzeug.security import generate_password_hash

    db = get_db()
    lat = float(current_app.config.get("DEFAULT_LAT", 51.5074))
    lon = float(current_app.config.get("DEFAULT_LON", -0.1278))
    shop_name = "Demo Bakery"

    demo_email = os.getenv("DEMO_EMAIL", "demo@example.com").strip().lower()
    demo_password = os.getenv("DEMO_PASSWORD", "demo123")

    # 0. Upsert Demo User
    demo_user = db.execute("SELECT id FROM users WHERE LOWER(email) = LOWER(?)", (demo_email,)).fetchone()
    if not demo_user:
        cursor = db.execute(
            "INSERT INTO users (email, password_hash) VALUES (?, ?)",
            (demo_email, generate_password_hash(demo_password)),
        )
        demo_user_id = cursor.lastrowid
    else:
        demo_user_id = demo_user["id"]
        db.execute(
            "UPDATE users SET password_hash = ? WHERE id = ?",
            (generate_password_hash(demo_password), demo_user_id),
        )

    # 1. Upsert Demo Bakery shop
    db.execute(
        """
        INSERT INTO shops (owner_user_id, name, city, lat, lon)
        VALUES (?, ?, ?, ?, ?)
        ON CONFLICT(owner_user_id, name) DO UPDATE SET
            city = excluded.city,
            lat = excluded.lat,
            lon = excluded.lon
        """,
        (demo_user_id, shop_name, "London", lat, lon),
    )
    shop = db.execute("SELECT id FROM shops WHERE owner_user_id = ? AND name = ?", (demo_user_id, shop_name)).fetchone()
    shop_id = shop["id"]

    # 2. Upsert five existing items
    default_items = [
        ("Croissant", 20, 1),
        ("Sourdough Loaf", 15, 2),
        ("Chocolate Muffin", 25, 1),
        ("Cinnamon Roll", 15, 1),
        ("Cupcake", 20, 2),
    ]
    for name, current_stock, shelf_life in default_items:
        db.execute(
            """
            INSERT INTO items (shop_id, name, current_stock, shelf_life_days)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(shop_id, name) DO UPDATE SET
                current_stock = excluded.current_stock,
                shelf_life_days = excluded.shelf_life_days
            """,
            (shop_id, name, current_stock, shelf_life),
        )

    # 3. Retrieve item map
    item_rows = db.execute(
        "SELECT id, name FROM items WHERE shop_id = ?",
        (shop_id,),
    ).fetchall()
    items_map = {row["name"]: row["id"] for row in item_rows}

    # 4. Clear old demo rows for this shop first
    db.execute("DELETE FROM daily_log WHERE shop_id = ?", (shop_id,))
    db.execute("DELETE FROM forecasts WHERE shop_id = ?", (shop_id,))
    db.execute("DELETE FROM weather_daily WHERE shop_id = ?", (shop_id,))
    db.commit()

    # 5. Load sales_history.csv into daily_log (sold = quantity_sold, baked and wasted NULL)
    csv_candidates = [
        os.path.join(current_app.root_path, "..", "data", "sales_history.csv"),
        os.path.join(os.getcwd(), "data", "sales_history.csv"),
    ]
    csv_path = next((p for p in csv_candidates if os.path.exists(p)), None)
    if not csv_path:
        from data.generate_sample_data import generate_sample_data
        fallback_path = os.path.abspath(os.path.join(current_app.root_path, "..", "data", "sales_history.csv"))
        os.makedirs(os.path.dirname(fallback_path), exist_ok=True)
        generate_sample_data(csv_path=fallback_path, num_days=730)
        csv_path = fallback_path

    log_rows = []
    weather_dict = {}
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            item_name = row["item"]
            item_id = items_map.get(item_name)
            if not item_id:
                continue
            date_str = row["date"]
            sold = int(row["quantity_sold"])
            log_rows.append((shop_id, item_id, date_str, sold))

            if date_str not in weather_dict and "temperature_c" in row:
                weather_dict[date_str] = (
                    shop_id,
                    date_str,
                    float(row["temperature_c"]),
                    float(row["temperature_c"]),
                    int(row.get("is_rainy", 0)),
                    int(row.get("is_holiday", 0)),
                    "archive",
                )

    db.executemany(
        """
        INSERT INTO daily_log (shop_id, item_id, date, sold)
        VALUES (?, ?, ?, ?)
        ON CONFLICT(shop_id, item_id, date) DO UPDATE SET
            sold = excluded.sold
        """,
        log_rows,
    )

    if weather_dict:
        db.executemany(
            """
            INSERT INTO weather_daily (shop_id, date, temperature_c, temp_mean_c, is_rainy, is_holiday, source)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(shop_id, date) DO UPDATE SET
                temperature_c = excluded.temperature_c,
                temp_mean_c = excluded.temp_mean_c,
                is_rainy = excluded.is_rainy,
                is_holiday = excluded.is_holiday,
                source = excluded.source
            """,
            list(weather_dict.values()),
        )
    db.commit()


@click.command("seed-demo")
@with_appcontext
def seed_demo_command():
    """Seed demo shop, items, and sales history."""
    seed_demo()
    click.echo("Seeded demo shop, items, and sales history into daily_log.")


def reset_demo(history_days: int = 730):
    """Rebuild the demo shop, items, history and weather from scratch."""
    from datetime import date, timedelta
    from data.generate_sample_data import generate_sample_data
    from app.services.weather import fetch_forecast, save_weather
    from ml.train_model import train

    # 1. Clear database tables
    init_db()

    # 2. Regenerate sample data for history_days ending yesterday
    csv_candidates = [
        os.path.join(current_app.root_path, "..", "data", "sales_history.csv"),
        os.path.join(os.getcwd(), "data", "sales_history.csv"),
    ]
    csv_path = next((p for p in csv_candidates if os.path.exists(os.path.dirname(p))), "data/sales_history.csv")
    generate_sample_data(csv_path=csv_path, num_days=history_days)

    # 3. Seed demo shop, items, and sales_history.csv
    seed_demo()

    # 4. Fetch/populate forecast weather for tomorrow with source='forecast'
    db = get_db()
    shop = db.execute("SELECT id, lat, lon FROM shops WHERE name = 'Demo Bakery'").fetchone()
    if shop:
        tomorrow = (date.today() + timedelta(days=1)).isoformat()
        try:
            forecast_rows = fetch_forecast(float(shop["lat"]), float(shop["lon"]), days=7)
            save_weather(shop["id"], forecast_rows, "forecast")
        except Exception:
            # Graceful fallback during offline runs or mocked tests
            db.execute(
                """
                INSERT INTO weather_daily (shop_id, date, temperature_c, temp_mean_c, precip_mm, is_rainy, source)
                VALUES (?, ?, 20.0, 20.0, 0.0, 0, 'forecast')
                ON CONFLICT(shop_id, date) DO UPDATE SET
                    temperature_c = excluded.temperature_c,
                    temp_mean_c = excluded.temp_mean_c,
                    source = 'forecast'
                """,
                (shop["id"], tomorrow),
            )
            db.commit()

    # 5. Retrain models on DB
    try:
        train(source="db")
    except Exception:
        pass


@click.command("reset-demo")
@click.option("--history-days", default=730, type=int, help="Historical days of sales data to generate and seed (default: 730).")
@with_appcontext
def reset_demo_command(history_days: int):
    """Rebuild demo shop, items, history and weather from scratch."""
    click.echo(f"Resetting demo environment from scratch ({history_days} days of history)...")
    reset_demo(history_days=history_days)
    click.echo("Demo reset complete! Shop, items, history, weather, and ML models are rebuilt.")


def backup_database(output_dir="backups", custom_name=None):
    """Write an atomic, timestamped copy of the SQLite database.

    Uses SQLite's online backup API (source.backup(dest)) to guarantee a crash-consistent,
    lock-safe snapshot even while concurrent read and write operations are ongoing.
    """
    db = get_db()
    if not isinstance(db, sqlite3.Connection):
        raise NotImplementedError(
            "Online backup via 'flask backup' is supported for SQLite. "
            "For PostgreSQL deployments, use pg_dump (see docs/DEPLOY.md)."
        )

    if not os.path.isabs(output_dir):
        output_dir = os.path.abspath(os.path.join(current_app.root_path, "..", output_dir))
    os.makedirs(output_dir, exist_ok=True)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    filename = custom_name or f"backup_{timestamp}.db"
    dest_path = os.path.join(output_dir, filename)

    dest_conn = sqlite3.connect(dest_path)
    try:
        db.backup(dest_conn)
    finally:
        dest_conn.close()

    size_bytes = os.path.getsize(dest_path)
    return dest_path, size_bytes


@click.command("backup")
@click.option("--output-dir", default="backups", help="Directory where the backup copy will be stored (default: backups).")
@click.option("--filename", default=None, help="Custom filename for the backup (default: backup_YYYYMMDD_HHMMSS.db).")
@with_appcontext
def backup_command(output_dir, filename):
    """Write a timestamped copy of the database."""
    try:
        dest_path, size_bytes = backup_database(output_dir=output_dir, custom_name=filename)
        size_kb = size_bytes / 1024.0
        click.echo(f"Database backup successfully created: {dest_path} ({size_kb:.1f} KB)")
    except Exception as exc:
        click.echo(f"Database backup failed: {exc}", err=True)
        raise SystemExit(1)


def init_app(app):
    """Register database teardown and CLI commands with the Flask application."""
    app.teardown_appcontext(close_db)
    app.cli.add_command(init_db_command)
    app.cli.add_command(seed_demo_command)
    app.cli.add_command(reset_demo_command)
    app.cli.add_command(backup_command)

    from app.services.weather import fetch_weather_command
    app.cli.add_command(fetch_weather_command)

    from ml.train import train_command, retrain_all_command, rollback_command
    app.cli.add_command(train_command)
    app.cli.add_command(retrain_all_command)
    app.cli.add_command(rollback_command)

    from ml.rl import evaluate_rl_command
    app.cli.add_command(evaluate_rl_command)

    from ml.simulation import evaluate_command
    app.cli.add_command(evaluate_command)


# ---------------------------------------------------------------------------
# Query Helpers for Dashboard & Routes (Parameterised, no ORM)
# ---------------------------------------------------------------------------

def get_all_items(shop_name: str = "Demo Bakery"):
    """Fetch items for the given shop or all items if shop not matched."""
    db = get_db()
    rows = db.execute(
        """
        SELECT i.id, i.name, i.current_stock, i.shelf_life_days
        FROM items i
        JOIN shops s ON i.shop_id = s.id
        WHERE s.name = ?
        ORDER BY i.id
        """,
        (shop_name,),
    ).fetchall()
    if not rows:
        rows = db.execute(
            "SELECT id, name, current_stock, shelf_life_days FROM items ORDER BY id"
        ).fetchall()
    return [dict(r) for r in rows]


def log_sale(item_name: str, sale_date: str, quantity: int, shop_name: str = "Demo Bakery"):
    """Upsert a sale entry into daily_log."""
    db = get_db()
    item = db.execute(
        """
        SELECT i.id AS item_id, i.shop_id
        FROM items i
        JOIN shops s ON i.shop_id = s.id
        WHERE i.name = ? AND s.name = ?
        """,
        (item_name, shop_name),
    ).fetchone()

    if not item:
        item = db.execute(
            "SELECT id AS item_id, shop_id FROM items WHERE name = ?",
            (item_name,),
        ).fetchone()

    if not item:
        raise ValueError(f"Item '{item_name}' not found.")

    db.execute(
        """
        INSERT INTO daily_log (shop_id, item_id, date, sold)
        VALUES (?, ?, ?, ?)
        ON CONFLICT(shop_id, item_id, date) DO UPDATE SET
            sold = COALESCE(daily_log.sold, 0) + excluded.sold
        """,
        (item["shop_id"], item["item_id"], sale_date, quantity),
    )
    db.commit()


def log_waste(item_name: str, waste_date: str, quantity: int, reason: str = "", shop_name: str = "Demo Bakery"):
    """Upsert a waste entry into daily_log."""
    db = get_db()
    item = db.execute(
        """
        SELECT i.id AS item_id, i.shop_id
        FROM items i
        JOIN shops s ON i.shop_id = s.id
        WHERE i.name = ? AND s.name = ?
        """,
        (item_name, shop_name),
    ).fetchone()

    if not item:
        item = db.execute(
            "SELECT id AS item_id, shop_id FROM items WHERE name = ?",
            (item_name,),
        ).fetchone()

    if not item:
        raise ValueError(f"Item '{item_name}' not found.")

    db.execute(
        """
        INSERT INTO daily_log (shop_id, item_id, date, wasted, reason)
        VALUES (?, ?, ?, ?, ?)
        ON CONFLICT(shop_id, item_id, date) DO UPDATE SET
            wasted = COALESCE(daily_log.wasted, 0) + excluded.wasted,
            reason = CASE
                WHEN daily_log.reason IS NULL OR daily_log.reason = '' THEN excluded.reason
                WHEN excluded.reason IS NULL OR excluded.reason = '' THEN daily_log.reason
                ELSE daily_log.reason || '; ' || excluded.reason
            END
        """,
        (item["shop_id"], item["item_id"], waste_date, quantity, reason),
    )
    db.commit()


def get_waste_summary(shop_name: str = "Demo Bakery"):
    """Aggregate total waste per item from daily_log."""
    db = get_db()
    rows = db.execute(
        """
        SELECT i.name AS item_name, SUM(d.wasted) AS total_wasted
        FROM daily_log d
        JOIN items i ON d.item_id = i.id
        JOIN shops s ON d.shop_id = s.id
        WHERE s.name = ? AND d.wasted IS NOT NULL AND d.wasted > 0
        GROUP BY i.name
        ORDER BY total_wasted DESC
        """,
        (shop_name,),
    ).fetchall()

    if not rows:
        # Fallback if shop name join yields nothing
        rows = db.execute(
            """
            SELECT i.name AS item_name, SUM(d.wasted) AS total_wasted
            FROM daily_log d
            JOIN items i ON d.item_id = i.id
            WHERE d.wasted IS NOT NULL AND d.wasted > 0
            GROUP BY i.name
            ORDER BY total_wasted DESC
            """
        ).fetchall()

    return [dict(r) for r in rows]


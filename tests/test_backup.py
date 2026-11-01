"""Tests for database backup functionality and 'flask backup' CLI command."""

import os
import sqlite3
import pytest
from app import create_app
from app.db import backup_database


@pytest.fixture
def app_with_demo(tmp_path):
    """Create and configure a clean app instance with initialized tables."""
    db_path = str(tmp_path / "test_source.db")
    app = create_app({
        "TESTING": True,
        "DATABASE": db_path,
        "SECRET_KEY": "test-backup-key",
        "WTF_CSRF_ENABLED": False,
        "RL_ENABLED": False,
    })

    with app.app_context():
        from app.db import init_db, get_db
        init_db()
        db = get_db()
        db.execute(
            "INSERT INTO users (email, password_hash) VALUES ('test@example.com', 'hash')"
        )
        user_id = db.execute("SELECT id FROM users WHERE email = 'test@example.com'").fetchone()["id"]
        db.execute(
            "INSERT INTO shops (owner_user_id, name, city, lat, lon) VALUES (?, ?, ?, ?, ?)",
            (user_id, "Test Bakery", "London", 51.5, -0.12),
        )
        shop_id = db.execute("SELECT id FROM shops WHERE name = 'Test Bakery'").fetchone()["id"]
        db.execute(
            "INSERT INTO items (shop_id, name, current_stock, shelf_life_days) VALUES (?, ?, ?, ?)",
            (shop_id, "Test Croissant", 20, 1),
        )
        db.commit()

    yield app


def test_backup_database_creates_valid_copy(app_with_demo, tmp_path):
    """Verify backup_database copies SQLite tables and rows accurately."""
    backup_dir = str(tmp_path / "backups_test")

    with app_with_demo.app_context():
        dest_path, size_bytes = backup_database(output_dir=backup_dir, custom_name="test_copy.db")

        assert os.path.exists(dest_path)
        assert size_bytes > 0
        assert os.path.basename(dest_path) == "test_copy.db"

        # Verify contents of backup database
        conn = sqlite3.connect(dest_path)
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()

        shop = cur.execute("SELECT name FROM shops WHERE name = 'Test Bakery'").fetchone()
        assert shop is not None
        assert shop["name"] == "Test Bakery"

        item = cur.execute("SELECT name, current_stock FROM items WHERE name = 'Test Croissant'").fetchone()
        assert item is not None
        assert item["name"] == "Test Croissant"
        assert item["current_stock"] == 20

        conn.close()


def test_backup_cli_command(app_with_demo, tmp_path):
    """Verify the 'flask backup' Click command runs successfully and reports size."""
    runner = app_with_demo.test_cli_runner()
    backup_dir = str(tmp_path / "cli_backups")
    custom_filename = "cli_snapshot.db"

    result = runner.invoke(args=["backup", "--output-dir", backup_dir, "--filename", custom_filename])
    assert result.exit_code == 0
    assert "Database backup successfully created" in result.output
    assert custom_filename in result.output

    expected_file = os.path.join(backup_dir, custom_filename)
    assert os.path.exists(expected_file)
    assert os.path.getsize(expected_file) > 0


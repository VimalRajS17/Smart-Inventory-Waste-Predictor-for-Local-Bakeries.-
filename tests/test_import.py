"""Tests for CSV sales history import, Kaggle converter, and history visualization."""

import io
import os
from unittest.mock import patch
import pytest

from scripts.load_kaggle import detect_columns, process_kaggle_file


def _make_csv_file(content: str, filename: str = "test.csv"):
    """Helper to create a BytesIO file tuple for multipart upload."""
    return (io.BytesIO(content.encode("utf-8")), filename)


# ---------------------------------------------------------------------------
# POST /api/import/csv Tests
# ---------------------------------------------------------------------------

def test_import_good_rows(client, app):
    """Valid CSV rows are imported into daily_log in a single transaction."""
    csv_data = (
        "date,item,sold,baked,wasted\n"
        "2026-03-01,Croissant,30,35,5\n"
        "2026-03-01,Sourdough Loaf,20,20,0\n"
        "2026-03-02,Croissant,25,30,5\n"
    )
    with patch("app.api.routes.sync_weather_for_date_range") as mock_weather:
        mock_weather.return_value = {"status": "ok", "days_fetched": 2}
        resp = client.post(
            "/api/import/csv",
            data={
                "file": _make_csv_file(csv_data),
            },
            content_type="multipart/form-data",
        )

    assert resp.status_code == 200
    data = resp.get_json()
    assert data["status"] == "success"
    assert data["rows_imported"] == 3
    assert data["rows_skipped"] == 0
    assert len(data["skipped_rows"]) == 0
    assert data["date_range"] == {"start": "2026-03-01", "end": "2026-03-02"}

    # Verify rows persisted in DB
    with app.app_context():
        from app.db import get_db
        db = get_db()
        rows = db.execute("SELECT item_id, date, sold, baked, wasted FROM daily_log WHERE date IN ('2026-03-01', '2026-03-02')").fetchall()
        assert len(rows) == 3


def test_import_bad_dates(client, app):
    """Rows with invalid dates are skipped and reported with 1-based row numbers and reasons."""
    csv_data = (
        "date,item,sold\n"
        "2026-03-01,Croissant,20\n"
        "2026-02-31,Croissant,25\n"      # Invalid date (row 3)
        "not-a-date,Croissant,15\n"      # Malformed date (row 4)
        "2026-03-02,Croissant,18\n"
    )
    with patch("app.api.routes.sync_weather_for_date_range") as mock_weather:
        mock_weather.return_value = {"status": "ok", "days_fetched": 2}
        resp = client.post(
            "/api/import/csv",
            data={"file": _make_csv_file(csv_data)},
            content_type="multipart/form-data",
        )

    assert resp.status_code == 200
    data = resp.get_json()
    assert data["rows_imported"] == 2
    assert data["rows_skipped"] == 2

    reasons = {r["row"]: r["reason"] for r in data["skipped_rows"]}
    assert 3 in reasons
    assert "Invalid date '2026-02-31'" in reasons[3]
    assert 4 in reasons
    assert "Invalid date 'not-a-date'" in reasons[4]


def test_import_negative_numbers(client, app):
    """Rows with negative quantities for sold, baked, or wasted are skipped."""
    csv_data = (
        "date,item,sold,baked,wasted\n"
        "2026-03-01,Croissant,20,20,0\n"
        "2026-03-02,Croissant,-5,20,0\n"   # Row 3: negative sold
        "2026-03-03,Croissant,20,-3,0\n"   # Row 4: negative baked
        "2026-03-04,Croissant,20,20,-1\n"  # Row 5: negative wasted
    )
    with patch("app.api.routes.sync_weather_for_date_range") as mock_weather:
        mock_weather.return_value = {"status": "ok", "days_fetched": 1}
        resp = client.post(
            "/api/import/csv",
            data={"file": _make_csv_file(csv_data)},
            content_type="multipart/form-data",
        )

    assert resp.status_code == 200
    data = resp.get_json()
    assert data["rows_imported"] == 1
    assert data["rows_skipped"] == 3

    skipped_by_row = {r["row"]: r["reason"] for r in data["skipped_rows"]}
    assert 3 in skipped_by_row and "sold" in skipped_by_row[3].lower() and "non-negative" in skipped_by_row[3]
    assert 4 in skipped_by_row and "baked" in skipped_by_row[4].lower()
    assert 5 in skipped_by_row and "wasted" in skipped_by_row[5].lower()


def test_import_duplicates_in_file(client, app):
    """When a file contains duplicate (date, item) rows, the first imports and subsequent rows are skipped."""
    csv_data = (
        "date,item,sold\n"
        "2026-03-01,Croissant,20\n"
        "2026-03-01,Croissant,25\n"   # Row 3: duplicate
        "2026-03-02,Croissant,22\n"
    )
    with patch("app.api.routes.sync_weather_for_date_range") as mock_weather:
        mock_weather.return_value = {"status": "ok", "days_fetched": 2}
        resp = client.post(
            "/api/import/csv",
            data={"file": _make_csv_file(csv_data)},
            content_type="multipart/form-data",
        )

    assert resp.status_code == 200
    data = resp.get_json()
    assert data["rows_imported"] == 2
    assert data["rows_skipped"] == 1
    assert data["skipped_rows"][0]["row"] == 3
    assert "Duplicate entry" in data["skipped_rows"][0]["reason"]


def test_import_missing_columns(client):
    """Uploading CSV missing required columns returns 400 Bad Request."""
    csv_data = "date,wasted\n2026-03-01,5\n"
    resp = client.post(
        "/api/import/csv",
        data={"file": _make_csv_file(csv_data)},
        content_type="multipart/form-data",
    )
    assert resp.status_code == 400
    data = resp.get_json()
    assert data["error"]["code"] == "missing_columns"
    assert "item" in data["error"]["message"].lower()


def test_import_custom_column_mapping(client, app):
    """Custom header names map properly when provided in form parameters."""
    csv_data = (
        "jour,produit,ventes\n"
        "2026-03-05,Croissant,42\n"
    )
    with patch("app.api.routes.sync_weather_for_date_range") as mock_weather:
        mock_weather.return_value = {"status": "ok", "days_fetched": 1}
        resp = client.post(
            "/api/import/csv",
            data={
                "file": _make_csv_file(csv_data),
                "col_date": "jour",
                "col_item": "produit",
                "col_sold": "ventes",
            },
            content_type="multipart/form-data",
        )

    assert resp.status_code == 200
    data = resp.get_json()
    assert data["rows_imported"] == 1
    assert data["rows_skipped"] == 0


def test_import_create_missing_items_toggle(client, app):
    """Test create_missing_items checkbox behavior: skip if unchecked, create if checked."""
    csv_data = (
        "date,item,sold\n"
        "2026-03-10,BrandNewPastry,15\n"
    )

    # 1. Unchecked (default): Row is skipped because item doesn't exist
    resp1 = client.post(
        "/api/import/csv",
        data={
            "file": _make_csv_file(csv_data),
            "create_missing_items": "false",
        },
        content_type="multipart/form-data",
    )
    assert resp1.status_code == 200
    d1 = resp1.get_json()
    assert d1["rows_imported"] == 0
    assert d1["rows_skipped"] == 1
    assert "BrandNewPastry" in d1["skipped_rows"][0]["reason"]

    # 2. Checked: Item is created in catalogue and row is imported
    with patch("app.api.routes.sync_weather_for_date_range") as mock_weather:
        mock_weather.return_value = {"status": "ok", "days_fetched": 1}
        resp2 = client.post(
            "/api/import/csv",
            data={
                "file": _make_csv_file(csv_data),
                "create_missing_items": "true",
            },
            content_type="multipart/form-data",
        )
    assert resp2.status_code == 200
    d2 = resp2.get_json()
    assert d2["rows_imported"] == 1
    assert d2["rows_skipped"] == 0
    assert "BrandNewPastry" in d2["created_items"]

    # Verify item exists in database
    with app.app_context():
        from app.db import get_db
        db = get_db()
        item = db.execute("SELECT id, name FROM items WHERE name = 'BrandNewPastry'").fetchone()
        assert item is not None


def test_import_file_size_limit(client):
    """Uploading a file exceeding 2 MB is rejected with 400 Bad Request."""
    # Generate ~2.2 MB CSV
    big_content = "date,item,sold\n" + ("2026-03-01,Croissant,10\n" * 100000)
    assert len(big_content) > 2 * 1024 * 1024

    resp = client.post(
        "/api/import/csv",
        data={"file": _make_csv_file(big_content)},
        content_type="multipart/form-data",
    )
    assert resp.status_code in (400, 413)
    data = resp.get_json()
    assert data["error"]["code"] == "file_too_large"


def test_import_weather_sync_invoked(client):
    """Weather synchronization is automatically triggered for imported date range."""
    csv_data = (
        "date,item,sold\n"
        "2026-01-10,Croissant,15\n"
        "2026-01-20,Croissant,22\n"
    )
    with patch("app.api.routes.sync_weather_for_date_range") as mock_weather:
        mock_weather.return_value = {
            "status": "ok",
            "start_date": "2026-01-10",
            "end_date": "2026-01-20",
            "days_fetched": 11,
        }
        resp = client.post(
            "/api/import/csv",
            data={"file": _make_csv_file(csv_data)},
            content_type="multipart/form-data",
        )

    assert resp.status_code == 200
    data = resp.get_json()
    assert data["weather_synced"]["status"] == "ok"
    assert data["weather_synced"]["days_fetched"] == 11
    mock_weather.assert_called_once()
    args, _ = mock_weather.call_args
    assert args[1] == "2026-01-10"
    assert args[2] == "2026-01-20"


# ---------------------------------------------------------------------------
# Kaggle Converter Script Tests (scripts/load_kaggle.py)
# ---------------------------------------------------------------------------

def test_kaggle_script_french_bakery(tmp_path, capsys):
    """Support French Bakery Daily Sales format with top N items, 'Other', and open-day zero sales."""
    raw_csv = tmp_path / "french_sample.csv"
    raw_csv.write_text(
        "date,time,ticket_number,article,Quantity,unit_price\n"
        "2021-01-02,08:30,101,BAGUETTE,3,0,90 €\n"
        "2021-01-02,08:35,102,CROISSANT,2,1,10 €\n"
        "2021-01-02,09:00,103,PAIN AU CHOCOLAT,1,1,20 €\n"
        "2021-01-03,09:10,104,BAGUETTE,4,0,90 €\n"
        "2021-01-03,09:15,105,SPECIALTY ROLL,1,2,50 €\n",
        encoding="utf-8",
    )

    out_dir = tmp_path / "processed"
    out_file = process_kaggle_file(str(raw_csv), output_dir=str(out_dir), top_n=2)

    captured = capsys.readouterr()
    # Check that real headers were detected and printed
    assert "Real headers detected" in captured.out
    assert "Quantity" in captured.out
    assert os.path.exists(out_file)

    # Read processed output
    with open(out_file, mode="r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    header = lines[0]
    assert header == "date,item,sold"

    rows = [l.split(",") for l in lines[1:]]
    # Check that BAGUETTE and CROISSANT are the top 2 items, others mapped to 'Other'
    items_in_output = set(r[1] for r in rows)
    assert "BAGUETTE" in items_in_output
    assert "CROISSANT" in items_in_output
    assert "Other" in items_in_output

    # Check zero-sale insertion on 2021-01-03 (CROISSANT had 0 sales on this open day)
    croissant_jan3 = [r for r in rows if r[0] == "2021-01-03" and r[1] == "CROISSANT"]
    assert len(croissant_jan3) == 1
    assert croissant_jan3[0][2] == "0"


def test_kaggle_script_bread_basket(tmp_path, capsys):
    """Support The Bread Basket format (row-level transactions, no quantity column)."""
    raw_csv = tmp_path / "breadbasket_sample.csv"
    raw_csv.write_text(
        "Date,Time,Transaction,Item\n"
        "2016-10-30,09:58:11,1,Bread\n"
        "2016-10-30,10:05:34,2,Scandinavian\n"
        "2016-10-30,10:07:57,3,Hot chocolate\n"
        "2016-10-30,10:07:57,3,Jam\n"
        "2016-10-30,10:07:57,3,NONE\n"      # Filtered out placeholder
        "2016-10-31,10:00:00,4,Bread\n",
        encoding="utf-8",
    )

    out_dir = tmp_path / "processed"
    out_file = process_kaggle_file(str(raw_csv), output_dir=str(out_dir), top_n=2)

    captured = capsys.readouterr()
    assert "Real headers detected" in captured.out
    assert "Bread" in captured.out

    with open(out_file, mode="r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    rows = [l.split(",") for l in lines[1:]]
    # NONE should not appear anywhere
    items = [r[1] for r in rows]
    assert "NONE" not in items

    # Bread on 2016-10-30 sold 1
    bread_day1 = [r for r in rows if r[0] == "2016-10-30" and r[1] == "Bread"]
    assert len(bread_day1) == 1
    assert bread_day1[0][2] == "1"


# ---------------------------------------------------------------------------
# Reports View History Test
# ---------------------------------------------------------------------------

def test_reports_view_imported_history(client, app):
    """Importing past historical records allows viewing them via Reports endpoint with ?to=date."""
    # Import historical rows from May 2024
    csv_data = (
        "date,item,sold\n"
        "2024-05-01,Croissant,40\n"
        "2024-05-02,Croissant,45\n"
        "2024-05-03,Croissant,50\n"
    )
    with patch("app.api.routes.sync_weather_for_date_range") as mock_weather:
        mock_weather.return_value = {"status": "ok", "days_fetched": 3}
        resp_imp = client.post(
            "/api/import/csv",
            data={"file": _make_csv_file(csv_data)},
            content_type="multipart/form-data",
        )
    assert resp_imp.status_code == 200

    # Query reports API specifying ?to=2024-05-03&days=7
    resp_rep = client.get("/api/reports/forecast-vs-actual?to=2024-05-03&days=7")
    assert resp_rep.status_code == 200
    rep_data = resp_rep.get_json()

    assert rep_data["summary"]["total_sold"] == 135
    assert rep_data["latest_available_date"] is not None
    # Check that daily series contains the actual sold quantities
    sold_map = {d["date"]: d["actual_sold"] for d in rep_data["daily"]}
    assert sold_map.get("2024-05-01") == 40
    assert sold_map.get("2024-05-02") == 45
    assert sold_map.get("2024-05-03") == 50

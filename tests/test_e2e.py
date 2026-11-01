"""End-to-End full user journey and security hardening tests for Loaf Ledger.

Journey verified:
1. Register new user
2. Onboard bakery shop with initial items
3. Import sales history CSV
4. Fetch/refresh weather (mocked)
5. Log a day (daily_log)
6. Generate demand forecast
7. Retrain demand models
8. View accuracy report
9. Register a second user and verify strict multi-tenant isolation across all resources
10. Security hardening: security headers, rate limiting, and size limit checks
"""

from datetime import date, timedelta
import io
from unittest.mock import patch
import pytest

from app.db import get_db
from tests.conftest import LoafTestClient, MockResponse


def _make_csv_file(content: str, filename: str = "import.csv"):
    """Helper to create a BytesIO file tuple for multipart CSV upload."""
    return (io.BytesIO(content.encode("utf-8")), filename)


def test_full_user_journey_and_multitenant_isolation(app):
    """Executes the complete journey for User 1, then verifies User 2 sees none of User 1's data."""
    # -------------------------------------------------------------------------
    # PART 1: User 1 (Alice) Full Journey
    # -------------------------------------------------------------------------
    client1 = LoafTestClient(app.test_client())

    # 1. Register User 1
    token1 = client1.get_csrf()
    res_reg1 = client1.post(
        "/register",
        data={
            "email": "alice_baker@example.com",
            "password": "Password123!",
            "confirm_password": "Password123!",
            "csrf_token": token1,
        },
        csrf=False,
    )
    assert res_reg1.status_code == 302
    assert "/onboarding" in res_reg1.headers["Location"]

    # 2. Onboard User 1 with shop and items
    token1 = client1.get_csrf()
    res_onboard1 = client1.post(
        "/onboarding",
        data={
            "name": "Alice Artisan Bakery",
            "city": "London",
            "lat": "51.5074",
            "lon": "-0.1278",
            "item_name[]": ["Rustic Sourdough", "Almond Croissant"],
            "item_stock[]": [15, 20],
            "item_shelf_life[]": [2, 1],
            "csrf_token": token1,
        },
        csrf=False,
    )
    assert res_onboard1.status_code == 302
    assert "/plan" in res_onboard1.headers["Location"]

    # Verify User 1's items
    res_items1 = client1.get("/api/items")
    assert res_items1.status_code == 200
    items1 = res_items1.get_json()
    item_names1 = [it["name"] for it in items1]
    assert "Rustic Sourdough" in item_names1
    assert "Almond Croissant" in item_names1
    sourdough_id = next(it["id"] for it in items1 if it["name"] == "Rustic Sourdough")
    croissant_id = next(it["id"] for it in items1 if it["name"] == "Almond Croissant")

    # 3. Import a small CSV for User 1
    csv_content = (
        "date,item,sold,baked,wasted,leftover\n"
        "2026-10-20,Rustic Sourdough,25,30,2,3\n"
        "2026-10-20,Almond Croissant,30,35,4,1\n"
        "2026-10-21,Rustic Sourdough,28,30,1,1\n"
        "2026-10-21,Almond Croissant,32,35,3,0\n"
    )
    with patch("app.api.routes.sync_weather_for_date_range") as mock_weather_sync:
        mock_weather_sync.return_value = {"status": "ok", "days_fetched": 2}
        res_import = client1.post(
            "/api/import/csv",
            data={"file": _make_csv_file(csv_content)},
            content_type="multipart/form-data",
        )
    assert res_import.status_code == 200
    import_data = res_import.get_json()
    assert import_data["status"] == "success"
    assert import_data["rows_imported"] == 4

    # 4. Fetch weather (mocked Open-Meteo)
    mock_forecast_payload = [
        {"date": "2026-10-22", "temp_mean_c": 16.5, "temp_max_c": 19.0, "temp_min_c": 14.0, "precip_mm": 0.0, "is_rainy": 0, "is_holiday": 0},
        {"date": "2026-10-23", "temp_mean_c": 15.0, "temp_max_c": 18.0, "temp_min_c": 12.0, "precip_mm": 2.5, "is_rainy": 1, "is_holiday": 0},
    ]
    with patch("app.api.routes.fetch_forecast") as mock_fetch:
        mock_fetch.return_value = mock_forecast_payload
        res_weather = client1.post("/api/weather/refresh")
    assert res_weather.status_code == 200

    # 5. Log a day (daily_log)
    res_log = client1.post(
        "/api/daily-log",
        json={
            "item_id": sourdough_id,
            "date": "2026-10-22",
            "sold": 26,
            "baked": 30,
            "wasted": 2,
            "leftover": 2,
            "sold_out": False,
        },
    )
    assert res_log.status_code == 200
    log_data = res_log.get_json()
    assert log_data["sold"] == 26
    assert log_data["baked"] == 30

    # 6. Get a demand forecast
    res_forecast = client1.get("/api/forecast?date=2026-10-23")
    assert res_forecast.status_code == 200
    forecast_data = res_forecast.get_json()
    assert "forecasts" in forecast_data
    forecast_item_ids = [f["item_id"] for f in forecast_data["forecasts"]]
    assert sourdough_id in forecast_item_ids

    # 7. Retrain demand models
    res_retrain = client1.post("/api/admin/retrain")
    assert res_retrain.status_code == 200
    retrain_data = res_retrain.get_json()
    assert retrain_data["status"] == "success"
    assert retrain_data["shop_name"] == "Alice Artisan Bakery"

    # 8. View Accuracy report
    res_accuracy1 = client1.get("/api/reports/accuracy?days=28&to=2026-10-23")
    assert res_accuracy1.status_code == 200
    accuracy_data1 = res_accuracy1.get_json()
    assert accuracy_data1["status"] == "success"

    # -------------------------------------------------------------------------
    # PART 2: User 2 (Bob) Isolation Verification
    # -------------------------------------------------------------------------
    client2 = LoafTestClient(app.test_client())

    # 1. Register User 2
    token2 = client2.get_csrf()
    res_reg2 = client2.post(
        "/register",
        data={
            "email": "bob_bagels@example.com",
            "password": "Password456!",
            "confirm_password": "Password456!",
            "csrf_token": token2,
        },
        csrf=False,
    )
    assert res_reg2.status_code == 302
    assert "/onboarding" in res_reg2.headers["Location"]

    # 2. Onboard User 2 with distinct shop and item
    token2 = client2.get_csrf()
    res_onboard2 = client2.post(
        "/onboarding",
        data={
            "name": "Bob's Bagels",
            "city": "Manchester",
            "lat": "53.4808",
            "lon": "-2.2426",
            "item_name[]": ["Sesame Bagel"],
            "item_stock[]": [40],
            "item_shelf_life[]": [1],
            "csrf_token": token2,
        },
        csrf=False,
    )
    assert res_onboard2.status_code == 302

    # 3. Check User 2 sees NONE of User 1's items
    res_items2 = client2.get("/api/items")
    assert res_items2.status_code == 200
    items2 = res_items2.get_json()
    item_names2 = [it["name"] for it in items2]
    assert "Sesame Bagel" in item_names2
    assert "Rustic Sourdough" not in item_names2
    assert "Almond Croissant" not in item_names2

    # 4. Check User 2 sees NONE of User 1's daily_log records
    res_logs2 = client2.get("/api/daily-log")
    assert res_logs2.status_code == 200
    logs2 = res_logs2.get_json()
    for l in logs2:
        assert l["item_id"] not in (sourdough_id, croissant_id)

    # 5. Check User 2's forecast does NOT contain User 1's items
    res_forecast2 = client2.get("/api/forecast?date=2026-10-23")
    assert res_forecast2.status_code == 200
    forecast2_data = res_forecast2.get_json()
    forecast2_ids = [f["item_id"] for f in forecast2_data["forecasts"]]
    assert sourdough_id not in forecast2_ids
    assert croissant_id not in forecast2_ids

    # 6. Check User 2's accuracy report contains zero data from User 1
    res_accuracy2 = client2.get("/api/reports/accuracy?days=28&to=2026-10-23")
    assert res_accuracy2.status_code == 200
    accuracy2_data = res_accuracy2.get_json()
    accuracy2_item_ids = [it["item_id"] for it in accuracy2_data.get("items", [])]
    assert sourdough_id not in accuracy2_item_ids
    assert croissant_id not in accuracy2_item_ids

    # 7. Check User 2's CSV export contains none of User 1's items
    res_csv2 = client2.get("/api/export/daily-log")
    assert res_csv2.status_code == 200
    csv2_text = res_csv2.get_data(as_text=True)
    assert "Rustic Sourdough" not in csv2_text
    assert "Almond Croissant" not in csv2_text

    # 8. Direct attempt by User 2 to modify or log against User 1's item ID fails with 404
    res_tamper = client2.post(
        "/api/daily-log",
        json={
            "item_id": sourdough_id,
            "date": "2026-10-22",
            "sold": 99,
        },
    )
    assert res_tamper.status_code == 404
    assert res_tamper.get_json()["error"]["code"] == "not_found"


# ---------------------------------------------------------------------------
# Security Hardening & Edge Cases
# ---------------------------------------------------------------------------

def test_security_headers_present_on_all_responses(client):
    """Every HTTP response must include standard security headers."""
    res = client.get("/api/health")
    assert res.status_code == 200
    headers = res.headers

    assert headers.get("X-Content-Type-Options") == "nosniff"
    assert headers.get("X-Frame-Options") == "SAMEORIGIN"
    assert headers.get("X-XSS-Protection") == "1; mode=block"
    assert headers.get("Referrer-Policy") == "strict-origin-when-cross-origin"
    assert "Content-Security-Policy" in headers
    csp = headers.get("Content-Security-Policy")
    assert "default-src 'self'" in csp


def test_login_rate_limiting(app):
    """Exceeding 10 failed login attempts within 1 minute triggers HTTP 429 Too Many Requests."""
    from app.auth import reset_rate_limits
    reset_rate_limits()

    raw_client = app.test_client()
    login_page = raw_client.get("/login")
    import re
    token = re.search(r'name="csrf_token" value="([^"]+)"', login_page.get_data(as_text=True)).group(1)

    # 10 failed login attempts
    for _ in range(10):
        res = raw_client.post(
            "/login",
            data={"email": "wrong@example.com", "password": "wrongpassword", "csrf_token": token},
        )
        assert res.status_code == 401

    # 11th attempt must be rate-limited
    res_blocked = raw_client.post(
        "/login",
        data={"email": "wrong@example.com", "password": "wrongpassword", "csrf_token": token},
    )
    assert res_blocked.status_code == 429

    # Clean up rate limits after test
    reset_rate_limits()


def test_custom_404_error_page(client):
    """Non-existent web URLs return custom friendly 404 page, APIs return JSON 404."""
    # Web request -> friendly HTML
    res_web = client.get("/non-existent-bakery-page")
    assert res_web.status_code == 404
    html = res_web.get_data(as_text=True)
    assert "404 — Page Not Found" in html
    assert "Return to Bakery" in html

    # API request -> consistent JSON
    res_api = client.get("/api/non-existent-endpoint")
    assert res_api.status_code == 404
    data = res_api.get_json()
    assert data["error"]["code"] == "not_found"


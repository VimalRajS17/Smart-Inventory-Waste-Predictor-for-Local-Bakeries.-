"""Tests for authentication, CSRF protection, and multi-tenant scoping (Phase 3A)."""

import json
from datetime import date
from flask_login import current_user
from app.db import get_db


def test_public_health_check_no_auth(unauth_client):
    """GET /api/health is public and returns 200 without authentication."""
    res = unauth_client.get("/api/health")
    assert res.status_code == 200
    data = res.get_json()
    assert data["status"] == "ok"
    assert data["database"] == "connected"


def test_unauthenticated_api_returns_401(unauth_client):
    """Unauthenticated requests to protected /api endpoints return 401 Unauthorized."""
    endpoints = [
        ("GET", "/api/items"),
        ("POST", "/api/items"),
        ("GET", "/api/daily-log"),
        ("POST", "/api/daily-log"),
        ("GET", "/api/weather?date=2026-10-31"),
        ("POST", "/api/weather/refresh"),
        ("GET", "/api/forecast?date=2026-10-31"),
        ("GET", "/api/reports/forecast-vs-actual?days=7"),
    ]

    for method, path in endpoints:
        if method == "GET":
            res = unauth_client.get(path)
        else:
            # Send with valid CSRF token to test authentication barrier specifically
            res = unauth_client.post(path, json={}, csrf=True)

        assert res.status_code == 401, f"{method} {path} should return 401, got {res.status_code}"
        data = res.get_json()
        assert data["error"]["code"] == "unauthorized"
        assert "Authentication required" in data["error"]["message"]


def test_unauthenticated_pages_redirect_to_login(unauth_client):
    """Unauthenticated page requests redirect to /login with next query parameter."""
    pages = ["/", "/plan", "/today", "/reports", "/items", "/onboarding"]
    for path in pages:
        res = unauth_client.get(path)
        assert res.status_code == 302, f"GET {path} should redirect, got {res.status_code}"
        assert "/login" in res.headers["Location"]


def test_wrong_password_fails_login(unauth_client):
    """POST /login with an invalid password returns 401 and does not authenticate."""
    token = unauth_client.get_csrf()

    # Form login with wrong password
    res = unauth_client.post(
        "/login",
        data={"email": "demo@example.com", "password": "wrong_password", "csrf_token": token},
        csrf=False,
    )
    assert res.status_code == 401
    html = res.get_data(as_text=True)
    assert "Invalid email or password" in html

    # Verify still unauthenticated
    res_items = unauth_client.get("/api/items")
    assert res_items.status_code == 401


def test_csrf_protection_enforcement(app):
    """POST/PUT/DELETE requests fail with 400 when CSRF token is missing or invalid."""
    # Create authenticated client without auto-csrf injection
    client = app.test_client()

    # 1. Obtain session by logging in
    login_page = client.get("/login")
    import re
    token = re.search(r'name="csrf_token" value="([^"]+)"', login_page.get_data(as_text=True)).group(1)

    res_login = client.post(
        "/login",
        data={"email": "demo@example.com", "password": "demo123", "csrf_token": token},
    )
    assert res_login.status_code == 302

    # 2. POST without X-CSRFToken -> 400
    res_no_csrf = client.post(
        "/api/items",
        json={"name": "No CSRF Bread", "current_stock": 5, "shelf_life_days": 1},
    )
    assert res_no_csrf.status_code == 400
    err = res_no_csrf.get_json()["error"]
    assert err["code"] == "csrf_error"

    # 3. POST with invalid X-CSRFToken -> 400
    res_bad_csrf = client.post(
        "/api/items",
        json={"name": "Bad CSRF Bread", "current_stock": 5, "shelf_life_days": 1},
        headers={"X-CSRFToken": "invalid-token-12345"},
    )
    assert res_bad_csrf.status_code == 400
    err2 = res_bad_csrf.get_json()["error"]
    assert err2["code"] == "csrf_error"

    # 4. POST with valid X-CSRFToken -> 201
    res_valid = client.post(
        "/api/items",
        json={"name": "Valid CSRF Bread", "current_stock": 5, "shelf_life_days": 1},
        headers={"X-CSRFToken": token},
    )
    assert res_valid.status_code == 201
    assert res_valid.get_json()["name"] == "Valid CSRF Bread"


def test_session_cookie_flags(unauth_client):
    """Verify session cookies have HttpOnly and SameSite=Lax flags."""
    token = unauth_client.get_csrf()
    res = unauth_client.post(
        "/login",
        data={"email": "demo@example.com", "password": "demo123", "csrf_token": token},
        csrf=False,
    )
    assert res.status_code == 302

    cookies = res.headers.getlist("Set-Cookie")
    session_cookie = next((c for c in cookies if "session=" in c), None)
    assert session_cookie is not None
    assert "HttpOnly" in session_cookie
    assert "SameSite=Lax" in session_cookie


def test_registration_and_onboarding_flow(app):
    """New user registers, is redirected to onboarding, configures shop and items, and reaches dashboard."""
    from tests.conftest import LoafTestClient
    c = LoafTestClient(app.test_client())

    # 1. Registration
    token = c.get_csrf()
    res_reg = c.post(
        "/register",
        data={
            "email": "baker_new@example.com",
            "password": "securepassword123",
            "confirm_password": "securepassword123",
            "csrf_token": token,
        },
        csrf=False,
    )
    assert res_reg.status_code == 302
    assert "/onboarding" in res_reg.headers["Location"]

    # 2. View Onboarding page
    res_onboard_get = c.get("/onboarding")
    assert res_onboard_get.status_code == 200
    assert "Welcome to Loaf Ledger!" in res_onboard_get.get_data(as_text=True)

    # 3. Submit Onboarding form
    token = c.get_csrf()
    res_onboard_post = c.post(
        "/onboarding",
        data={
            "name": "Sunrise Artisan Bakery",
            "city": "Bristol",
            "lat": "51.4545",
            "lon": "-2.5879",
            "item_name[]": ["Cardamom Bun", "Focaccia"],
            "item_stock[]": [12, 8],
            "item_shelf_life[]": [1, 2],
            "csrf_token": token,
        },
        csrf=False,
    )
    assert res_onboard_post.status_code == 302
    assert "/plan" in res_onboard_post.headers["Location"]

    # 4. View Items for newly onboarded shop
    res_items = c.get("/api/items")
    assert res_items.status_code == 200
    items = res_items.get_json()
    item_names = [it["name"] for it in items]
    assert "Cardamom Bun" in item_names
    assert "Focaccia" in item_names
    # Demo items should NOT be present
    assert "Croissant" not in item_names


def test_multi_tenant_isolation(app):
    """User A and User B can register, onboard separate shops, and never access each other's data."""
    from tests.conftest import LoafTestClient

    # Setup User A
    client_a = LoafTestClient(app.test_client())
    token_a = client_a.get_csrf()
    client_a.post(
        "/register",
        data={"email": "alice@bakery.com", "password": "password123", "confirm_password": "password123", "csrf_token": token_a},
        csrf=False,
    )
    token_a = client_a.get_csrf()
    client_a.post(
        "/onboarding",
        data={
            "name": "Alice Bakery",
            "city": "Oxford",
            "lat": "51.7520",
            "lon": "-1.2577",
            "item_name[]": ["Alice Sourdough"],
            "item_stock[]": [25],
            "item_shelf_life[]": [2],
            "csrf_token": token_a,
        },
        csrf=False,
    )

    # Setup User B
    client_b = LoafTestClient(app.test_client())
    token_b = client_b.get_csrf()
    client_b.post(
        "/register",
        data={"email": "bob@bakery.com", "password": "password123", "confirm_password": "password123", "csrf_token": token_b},
        csrf=False,
    )
    token_b = client_b.get_csrf()
    client_b.post(
        "/onboarding",
        data={
            "name": "Bob Patisserie",
            "city": "Cambridge",
            "lat": "52.2053",
            "lon": "0.1218",
            "item_name[]": ["Bob Brioche"],
            "item_stock[]": [18],
            "item_shelf_life[]": [1],
            "csrf_token": token_b,
        },
        csrf=False,
    )

    # 1. Item Listing Isolation
    items_a = client_a.get("/api/items").get_json()
    items_b = client_b.get("/api/items").get_json()

    assert len(items_a) == 1
    assert items_a[0]["name"] == "Alice Sourdough"
    item_a_id = items_a[0]["id"]

    assert len(items_b) == 1
    assert items_b[0]["name"] == "Bob Brioche"
    item_b_id = items_b[0]["id"]

    # 2. Cross-tenant Item Modification Prevention
    # User A tries to edit User B's item
    res_edit = client_a.put(f"/api/items/{item_b_id}", json={"name": "Hacked Brioche"})
    assert res_edit.status_code == 404
    assert res_edit.get_json()["error"]["code"] == "not_found"

    # User B's item remains unchanged
    item_b_check = client_b.get("/api/items").get_json()[0]
    assert item_b_check["name"] == "Bob Brioche"

    # User A tries to delete User B's item
    res_del = client_a.delete(f"/api/items/{item_b_id}")
    assert res_del.status_code == 404
    assert res_del.get_json()["error"]["code"] == "not_found"

    # 3. Cross-tenant Daily Log Isolation
    today_str = date.today().isoformat()

    # User A records daily log for Item A
    res_log_a = client_a.post("/api/daily-log", json={
        "item_id": item_a_id,
        "date": today_str,
        "sold": 20,
        "baked": 25,
        "wasted": 2,
    })
    assert res_log_a.status_code == 200

    # User A tries to record daily log for User B's item -> 404
    res_cross_log = client_a.post("/api/daily-log", json={
        "item_id": item_b_id,
        "date": today_str,
        "sold": 10,
    })
    assert res_cross_log.status_code == 404

    # User B cannot see User A's daily logs
    logs_b = client_b.get("/api/daily-log").get_json()
    assert len(logs_b) == 0

    # User A sees only their own daily logs
    logs_a = client_a.get("/api/daily-log").get_json()
    assert len(logs_a) == 1
    assert logs_a[0]["item_id"] == item_a_id

    # 4. Cross-tenant Forecast Scoping
    forecast_a = client_a.get("/api/forecast").get_json()["forecasts"]
    forecast_b = client_b.get("/api/forecast").get_json()["forecasts"]

    assert len(forecast_a) == 1
    assert forecast_a[0]["name"] == "Alice Sourdough"

    assert len(forecast_b) == 1
    assert forecast_b[0]["name"] == "Bob Brioche"

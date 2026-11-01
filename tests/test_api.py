"""Tests for REST API endpoints and input validation constraints."""

from datetime import date
from tests.conftest import MockResponse


def assert_error_format(res, expected_status: int, expected_code: str = None):
    """Assert response follows standard error shape: {"error": {"code": "...", "message": "..."}}."""
    assert res.status_code == expected_status
    body = res.get_json()
    assert isinstance(body, dict)
    assert "error" in body
    assert "code" in body["error"]
    assert "message" in body["error"]
    if expected_code:
        assert body["error"]["code"] == expected_code


def test_api_health(client):
    """GET /api/health returns 200 ok and database connectivity status."""
    res = client.get("/api/health")
    assert res.status_code == 200
    data = res.get_json()
    assert data["status"] == "ok"
    assert data["database"] == "connected"
    assert "scheduler" in data
    assert "rl_enabled" in data
    assert "timestamp" in data


def test_api_items_crud_and_validation(client):
    """Test items CRUD operations and input validations."""
    # 1. GET /api/items
    res = client.get("/api/items")
    assert res.status_code == 200
    items = res.get_json()
    assert len(items) >= 5

    # 2. POST /api/items validation
    # Duplicate name
    res_dup = client.post("/api/items", json={"name": "Croissant"})
    assert_error_format(res_dup, 400, "duplicate_item")

    # Negative stock
    res_neg = client.post("/api/items", json={"name": "Brioche", "current_stock": -2})
    assert_error_format(res_neg, 400, "invalid_input")

    # Invalid JSON
    res_raw = client.post("/api/items", data="not json", content_type="application/json")
    assert_error_format(res_raw, 400, "invalid_json")

    # Empty name
    res_empty = client.post("/api/items", json={"name": "   "})
    assert_error_format(res_empty, 400, "invalid_input")

    # Successful creation
    res_ok = client.post("/api/items", json={"name": "Brioche Bun", "current_stock": 15, "shelf_life_days": 2})
    assert res_ok.status_code == 201
    created = res_ok.get_json()
    assert created["name"] == "Brioche Bun"
    assert created["current_stock"] == 15
    item_id = created["id"]

    # 3. PUT /api/items/<id>
    # 404 on nonexistent item
    res_put_404 = client.put("/api/items/9999", json={"name": "Ghost"})
    assert_error_format(res_put_404, 404, "not_found")

    # Negative stock update
    res_put_neg = client.put(f"/api/items/{item_id}", json={"current_stock": -10})
    assert_error_format(res_put_neg, 400, "invalid_input")

    # Valid update
    res_put = client.put(f"/api/items/{item_id}", json={"current_stock": 25})
    assert res_put.status_code == 200
    assert res_put.get_json()["current_stock"] == 25

    # 4. DELETE /api/items/<id>
    res_del = client.delete(f"/api/items/{item_id}")
    assert res_del.status_code == 200
    assert res_del.get_json() == {"status": "deleted", "id": item_id}

    # Second delete returns 404
    res_del_404 = client.delete(f"/api/items/{item_id}")
    assert_error_format(res_del_404, 404, "not_found")


def test_api_daily_log_validation(client):
    """Test POST /api/daily-log validation and upsert."""
    # Missing item
    res_item_404 = client.post("/api/daily-log", json={"item_id": 9999, "date": "2026-10-30", "sold": 5})
    assert_error_format(res_item_404, 404, "not_found")

    # Get a valid item ID
    items = client.get("/api/items").get_json()
    valid_id = items[0]["id"]

    # Invalid date format
    res_date_err = client.post("/api/daily-log", json={"item_id": valid_id, "date": "10-30-2026", "sold": 5})
    assert_error_format(res_date_err, 400, "invalid_input")

    # Negative sold quantity
    res_sold_err = client.post("/api/daily-log", json={"item_id": valid_id, "date": "2026-10-30", "sold": -1})
    assert_error_format(res_sold_err, 400, "invalid_input")

    # Valid upsert
    res_ok = client.post("/api/daily-log", json={
        "item_id": valid_id,
        "date": "2026-10-30",
        "sold": 18,
        "baked": 20,
        "wasted": 2,
        "sold_out": True,
    })
    assert res_ok.status_code == 200
    data = res_ok.get_json()
    assert data["sold"] == 18
    assert data["baked"] == 20
    assert data["wasted"] == 2
    assert data["sold_out"] == 1


def test_api_daily_log_query(client):
    """Test GET /api/daily-log with date range and item filtering."""
    items = client.get("/api/items").get_json()
    item_id = items[0]["id"]

    # Create logs
    client.post("/api/daily-log", json={"item_id": item_id, "date": "2026-10-20", "sold": 10})
    client.post("/api/daily-log", json={"item_id": item_id, "date": "2026-10-21", "sold": 15})

    # Query with valid range
    res = client.get(f"/api/daily-log?from=2026-10-20&to=2026-10-21&item_id={item_id}")
    assert res.status_code == 200
    rows = res.get_json()
    assert len(rows) == 2

    # Query with inverted date range
    res_err = client.get("/api/daily-log?from=2026-10-25&to=2026-10-20")
    assert_error_format(res_err, 400, "invalid_input")


def test_api_weather_endpoints(client, monkeypatch):
    """Test GET /api/weather and POST /api/weather/refresh."""
    # Invalid date format
    res_err = client.get("/api/weather?date=bad-date")
    assert_error_format(res_err, 400, "invalid_input")

    # Mock weather API for refresh
    mock_data = {
        "daily": {
            "time": ["2026-10-30"],
            "temperature_2m_max": [16.0],
            "temperature_2m_min": [8.0],
            "temperature_2m_mean": [12.0],
            "precipitation_sum": [0.0],
        }
    }
    monkeypatch.setattr("requests.get", lambda url, params=None, timeout=None: MockResponse(mock_data))

    # Refresh weather
    res_ref = client.post("/api/weather/refresh")
    assert res_ref.status_code == 200
    assert res_ref.get_json()["status"] == "ok"

    # Query refreshed weather
    res_w = client.get("/api/weather?date=2026-10-30")
    assert res_w.status_code == 200
    w_data = res_w.get_json()
    assert w_data["date"] == "2026-10-30"
    assert w_data["temp_mean_c"] == 12.0


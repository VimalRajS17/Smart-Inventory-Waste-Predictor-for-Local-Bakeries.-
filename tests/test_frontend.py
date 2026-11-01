"""Tests for frontend HTML template rendering across all routes and end-to-end forecast reactivity."""

from datetime import date, timedelta
from app.db import get_db


def test_plan_page_renders(client):
    """GET / and GET /plan render base.html with plan.html child template."""
    for path in ["/", "/plan"]:
        res = client.get(path)
        assert res.status_code == 200
        html = res.get_data(as_text=True)

        # Base elements: header, brand, weather widget, theme toggle, nav tabs
        assert "Loaf Ledger" in html
        assert 'id="weather-widget"' in html
        assert 'id="theme-toggle-btn"' in html
        assert 'Plan' in html
        assert 'Today' in html
        assert 'Reports' in html
        assert 'Items' in html

        # Plan specific elements
        assert 'id="plan-date-input"' in html
        assert 'id="plan-tomorrow-btn"' in html
        assert 'id="plan-refresh-btn"' in html
        assert 'id="plan-container"' in html

        # Shared JS & CSS
        assert "api.js" in html
        assert "style.css" in html


def test_today_page_renders_with_touch_targets(client):
    """GET /today renders the Today production log form with touch-friendly controls."""
    res = client.get("/today")
    assert res.status_code == 200
    html = res.get_data(as_text=True)

    # Base layout
    assert "Loaf Ledger" in html
    assert 'id="weather-widget"' in html

    # Form and big touch target inputs
    assert 'id="today-log-form"' in html
    assert 'id="today-item-select"' in html
    assert 'id="today-date-input"' in html
    assert 'id="today-sold-input"' in html
    assert 'id="today-baked-input"' in html
    assert 'id="today-wasted-input"' in html
    assert 'id="today-sold-out-checkbox"' in html
    assert 'id="today-submit-btn"' in html

    # Recent logs table
    assert 'id="recent-logs-container"' in html
    assert 'id="refresh-logs-btn"' in html


def test_reports_page_renders(client):
    """GET /reports renders the reports and accuracy screen with item selector."""
    res = client.get("/reports")
    assert res.status_code == 200
    html = res.get_data(as_text=True)

    assert "Loaf Ledger" in html
    assert 'id="report-item-select"' in html
    assert 'for="report-item-select"' in html
    assert 'id="report-period-select"' in html
    assert 'for="report-period-select"' in html
    assert 'id="reports-kpis-container"' in html
    assert 'id="reports-chart-wrapper"' in html
    assert 'id="reports-table-container"' in html


def test_items_page_renders(client):
    """GET /items renders the bakery products inventory screen with inline validation error spans."""
    res = client.get("/items")
    assert res.status_code == 200
    html = res.get_data(as_text=True)

    assert "Loaf Ledger" in html
    assert 'id="items-container"' in html
    assert 'id="add-item-btn"' in html
    assert 'id="item-modal"' in html

    # Validation message containers next to fields
    assert 'id="item-name-error"' in html
    assert 'id="item-stock-error"' in html
    assert 'id="item-shelf-life-error"' in html

    # Accessible labels
    assert 'for="item-name-input"' in html
    assert 'for="item-stock-input"' in html
    assert 'for="item-shelf-life-input"' in html


def test_polish_css_rules(client):
    """Verify stylesheet contains focus-visible styles, 44px touch targets, and 360px overflow rules."""
    res = client.get("/static/style.css")
    assert res.status_code == 200
    css = res.get_data(as_text=True)

    # 1. Keyboard focus-visible
    assert ":focus-visible" in css

    # 2. Touch targets >= 44px on mobile
    assert "44px" in css
    assert "min-height: 44px" in css

    # 3. 360px width overflow-x hidden
    assert "overflow-x: hidden" in css
    assert "max-width: 100vw" in css



def test_log_day_changes_next_forecast(client, app):
    """End-to-end test: Logging a day on the Today screen causes the Plan screen's forecast to update after refresh."""
    today_str = date.today().isoformat()
    tomorrow_str = (date.today() + timedelta(days=1)).isoformat()

    with app.app_context():
        db = get_db()
        item = db.execute("SELECT id, name FROM items WHERE name = 'Croissant'").fetchone()
        item_id = item["id"]

        # Ensure no log exists yet for today
        db.execute("DELETE FROM daily_log WHERE item_id = ? AND date = ?", (item_id, today_str))
        db.commit()

    # 1. Fetch baseline forecast for tomorrow (what Plan screen initially loads)
    res_before = client.get(f"/api/forecast?date={tomorrow_str}")
    assert res_before.status_code == 200
    forecasts_before = res_before.get_json()["forecasts"]
    item_before = next(f for f in forecasts_before if f["item_id"] == item_id)
    pred_before = item_before["predicted_qty"]

    # 2. User logs high sales on the Today screen (POST /api/daily-log)
    res_log = client.post("/api/daily-log", json={
        "item_id": item_id,
        "date": today_str,
        "sold": 80,
        "baked": 85,
        "wasted": 5,
        "sold_out": True,
    })
    assert res_log.status_code == 200
    log_data = res_log.get_json()
    assert log_data["sold"] == 80
    assert log_data["sold_out"] == 1

    # 3. User refreshes or views the Plan screen (GET /api/forecast?date=<tomorrow>)
    res_after = client.get(f"/api/forecast?date={tomorrow_str}")
    assert res_after.status_code == 200
    forecasts_after = res_after.get_json()["forecasts"]
    item_after = next(f for f in forecasts_after if f["item_id"] == item_id)
    pred_after = item_after["predicted_qty"]

    # 4. Verify that tomorrow's forecast reflects the new sale and is fresh
    assert item_after["lag_features_stale"] is False
    assert pred_after >= pred_before
    # Suggested bake quantity reflects the prediction
    assert "suggested_bake_qty" in item_after
    assert "reason" in item_after

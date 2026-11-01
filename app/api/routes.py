"""Route definitions for the Loaf Ledger /api blueprint."""

import csv
from datetime import date, datetime, timedelta, timezone
import io
from flask import Response, jsonify, request
from flask_login import current_user, login_required

from app.api import api_bp, error_response
from app.api.validators import parse_iso_date, parse_non_negative_int
from app.db import get_db
from app.services.weather import fetch_forecast, get_weather_for_date, save_weather, sync_weather_for_date_range
from ml.predictor import predict_all_items


def get_current_user_shop():
    """Retrieve current logged in user's shop, or return None."""
    if not current_user.is_authenticated:
        return None
    return current_user.get_shop()


# ---------------------------------------------------------------------------
# Health Check (Public - No Login Required)
# ---------------------------------------------------------------------------

@api_bp.route("/health", methods=["GET"])
def health_check():
    """GET /api/health -> Return API and database connectivity status."""
    from app.scheduler import get_scheduler_status
    from flask import current_app

    health_data = {
        "status": "ok",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "database": "unknown",
        "scheduler": get_scheduler_status(),
        "rl_enabled": bool(current_app.config.get("RL_ENABLED", False)),
    }

    try:
        db = get_db()
        db.execute("SELECT 1;").fetchone()
        health_data["database"] = "connected"
        return jsonify(health_data), 200
    except Exception as exc:
        health_data["status"] = "degraded"
        health_data["database"] = "disconnected"
        health_data["error"] = str(exc)
        return jsonify(health_data), 503


# ---------------------------------------------------------------------------
# Items Endpoints
# ---------------------------------------------------------------------------

@api_bp.route("/items", methods=["GET"])
@login_required
def list_items():
    """GET /api/items -> List all bakery items for current user's shop."""
    shop = get_current_user_shop()
    if not shop:
        return error_response("no_shop", "No bakery shop found for current user. Please complete onboarding.", 400)
    db = get_db()
    rows = db.execute(
        "SELECT id, shop_id, name, current_stock, shelf_life_days FROM items WHERE shop_id = ? ORDER BY id ASC",
        (shop["id"],),
    ).fetchall()
    return jsonify([dict(r) for r in rows])


@api_bp.route("/items", methods=["POST"])
@login_required
def create_item():
    """POST /api/items -> Create a new bakery item for current user's shop."""
    shop = get_current_user_shop()
    if not shop:
        return error_response("no_shop", "No bakery shop found for current user. Please complete onboarding.", 400)
    shop_id = shop["id"]

    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return error_response("invalid_json", "Request body must be a valid JSON object.", 400)

    name = data.get("name")
    if not name or not isinstance(name, str) or not name.strip():
        return error_response("invalid_input", "Field 'name' is required and must be a non-empty string.", 400)
    name = name.strip()

    stock, err = parse_non_negative_int(data.get("current_stock", 0), "current_stock")
    if err:
        return error_response("invalid_input", err, 400)

    shelf_life, err = parse_non_negative_int(data.get("shelf_life_days", 1), "shelf_life_days")
    if err:
        return error_response("invalid_input", err, 400)
    if shelf_life == 0:
        return error_response("invalid_input", "Field 'shelf_life_days' must be at least 1.", 400)

    db = get_db()

    # Check for duplicate within user's shop
    existing = db.execute("SELECT id FROM items WHERE shop_id = ? AND name = ?", (shop_id, name)).fetchone()
    if existing:
        return error_response("duplicate_item", f"Item '{name}' already exists.", 400)

    cursor = db.execute(
        "INSERT INTO items (shop_id, name, current_stock, shelf_life_days) VALUES (?, ?, ?, ?)",
        (shop_id, name, stock, shelf_life),
    )
    db.commit()
    new_id = cursor.lastrowid
    created = db.execute(
        "SELECT id, shop_id, name, current_stock, shelf_life_days FROM items WHERE id = ? AND shop_id = ?",
        (new_id, shop_id),
    ).fetchone()
    return jsonify(dict(created)), 201


@api_bp.route("/items/<int:item_id>", methods=["PUT"])
@login_required
def update_item(item_id: int):
    """PUT /api/items/<id> -> Update an existing bakery item in current user's shop."""
    shop = get_current_user_shop()
    if not shop:
        return error_response("no_shop", "No bakery shop found for current user. Please complete onboarding.", 400)
    shop_id = shop["id"]

    db = get_db()
    item = db.execute(
        "SELECT id, shop_id, name, current_stock, shelf_life_days FROM items WHERE id = ? AND shop_id = ?",
        (item_id, shop_id),
    ).fetchone()
    if not item:
        return error_response("not_found", f"Item with id {item_id} does not exist.", 404)

    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return error_response("invalid_json", "Request body must be a valid JSON object.", 400)

    new_name = item["name"]
    if "name" in data:
        n = data["name"]
        if not n or not isinstance(n, str) or not n.strip():
            return error_response("invalid_input", "Field 'name' must be a non-empty string.", 400)
        new_name = n.strip()
        # Verify name unique among other items in user's shop
        dup = db.execute("SELECT id FROM items WHERE shop_id = ? AND name = ? AND id != ?", (shop_id, new_name, item_id)).fetchone()
        if dup:
            return error_response("duplicate_item", f"Another item with name '{new_name}' already exists.", 400)

    new_stock = item["current_stock"]
    if "current_stock" in data:
        st, err = parse_non_negative_int(data["current_stock"], "current_stock")
        if err:
            return error_response("invalid_input", err, 400)
        new_stock = st

    new_shelf_life = item["shelf_life_days"]
    if "shelf_life_days" in data:
        sl, err = parse_non_negative_int(data["shelf_life_days"], "shelf_life_days")
        if err:
            return error_response("invalid_input", err, 400)
        if sl == 0:
            return error_response("invalid_input", "Field 'shelf_life_days' must be at least 1.", 400)
        new_shelf_life = sl

    db.execute(
        "UPDATE items SET name = ?, current_stock = ?, shelf_life_days = ? WHERE id = ? AND shop_id = ?",
        (new_name, new_stock, new_shelf_life, item_id, shop_id),
    )
    db.commit()
    updated = db.execute(
        "SELECT id, shop_id, name, current_stock, shelf_life_days FROM items WHERE id = ? AND shop_id = ?",
        (item_id, shop_id),
    ).fetchone()
    return jsonify(dict(updated)), 200


@api_bp.route("/items/<int:item_id>", methods=["DELETE"])
@login_required
def delete_item(item_id: int):
    """DELETE /api/items/<id> -> Delete an item in current user's shop."""
    shop = get_current_user_shop()
    if not shop:
        return error_response("no_shop", "No bakery shop found for current user. Please complete onboarding.", 400)
    shop_id = shop["id"]

    db = get_db()
    item = db.execute(
        "SELECT id FROM items WHERE id = ? AND shop_id = ?",
        (item_id, shop_id),
    ).fetchone()
    if not item:
        return error_response("not_found", f"Item with id {item_id} does not exist.", 404)

    db.execute("DELETE FROM items WHERE id = ? AND shop_id = ?", (item_id, shop_id))
    db.commit()
    return jsonify({"status": "deleted", "id": item_id}), 200


# ---------------------------------------------------------------------------
# Daily Log Endpoints
# ---------------------------------------------------------------------------

@api_bp.route("/daily-log", methods=["POST"])
@login_required
def post_daily_log():
    """POST /api/daily-log -> Upsert daily log for an item on a date for current user's shop."""
    shop = get_current_user_shop()
    if not shop:
        return error_response("no_shop", "No bakery shop found for current user. Please complete onboarding.", 400)
    shop_id = shop["id"]

    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return error_response("invalid_json", "Request body must be a valid JSON object.", 400)

    # Validate item_id belongs to current user's shop
    item_id_val, err = parse_non_negative_int(data.get("item_id"), "item_id")
    if err:
        return error_response("invalid_input", err, 400)

    db = get_db()
    item = db.execute(
        "SELECT id, shop_id FROM items WHERE id = ? AND shop_id = ?",
        (item_id_val, shop_id),
    ).fetchone()
    if not item:
        return error_response("not_found", f"Item with id {item_id_val} does not exist.", 404)

    # Validate date
    date_str, err = parse_iso_date(data.get("date"), "date")
    if err:
        return error_response("invalid_input", err, 400)

    # Validate numeric quantities
    sold, err = parse_non_negative_int(data.get("sold"), "sold", allow_none=True)
    if err:
        return error_response("invalid_input", err, 400)

    baked, err = parse_non_negative_int(data.get("baked"), "baked", allow_none=True)
    if err:
        return error_response("invalid_input", err, 400)

    wasted, err = parse_non_negative_int(data.get("wasted"), "wasted", allow_none=True)
    if err:
        return error_response("invalid_input", err, 400)

    leftover, err = parse_non_negative_int(data.get("leftover"), "leftover", allow_none=True)
    if err:
        return error_response("invalid_input", err, 400)

    has_sold_out = "sold_out" in data
    sold_out_val = (1 if bool(data.get("sold_out")) else 0) if has_sold_out else None

    db.execute(
        """
        INSERT INTO daily_log (shop_id, item_id, date, sold, baked, wasted, leftover, sold_out)
        VALUES (?, ?, ?, ?, ?, ?, ?, COALESCE(?, 0))
        ON CONFLICT(shop_id, item_id, date) DO UPDATE SET
            sold = CASE WHEN excluded.sold IS NOT NULL THEN excluded.sold ELSE daily_log.sold END,
            baked = CASE WHEN excluded.baked IS NOT NULL THEN excluded.baked ELSE daily_log.baked END,
            wasted = CASE WHEN excluded.wasted IS NOT NULL THEN excluded.wasted ELSE daily_log.wasted END,
            leftover = CASE WHEN excluded.leftover IS NOT NULL THEN excluded.leftover ELSE daily_log.leftover END,
            sold_out = CASE WHEN ? IS NOT NULL THEN ? ELSE daily_log.sold_out END,
            created_at = CURRENT_TIMESTAMP
        """,
        (shop_id, item_id_val, date_str, sold, baked, wasted, leftover, sold_out_val, sold_out_val, sold_out_val),
    )
    if sold is not None:
        db.execute(
            """
            UPDATE forecasts
            SET actual_qty = ?
            WHERE shop_id = ? AND item_id = ? AND date = ?
            """,
            (sold, shop_id, item_id_val, date_str),
        )
    db.commit()

    # Online feedback update for Contextual Bandit policy (Phase 4D) if enabled
    try:
        from ml.rl import online_update_on_daily_log
        online_update_on_daily_log(
            shop_id=shop_id,
            item_id=item_id_val,
            log_date_str=date_str,
            sold=sold if sold is not None else 0,
            baked=baked,
            sold_out=bool(sold_out_val),
            db=db,
        )
    except Exception:
        pass

    log_row = db.execute(
        "SELECT id, shop_id, item_id, date, sold, baked, wasted, leftover, sold_out, reason FROM daily_log WHERE shop_id = ? AND item_id = ? AND date = ?",
        (shop_id, item_id_val, date_str),
    ).fetchone()
    return jsonify(dict(log_row)), 200


@api_bp.route("/daily-log", methods=["GET"])
@login_required
def get_daily_logs():
    """GET /api/daily-log?from=&to=&item_id= -> Query daily logs for current user's shop."""
    shop = get_current_user_shop()
    if not shop:
        return error_response("no_shop", "No bakery shop found for current user. Please complete onboarding.", 400)
    shop_id = shop["id"]

    from_date_raw = request.args.get("from")
    to_date_raw = request.args.get("to")
    item_id_raw = request.args.get("item_id")

    from_date, err = parse_iso_date(from_date_raw, "from", allow_none=True)
    if err:
        return error_response("invalid_input", err, 400)

    to_date, err = parse_iso_date(to_date_raw, "to", allow_none=True)
    if err:
        return error_response("invalid_input", err, 400)

    if from_date and to_date and from_date > to_date:
        return error_response("invalid_input", f"Query parameter 'from' ({from_date}) cannot be after 'to' ({to_date}).", 400)

    item_id = None
    db = get_db()
    if item_id_raw is not None:
        item_id, err = parse_non_negative_int(item_id_raw, "item_id")
        if err:
            return error_response("invalid_input", err, 400)

        item = db.execute(
            "SELECT id FROM items WHERE id = ? AND shop_id = ?",
            (item_id, shop_id),
        ).fetchone()
        if not item:
            return error_response("not_found", f"Item with id {item_id} does not exist.", 404)

    query_parts = ["SELECT d.*, i.name AS item_name FROM daily_log d JOIN items i ON d.item_id = i.id WHERE d.shop_id = ?"]
    params = [shop_id]

    if from_date:
        query_parts.append("AND d.date >= ?")
        params.append(from_date)

    if to_date:
        query_parts.append("AND d.date <= ?")
        params.append(to_date)

    if item_id is not None:
        query_parts.append("AND d.item_id = ?")
        params.append(item_id)

    query_parts.append("ORDER BY d.date DESC, d.item_id ASC")
    sql = " ".join(query_parts)

    rows = db.execute(sql, params).fetchall()
    return jsonify([dict(r) for r in rows]), 200


# ---------------------------------------------------------------------------
# Weather Endpoints
# ---------------------------------------------------------------------------

@api_bp.route("/weather", methods=["GET"])
@login_required
def get_weather():
    """GET /api/weather?date=YYYY-MM-DD -> Get weather for current user's shop with auto-refresh."""
    shop = get_current_user_shop()
    if not shop:
        return error_response("no_shop", "No bakery shop found for current user. Please complete onboarding.", 400)
    shop_id = shop["id"]

    date_raw = request.args.get("date")
    target_date, err = parse_iso_date(date_raw, "date")
    if err:
        return error_response("invalid_input", err, 400)

    weather = get_weather_for_date(shop_id, target_date)
    if not weather:
        return error_response("not_found", f"Weather data not available for date {target_date}.", 404)

    return jsonify({
        "date": weather["date"],
        "temp_mean_c": weather.get("temp_mean_c"),
        "temp_max_c": weather.get("temp_max_c"),
        "temp_min_c": weather.get("temp_min_c"),
        "precip_mm": weather.get("precip_mm"),
        "is_rainy": weather.get("is_rainy"),
        "source": weather.get("source"),
        "fetched_at": str(weather.get("fetched_at")),
    }), 200


@api_bp.route("/weather/refresh", methods=["POST"])
@login_required
def refresh_weather():
    """POST /api/weather/refresh -> Trigger forecast weather refresh for current user's shop."""
    shop = get_current_user_shop()
    if not shop:
        return error_response("no_shop", "No bakery shop found for current user. Please complete onboarding.", 400)
    shop_id = shop["id"]

    try:
        forecast_rows = fetch_forecast(float(shop["lat"]), float(shop["lon"]), days=7)
        save_weather(shop_id, forecast_rows, "forecast")
        refreshed_count = len(forecast_rows)
    except Exception as exc:
        return error_response("weather_fetch_failed", f"Failed to refresh weather: {str(exc)}", 502)

    return jsonify({"status": "ok", "refreshed_records": refreshed_count}), 200


# ---------------------------------------------------------------------------
# Forecast Endpoints
# ---------------------------------------------------------------------------

@api_bp.route("/forecast", methods=["GET"])
@login_required
def get_forecast():
    """GET /api/forecast?date=YYYY-MM-DD -> Generate and persist multi-item demand forecast for current user's shop."""
    shop = get_current_user_shop()
    if not shop:
        return error_response("no_shop", "No bakery shop found for current user. Please complete onboarding.", 400)
    shop_id = shop["id"]

    date_raw = request.args.get("date")
    if date_raw:
        target_date, err = parse_iso_date(date_raw, "date")
        if err:
            return error_response("invalid_input", err, 400)
    else:
        target_date = (date.today() + timedelta(days=1)).isoformat()

    try:
        forecasts = predict_all_items(target_date, shop_id=shop_id)
        return jsonify({"date": target_date, "forecasts": forecasts}), 200
    except Exception as exc:
        return error_response("prediction_error", str(exc), 500)


# ---------------------------------------------------------------------------
# Reports Endpoints (Forecast vs Actual)
# ---------------------------------------------------------------------------

@api_bp.route("/reports/forecast-vs-actual", methods=["GET"])
@api_bp.route("/reports", methods=["GET"])
@login_required
def get_reports_forecast_vs_actual():
    """GET /api/reports/forecast-vs-actual?item_id=&days=7&to=YYYY-MM-DD
    
    Returns forecast vs actual comparison data, summary KPIs, daily totals for charting,
    and records from daily_log and forecasts tables for the current user's shop.
    """
    shop = get_current_user_shop()
    if not shop:
        return error_response("no_shop", "No bakery shop found for current user. Please complete onboarding.", 400)
    shop_id = shop["id"]

    days_raw = request.args.get("days", 7)
    days, err = parse_non_negative_int(days_raw, "days")
    if err or days == 0:
        return error_response("invalid_input", err or "Field 'days' must be at least 1.", 400)
    if days > 90:
        days = 90

    to_date_raw = request.args.get("to")
    if to_date_raw:
        to_date_str, err = parse_iso_date(to_date_raw, "to")
        if err:
            return error_response("invalid_input", err, 400)
        end_date = date.fromisoformat(to_date_str)
    else:
        end_date = date.today()

    start_date = end_date - timedelta(days=days - 1)
    start_date_str = start_date.isoformat()
    end_date_str = end_date.isoformat()

    db = get_db()

    # Optional item_id filter
    item_id_raw = request.args.get("item_id")
    filter_item_id = None
    item_info = None

    if item_id_raw is not None and item_id_raw.strip() != "":
        parsed_id, err = parse_non_negative_int(item_id_raw.strip(), "item_id")
        if err or parsed_id == 0:
            return error_response("invalid_input", err or "Field 'item_id' must be a positive integer.", 400)
        
        item_row = db.execute(
            "SELECT id, name FROM items WHERE id = ? AND shop_id = ?",
            (parsed_id, shop_id),
        ).fetchone()
        if not item_row:
            return error_response("not_found", f"Item with id {parsed_id} not found.", 404)
        filter_item_id = parsed_id
        item_info = {"id": item_row["id"], "name": item_row["name"]}

    if filter_item_id is not None:
        sql = """
        WITH combined_keys AS (
            SELECT shop_id, item_id, date FROM forecasts
            WHERE shop_id = ? AND item_id = ? AND date >= ? AND date <= ?
            UNION
            SELECT shop_id, item_id, date FROM daily_log
            WHERE shop_id = ? AND item_id = ? AND date >= ? AND date <= ?
        )
        SELECT
            ck.date,
            ck.item_id,
            i.name AS item_name,
            f.predicted_quantity AS forecast_qty,
            d.sold AS actual_sold,
            d.baked AS baked_qty,
            d.wasted AS wasted_qty,
            d.sold_out AS sold_out,
            f.weather_source
        FROM combined_keys ck
        JOIN items i ON ck.item_id = i.id
        LEFT JOIN forecasts f ON ck.shop_id = f.shop_id AND ck.item_id = f.item_id AND ck.date = f.date
        LEFT JOIN daily_log d ON ck.shop_id = d.shop_id AND ck.item_id = d.item_id AND ck.date = d.date
        ORDER BY ck.date DESC, i.name ASC;
        """
        rows = db.execute(
            sql,
            (shop_id, filter_item_id, start_date_str, end_date_str, shop_id, filter_item_id, start_date_str, end_date_str),
        ).fetchall()
    else:
        sql = """
        WITH combined_keys AS (
            SELECT shop_id, item_id, date FROM forecasts
            WHERE shop_id = ? AND date >= ? AND date <= ?
            UNION
            SELECT shop_id, item_id, date FROM daily_log
            WHERE shop_id = ? AND date >= ? AND date <= ?
        )
        SELECT
            ck.date,
            ck.item_id,
            i.name AS item_name,
            f.predicted_quantity AS forecast_qty,
            d.sold AS actual_sold,
            d.baked AS baked_qty,
            d.wasted AS wasted_qty,
            d.sold_out AS sold_out,
            f.weather_source
        FROM combined_keys ck
        JOIN items i ON ck.item_id = i.id
        LEFT JOIN forecasts f ON ck.shop_id = f.shop_id AND ck.item_id = f.item_id AND ck.date = f.date
        LEFT JOIN daily_log d ON ck.shop_id = d.shop_id AND ck.item_id = d.item_id AND ck.date = d.date
        ORDER BY ck.date DESC, i.name ASC;
        """
        rows = db.execute(
            sql,
            (shop_id, start_date_str, end_date_str, shop_id, start_date_str, end_date_str),
        ).fetchall()

    # Build chronological dates map for consistent daily chart series
    date_series = {}
    curr = start_date
    while curr <= end_date:
        d_str = curr.isoformat()
        date_series[d_str] = {
            "date": d_str,
            "day_name": curr.strftime("%a"),
            "forecast_qty": 0.0,
            "actual_sold": 0,
            "baked_qty": 0,
            "wasted_qty": 0,
        }
        curr += timedelta(days=1)

    items_summary = {}
    records = []
    abs_errors = []
    total_forecast = 0.0
    total_sold = 0
    total_baked = 0
    total_wasted = 0

    for r in rows:
        d_str = r["date"]
        item_id = r["item_id"]
        item_name = r["item_name"]
        f_qty = round(float(r["forecast_qty"]), 1) if r["forecast_qty"] is not None else None
        s_qty = int(r["actual_sold"]) if r["actual_sold"] is not None else None
        b_qty = int(r["baked_qty"]) if r["baked_qty"] is not None else None
        w_qty = int(r["wasted_qty"]) if r["wasted_qty"] is not None else None
        sold_out = int(r["sold_out"]) if r["sold_out"] is not None else 0

        diff = round(s_qty - f_qty, 1) if (s_qty is not None and f_qty is not None) else None

        if s_qty is not None and f_qty is not None:
            abs_errors.append(abs(s_qty - f_qty))

        if f_qty is not None:
            total_forecast += f_qty
        if s_qty is not None:
            total_sold += s_qty
        if b_qty is not None:
            total_baked += b_qty
        if w_qty is not None:
            total_wasted += w_qty

        # Aggregate into daily chart series
        if d_str in date_series:
            if f_qty is not None:
                date_series[d_str]["forecast_qty"] = round(date_series[d_str]["forecast_qty"] + f_qty, 1)
            if s_qty is not None:
                date_series[d_str]["actual_sold"] += s_qty
            if b_qty is not None:
                date_series[d_str]["baked_qty"] += b_qty
            if w_qty is not None:
                date_series[d_str]["wasted_qty"] += w_qty

        # Aggregate into per-item summary
        if item_id not in items_summary:
            items_summary[item_id] = {
                "item_id": item_id,
                "item_name": item_name,
                "forecast_qty": 0.0,
                "actual_sold": 0,
                "baked_qty": 0,
                "wasted_qty": 0,
                "errors": [],
            }
        if f_qty is not None:
            items_summary[item_id]["forecast_qty"] = round(items_summary[item_id]["forecast_qty"] + f_qty, 1)
        if s_qty is not None:
            items_summary[item_id]["actual_sold"] += s_qty
        if b_qty is not None:
            items_summary[item_id]["baked_qty"] += b_qty
        if w_qty is not None:
            items_summary[item_id]["wasted_qty"] += w_qty
        if diff is not None:
            items_summary[item_id]["errors"].append(abs(diff))

        records.append({
            "date": d_str,
            "item_id": item_id,
            "item_name": item_name,
            "forecast_qty": f_qty,
            "actual_sold": s_qty,
            "baked_qty": b_qty,
            "wasted_qty": w_qty,
            "difference": diff,
            "sold_out": sold_out,
            "weather_source": r["weather_source"],
        })

    # Finalize items list
    items_list = []
    for it in items_summary.values():
        errs = it.pop("errors")
        it["mae"] = round(sum(errs) / len(errs), 1) if errs else None
        items_list.append(it)
    items_list.sort(key=lambda x: x["item_name"])

    daily_list = list(date_series.values())
    daily_list.sort(key=lambda x: x["date"])

    mae = round(sum(abs_errors) / len(abs_errors), 1) if abs_errors else None

    # Determine latest available date for this shop
    latest_row = db.execute(
        """
        SELECT MAX(d) AS max_d FROM (
            SELECT MAX(date) AS d FROM daily_log WHERE shop_id = ?
            UNION
            SELECT MAX(date) AS d FROM forecasts WHERE shop_id = ?
        )
        """,
        (shop_id, shop_id),
    ).fetchone()
    latest_date_in_db = latest_row["max_d"] if latest_row and latest_row["max_d"] else None

    # Retrieve active model version and last trained timestamp for telemetry
    active_model_info = None
    if filter_item_id is not None:
        m_row = db.execute(
            """
            SELECT model_type, version, trained_at, mae, rmse, wape, beats_baseline
            FROM model_registry
            WHERE shop_id = ? AND item_id = ? AND active = 1
            ORDER BY version DESC
            LIMIT 1
            """,
            (shop_id, filter_item_id),
        ).fetchone()
    else:
        m_row = db.execute(
            """
            SELECT model_type, version, trained_at, mae, rmse, wape, beats_baseline
            FROM model_registry
            WHERE shop_id = ? AND active = 1
            ORDER BY trained_at DESC, version DESC
            LIMIT 1
            """,
            (shop_id,),
        ).fetchone()

    if m_row:
        active_model_info = {
            "model_type": m_row["model_type"].replace("_", " ").title(),
            "version": m_row["version"],
            "trained_at": m_row["trained_at"],
            "mae": round(float(m_row["mae"]), 2) if m_row["mae"] is not None else None,
            "beats_baseline": bool(m_row["beats_baseline"]),
        }

    return jsonify({
        "item": item_info,
        "period": {
            "start_date": start_date_str,
            "end_date": end_date_str,
            "days": days,
        },
        "summary": {
            "total_forecast": round(total_forecast, 1),
            "total_sold": total_sold,
            "total_baked": total_baked,
            "total_wasted": total_wasted,
            "mae": mae,
            "records_count": len(records),
        },
        "active_model": active_model_info,
        "latest_available_date": latest_date_in_db,
        "daily": daily_list,
        "items": items_list,
        "records": records,
    }), 200


# ---------------------------------------------------------------------------
# CSV Import Endpoint
# ---------------------------------------------------------------------------

MAX_IMPORT_BYTES = 2 * 1024 * 1024  # 2 MB limit


def _parse_import_int(val, field_name: str, allow_none: bool = False):
    """Validate quantity value for import. Handles integers, float representations (e.g. 10.0), and rejects negatives."""
    if val is None or (isinstance(val, str) and not val.strip()):
        if allow_none:
            return None, None
        return None, f"Field '{field_name}' is required."

    try:
        f_val = float(val)
        if not f_val.is_integer():
            return None, f"Field '{field_name}' must be an integer (got '{val}')."
        int_val = int(f_val)
    except (ValueError, TypeError):
        return None, f"Field '{field_name}' must be an integer (got '{val}')."

    if int_val < 0:
        return None, f"Field '{field_name}' must be a non-negative integer (got '{val}')."

    return int_val, None


@api_bp.route("/import/csv", methods=["POST"])
@login_required
def import_csv():
    """POST /api/import/csv -> Upload and import sales history CSV (max 2 MB)."""
    shop = get_current_user_shop()
    if not shop:
        return error_response("no_shop", "No bakery shop found for current user. Please complete onboarding.", 400)
    shop_id = shop["id"]

    # 1. Check Content-Length if present
    if request.content_length and request.content_length > MAX_IMPORT_BYTES:
        return error_response("file_too_large", "CSV file exceeds maximum limit of 2 MB.", 400)

    # 2. Extract uploaded file
    file = request.files.get("file")
    if not file or not file.filename:
        return error_response("missing_file", "No CSV file provided in upload.", 400)

    # 3. Read content safely up to MAX_IMPORT_BYTES + 1
    raw_bytes = file.read(MAX_IMPORT_BYTES + 1)
    if len(raw_bytes) > MAX_IMPORT_BYTES:
        return error_response("file_too_large", "CSV file exceeds maximum limit of 2 MB.", 400)

    try:
        content_str = raw_bytes.decode("utf-8-sig")
    except UnicodeDecodeError:
        try:
            content_str = raw_bytes.decode("latin-1")
        except Exception as e:
            return error_response("invalid_encoding", f"Could not decode CSV text: {e}", 400)

    # 4. Parse CSV
    reader = csv.reader(io.StringIO(content_str))
    try:
        raw_headers = next(reader)
    except StopIteration:
        return error_response("empty_csv", "The uploaded CSV file is empty.", 400)

    # 5. Column Mapping
    col_date = (request.form.get("col_date") or "date").strip()
    col_item = (request.form.get("col_item") or "item").strip()
    col_sold = (request.form.get("col_sold") or "sold").strip()
    col_baked = (request.form.get("col_baked") or "baked").strip()
    col_wasted = (request.form.get("col_wasted") or "wasted").strip()
    col_leftover = (request.form.get("col_leftover") or "leftover").strip()
    create_missing_str = (request.form.get("create_missing_items") or "").strip().lower()
    create_missing_items = create_missing_str in ("true", "1", "yes", "on")

    header_indices = {h.strip().lower(): i for i, h in enumerate(raw_headers)}
    date_col_idx = header_indices.get(col_date.lower())
    item_col_idx = header_indices.get(col_item.lower())
    sold_col_idx = header_indices.get(col_sold.lower())
    baked_col_idx = header_indices.get(col_baked.lower()) if col_baked else None
    wasted_col_idx = header_indices.get(col_wasted.lower()) if col_wasted else None
    leftover_col_idx = header_indices.get(col_leftover.lower()) if col_leftover else None

    missing_cols = []
    if date_col_idx is None:
        missing_cols.append(col_date)
    if item_col_idx is None:
        missing_cols.append(col_item)
    if sold_col_idx is None:
        missing_cols.append(col_sold)

    if missing_cols:
        clean_headers = [h.strip() for h in raw_headers]
        return error_response(
            "missing_columns",
            f"Missing required column(s): {missing_cols}. Found headers: {clean_headers}",
            400,
        )

    # 6. Row Validation
    db = get_db()
    existing_rows = db.execute("SELECT id, name FROM items WHERE shop_id = ?", (shop_id,)).fetchall()
    existing_items = {r["name"].lower(): r["id"] for r in existing_rows}

    valid_rows = []
    skipped_rows = []
    seen_keys = set()
    new_items_to_create = {}

    for line_num, row in enumerate(reader, start=2):
        if not row or all(c.strip() == "" for c in row):
            continue

        date_val = row[date_col_idx].strip() if date_col_idx < len(row) else ""
        item_val = row[item_col_idx].strip() if item_col_idx < len(row) else ""
        sold_val = row[sold_col_idx].strip() if sold_col_idx < len(row) else ""
        baked_val = row[baked_col_idx].strip() if baked_col_idx is not None and baked_col_idx < len(row) else ""
        wasted_val = row[wasted_col_idx].strip() if wasted_col_idx is not None and wasted_col_idx < len(row) else ""
        leftover_val = row[leftover_col_idx].strip() if leftover_col_idx is not None and leftover_col_idx < len(row) else ""

        # Validate date
        clean_date, date_err = parse_iso_date(date_val, "date")
        if date_err:
            skipped_rows.append({"row": line_num, "reason": f"Invalid date '{date_val}' (expected YYYY-MM-DD)"})
            continue

        # Validate item
        if not item_val:
            skipped_rows.append({"row": line_num, "reason": "Missing or empty item name"})
            continue

        # Validate duplicate within file
        key = (clean_date, item_val.lower())
        if key in seen_keys:
            skipped_rows.append({"row": line_num, "reason": f"Duplicate entry for item '{item_val}' on date '{clean_date}' in file"})
            continue

        # Validate sold
        sold_num, sold_err = _parse_import_int(sold_val, "sold")
        if sold_err:
            skipped_rows.append({"row": line_num, "reason": f"Invalid sold quantity '{sold_val}' (must be non-negative integer)"})
            continue

        # Validate baked
        baked_num = None
        if baked_val:
            b_num, b_err = _parse_import_int(baked_val, "baked", allow_none=True)
            if b_err:
                skipped_rows.append({"row": line_num, "reason": f"Invalid baked quantity '{baked_val}' (must be non-negative integer)"})
                continue
            baked_num = b_num

        # Validate wasted
        wasted_num = None
        if wasted_val:
            w_num, w_err = _parse_import_int(wasted_val, "wasted", allow_none=True)
            if w_err:
                skipped_rows.append({"row": line_num, "reason": f"Invalid wasted quantity '{wasted_val}' (must be non-negative integer)"})
                continue
            wasted_num = w_num

        # Validate leftover
        leftover_num = None
        if leftover_val:
            l_num, l_err = _parse_import_int(leftover_val, "leftover", allow_none=True)
            if l_err:
                skipped_rows.append({"row": line_num, "reason": f"Invalid leftover quantity '{leftover_val}' (must be non-negative integer)"})
                continue
            leftover_num = l_num

        # Check item existence in bakery catalogue
        item_key = item_val.lower()
        if item_key not in existing_items and item_key not in new_items_to_create:
            if create_missing_items:
                new_items_to_create[item_key] = item_val
            else:
                skipped_rows.append({"row": line_num, "reason": f"Item '{item_val}' does not exist and create missing items was not checked"})
                continue

        seen_keys.add(key)
        valid_rows.append({
            "date": clean_date,
            "item_key": item_key,
            "sold": sold_num,
            "baked": baked_num,
            "wasted": wasted_num,
            "leftover": leftover_num,
        })

    # 7. Atomic Database Transaction
    created_items_list = []
    if new_items_to_create:
        for k, original_name in new_items_to_create.items():
            cur = db.execute(
                "INSERT INTO items (shop_id, name) VALUES (?, ?)",
                (shop_id, original_name),
            )
            existing_items[k] = cur.lastrowid
            created_items_list.append(original_name)

    for vr in valid_rows:
        item_id = existing_items[vr["item_key"]]
        db.execute(
            """
            INSERT INTO daily_log (shop_id, item_id, date, sold, baked, wasted, leftover)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(shop_id, item_id, date) DO UPDATE SET
                sold = excluded.sold,
                baked = COALESCE(excluded.baked, daily_log.baked),
                wasted = COALESCE(excluded.wasted, daily_log.wasted),
                leftover = COALESCE(excluded.leftover, daily_log.leftover)
            """,
            (shop_id, item_id, vr["date"], vr["sold"], vr["baked"], vr["wasted"], vr["leftover"]),
        )
        if vr["sold"] is not None:
            db.execute(
                """
                UPDATE forecasts
                SET actual_qty = ?
                WHERE shop_id = ? AND item_id = ? AND date = ?
                """,
                (vr["sold"], shop_id, item_id, vr["date"]),
            )

    db.commit()

    # 8. Trigger Post-Import Weather Synchronization
    weather_result = None
    date_range = None
    if valid_rows:
        min_date = min(r["date"] for r in valid_rows)
        max_date = max(r["date"] for r in valid_rows)
        date_range = {"start": min_date, "end": max_date}
        weather_result = sync_weather_for_date_range(shop_id, min_date, max_date)

    return jsonify({
        "status": "success",
        "rows_imported": len(valid_rows),
        "rows_skipped": len(skipped_rows),
        "created_items": created_items_list,
        "skipped_rows": skipped_rows,
        "date_range": date_range,
        "weather_synced": weather_result,
    }), 200


# ---------------------------------------------------------------------------
# Admin & Model Retraining Endpoints
# ---------------------------------------------------------------------------

@api_bp.route("/admin/retrain", methods=["POST"])
@login_required
def admin_retrain():
    """POST /api/admin/retrain -> Retrain demand models for current shop owner."""
    shop = get_current_user_shop()
    if not shop:
        return error_response("no_shop", "No bakery shop found for current user.", 400)

    # Check ownership
    if shop["owner_user_id"] != current_user.id:
        return error_response("forbidden", "Only the shop owner can retrain demand models.", 403)

    shop_id = shop["id"]
    db = get_db()
    try:
        from ml.train import train_shop_models
        evaluations = train_shop_models(shop_id, db=db)
        items_summary = []
        for ev in evaluations:
            rec = ev.get("registry_record", {})
            items_summary.append({
                "item_id": ev.get("item_id"),
                "item_name": ev.get("item_name"),
                "model_type": rec.get("model_type"),
                "version": rec.get("version"),
                "mae": rec.get("mae"),
                "rmse": rec.get("rmse"),
                "wape": rec.get("wape"),
                "beats_baseline": bool(rec.get("beats_baseline")),
                "active": bool(rec.get("active")),
                "cold_start": ev.get("cold_start", False),
            })
        return jsonify({
            "status": "success",
            "shop_id": shop_id,
            "shop_name": shop["name"],
            "trained_at": datetime.now(timezone.utc).isoformat(),
            "items_trained": items_summary,
            "message": f"Successfully retrained models for {len(items_summary)} items.",
        }), 200
    except Exception as exc:
        return error_response("training_error", str(exc), 500)


@api_bp.route("/admin/model-status", methods=["GET"])
@login_required
def admin_model_status():
    """GET /api/admin/model-status -> Returns active models and versions for current shop."""
    shop = get_current_user_shop()
    if not shop:
        return error_response("no_shop", "No bakery shop found for current user.", 400)

    shop_id = shop["id"]
    db = get_db()
    rows = db.execute(
        """
        SELECT m.id, m.item_id, i.name AS item_name, m.model_type, m.version,
               m.trained_at, m.mae, m.rmse, m.wape, m.beats_baseline, m.active
        FROM model_registry m
        JOIN items i ON m.item_id = i.id
        WHERE m.shop_id = ? AND m.active = 1
        ORDER BY i.name ASC
        """,
        (shop_id,),
    ).fetchall()

    return jsonify([dict(r) for r in rows]), 200


# ---------------------------------------------------------------------------
# Phase 5: Accuracy, Waste Report, Simulation, and CSV Exports
# ---------------------------------------------------------------------------

@api_bp.route("/reports/accuracy", methods=["GET"])
@login_required
def get_reports_accuracy():
    """GET /api/reports/accuracy?days=28&to=YYYY-MM-DD
    
    Computes per-item and shop aggregate MAE, model WAPE, and 7-day-lag naive baseline WAPE
    over the last 28 days using forecasts where actual_qty is populated.
    """
    shop = get_current_user_shop()
    if not shop:
        return error_response("no_shop", "No bakery shop found for current user. Please complete onboarding.", 400)
    shop_id = shop["id"]

    days_raw = request.args.get("days", 28)
    days, err = parse_non_negative_int(days_raw, "days")
    if err or days == 0:
        return error_response("invalid_input", err or "Field 'days' must be at least 1.", 400)

    to_date_raw = request.args.get("to")
    if to_date_raw:
        to_date_str, err = parse_iso_date(to_date_raw, "to")
        if err:
            return error_response("invalid_input", err, 400)
        end_date = date.fromisoformat(to_date_str)
    else:
        db = get_db()
        max_d_row = db.execute(
            "SELECT MAX(date) AS max_d FROM forecasts WHERE shop_id = ? AND actual_qty IS NOT NULL",
            (shop_id,),
        ).fetchone()
        if max_d_row and max_d_row["max_d"]:
            end_date = date.fromisoformat(max_d_row["max_d"])
        else:
            end_date = date.today()

    start_date = end_date - timedelta(days=days - 1)
    start_date_str = start_date.isoformat()
    end_date_str = end_date.isoformat()

    db = get_db()
    query = """
    SELECT
        f.item_id,
        i.name AS item_name,
        f.date,
        f.predicted_quantity,
        f.actual_qty,
        prev.sold AS lag7_sold
    FROM forecasts f
    JOIN items i ON f.item_id = i.id
    LEFT JOIN daily_log prev ON prev.shop_id = f.shop_id AND prev.item_id = f.item_id AND prev.date = date(f.date, '-7 days')
    WHERE f.shop_id = ? AND f.actual_qty IS NOT NULL AND f.date >= ? AND f.date <= ?
    ORDER BY i.name ASC, f.date ASC;
    """
    rows = db.execute(query, (shop_id, start_date_str, end_date_str)).fetchall()

    items_map = {}
    all_abs_errors = []
    all_actuals = []
    all_base_errors = []

    for r in rows:
        item_id = r["item_id"]
        item_name = r["item_name"]
        pred = float(r["predicted_quantity"])
        act = float(r["actual_qty"])
        lag7 = float(r["lag7_sold"]) if r["lag7_sold"] is not None else None

        abs_err = abs(act - pred)
        base_err = abs(act - lag7) if lag7 is not None else abs_err

        if item_id not in items_map:
            items_map[item_id] = {
                "item_id": item_id,
                "item_name": item_name,
                "records_count": 0,
                "abs_errors": [],
                "actuals": [],
                "base_errors": [],
            }

        items_map[item_id]["records_count"] += 1
        items_map[item_id]["abs_errors"].append(abs_err)
        items_map[item_id]["actuals"].append(act)
        items_map[item_id]["base_errors"].append(base_err)

        all_abs_errors.append(abs_err)
        all_actuals.append(act)
        all_base_errors.append(base_err)

    items_list = []
    for it in items_map.values():
        sum_act = sum(it["actuals"])
        sum_err = sum(it["abs_errors"])
        sum_base_err = sum(it["base_errors"])
        n = it["records_count"]

        mae = round(sum_err / n, 2) if n > 0 else 0.0
        wape = round(sum_err / sum_act, 4) if sum_act > 0 else 0.0
        baseline_wape = round(sum_base_err / sum_act, 4) if sum_act > 0 else 0.0
        beats_baseline = bool(wape < baseline_wape)

        items_list.append({
            "item_id": it["item_id"],
            "item_name": it["item_name"],
            "records_count": n,
            "mae": mae,
            "wape": wape,
            "wape_pct": round(wape * 100, 2),
            "baseline_wape": baseline_wape,
            "baseline_wape_pct": round(baseline_wape * 100, 2),
            "beats_baseline": beats_baseline,
        })

    overall_n = len(all_actuals)
    overall_sum_act = sum(all_actuals)
    overall_sum_err = sum(all_abs_errors)
    overall_sum_base_err = sum(all_base_errors)

    overall_mae = round(overall_sum_err / overall_n, 2) if overall_n > 0 else 0.0
    overall_wape = round(overall_sum_err / overall_sum_act, 4) if overall_sum_act > 0 else 0.0
    overall_base_wape = round(overall_sum_base_err / overall_sum_act, 4) if overall_sum_act > 0 else 0.0
    overall_beats = bool(overall_wape < overall_base_wape)

    return jsonify({
        "status": "success",
        "period": {
            "days": days,
            "start_date": start_date_str,
            "end_date": end_date_str,
        },
        "items": items_list,
        "overall": {
            "records_count": overall_n,
            "mae": overall_mae,
            "wape": overall_wape,
            "wape_pct": round(overall_wape * 100, 2),
            "baseline_wape": overall_base_wape,
            "baseline_wape_pct": round(overall_base_wape * 100, 2),
            "beats_baseline": overall_beats,
        },
    }), 200


@api_bp.route("/reports/waste", methods=["GET"])
@login_required
def get_reports_waste():
    """GET /api/reports/waste?days=28&to=YYYY-MM-DD
    
    Aggregates logged wasted and leftover quantities and costs for the last 4 weeks (28 days).
    """
    shop = get_current_user_shop()
    if not shop:
        return error_response("no_shop", "No bakery shop found for current user. Please complete onboarding.", 400)
    shop_id = shop["id"]

    days_raw = request.args.get("days", 28)
    days, err = parse_non_negative_int(days_raw, "days")
    if err or days == 0:
        return error_response("invalid_input", err or "Field 'days' must be at least 1.", 400)

    to_date_raw = request.args.get("to")
    if to_date_raw:
        to_date_str, err = parse_iso_date(to_date_raw, "to")
        if err:
            return error_response("invalid_input", err, 400)
        end_date = date.fromisoformat(to_date_str)
    else:
        db = get_db()
        max_d_row = db.execute(
            "SELECT MAX(date) AS max_d FROM daily_log WHERE shop_id = ?",
            (shop_id,),
        ).fetchone()
        if max_d_row and max_d_row["max_d"]:
            end_date = date.fromisoformat(max_d_row["max_d"])
        else:
            end_date = date.today()

    start_date = end_date - timedelta(days=days - 1)
    start_date_str = start_date.isoformat()
    end_date_str = end_date.isoformat()

    db = get_db()
    rows = db.execute(
        """
        SELECT
            d.item_id,
            i.name AS item_name,
            COALESCE(i.unit_cost, 0.5) AS unit_cost,
            SUM(COALESCE(d.baked, 0)) AS total_baked,
            SUM(COALESCE(d.sold, 0)) AS total_sold,
            SUM(COALESCE(d.wasted, 0)) AS total_wasted,
            SUM(COALESCE(d.leftover, 0)) AS total_leftover
        FROM daily_log d
        JOIN items i ON d.item_id = i.id
        WHERE d.shop_id = ? AND d.date >= ? AND d.date <= ?
        GROUP BY d.item_id, i.name, i.unit_cost
        ORDER BY i.name ASC;
        """,
        (shop_id, start_date_str, end_date_str),
    ).fetchall()

    items_list = []
    shop_baked = 0
    shop_sold = 0
    shop_wasted = 0
    shop_leftover = 0
    shop_loss_units = 0
    shop_waste_cost = 0.0

    for r in rows:
        baked = int(r["total_baked"])
        sold = int(r["total_sold"])
        wasted = int(r["total_wasted"])
        leftover = int(r["total_leftover"])
        cost = float(r["unit_cost"])

        total_loss = wasted + leftover
        waste_cost = round(total_loss * cost, 2)
        waste_rate = round((total_loss / baked) * 100, 2) if baked > 0 else 0.0

        shop_baked += baked
        shop_sold += sold
        shop_wasted += wasted
        shop_leftover += leftover
        shop_loss_units += total_loss
        shop_waste_cost += waste_cost

        items_list.append({
            "item_id": r["item_id"],
            "item_name": r["item_name"],
            "unit_cost": cost,
            "total_baked": baked,
            "total_sold": sold,
            "total_wasted": wasted,
            "total_leftover": leftover,
            "total_loss_units": total_loss,
            "waste_cost": waste_cost,
            "waste_rate_pct": waste_rate,
        })

    shop_waste_rate = round((shop_loss_units / shop_baked) * 100, 2) if shop_baked > 0 else 0.0

    return jsonify({
        "status": "success",
        "period": {
            "days": days,
            "start_date": start_date_str,
            "end_date": end_date_str,
        },
        "items": items_list,
        "summary": {
            "total_baked": shop_baked,
            "total_sold": shop_sold,
            "total_wasted": shop_wasted,
            "total_leftover": shop_leftover,
            "total_loss_units": shop_loss_units,
            "total_waste_cost": round(shop_waste_cost, 2),
            "waste_rate_pct": shop_waste_rate,
        },
    }), 200


@api_bp.route("/reports/simulation", methods=["GET"])
@login_required
def get_reports_simulation():
    """GET /api/reports/simulation?days=60
    
    Returns 60-day historical replay simulation comparing ordering policies from evaluation_results table,
    or executes on-the-fly simulation if not yet populated.
    """
    shop = get_current_user_shop()
    if not shop:
        return error_response("no_shop", "No bakery shop found for current user. Please complete onboarding.", 400)
    shop_id = shop["id"]
    db = get_db()

    rows = db.execute(
        """
        SELECT policy_name, period_start, period_end, days_count,
               total_baked, total_sold, total_waste_units, total_waste_cost,
               total_lost_sales_units, total_lost_sales_cost, net_profit, savings_vs_baseline
        FROM evaluation_results
        WHERE shop_id = ? AND item_id IS NULL
        ORDER BY id ASC;
        """,
        (shop_id,),
    ).fetchall()

    if not rows:
        from ml.simulation import run_historical_simulation
        res = run_historical_simulation(shop_id=shop_id, days_count=60, db=db, verbose=False)
        return jsonify(res), 200

    from ml.simulation import POLICY_KEYS
    policy_label_map = dict(POLICY_KEYS)

    policies = []
    p_start = rows[0]["period_start"]
    p_end = rows[0]["period_end"]
    p_days = rows[0]["days_count"]

    for r in rows:
        policies.append({
            "policy_name": r["policy_name"],
            "display_name": policy_label_map.get(r["policy_name"], r["policy_name"]),
            "total_baked": r["total_baked"],
            "total_sold": r["total_sold"],
            "waste_units": r["total_waste_units"],
            "waste_cost": r["total_waste_cost"],
            "lost_sales_units": r["total_lost_sales_units"],
            "lost_sales_cost": r["total_lost_sales_cost"],
            "net_profit": r["net_profit"],
            "savings_vs_baseline": r["savings_vs_baseline"],
        })

    return jsonify({
        "status": "success",
        "simulation_label": "Simulation on past data",
        "shop_id": shop_id,
        "shop_name": shop["name"],
        "period": {
            "start": p_start,
            "end": p_end,
            "days": p_days,
        },
        "policies": policies,
    }), 200


@api_bp.route("/export/daily-log", methods=["GET"])
@login_required
def export_daily_log_csv():
    """GET /api/export/daily-log -> Export shop's daily_log as CSV."""
    shop = get_current_user_shop()
    if not shop:
        return error_response("no_shop", "No bakery shop found for current user.", 400)
    shop_id = shop["id"]
    db = get_db()

    rows = db.execute(
        """
        SELECT d.date, i.name AS item_name, d.sold, d.baked, d.wasted, d.leftover, d.sold_out, d.reason
        FROM daily_log d
        JOIN items i ON d.item_id = i.id
        WHERE d.shop_id = ?
        ORDER BY d.date DESC, i.name ASC;
        """,
        (shop_id,),
    ).fetchall()

    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["date", "item_name", "sold", "baked", "wasted", "leftover", "sold_out", "reason"])
    for r in rows:
        writer.writerow([
            r["date"],
            r["item_name"],
            r["sold"] if r["sold"] is not None else "",
            r["baked"] if r["baked"] is not None else "",
            r["wasted"] if r["wasted"] is not None else "",
            r["leftover"] if r["leftover"] is not None else "",
            r["sold_out"] if r["sold_out"] is not None else 0,
            r["reason"] or "",
        ])

    csv_data = output.getvalue()
    today_str = date.today().isoformat()
    return Response(
        csv_data,
        mimetype="text/csv",
        headers={"Content-Disposition": f"attachment; filename=daily_log_{today_str}.csv"},
    )


@api_bp.route("/export/forecasts", methods=["GET"])
@login_required
def export_forecasts_csv():
    """GET /api/export/forecasts -> Export shop's forecasts history as CSV."""
    shop = get_current_user_shop()
    if not shop:
        return error_response("no_shop", "No bakery shop found for current user.", 400)
    shop_id = shop["id"]
    db = get_db()

    rows = db.execute(
        """
        SELECT f.date, i.name AS item_name, f.predicted_quantity, f.actual_qty,
               f.weather_source, f.cold_start, f.model_version, f.created_at
        FROM forecasts f
        JOIN items i ON f.item_id = i.id
        WHERE f.shop_id = ?
        ORDER BY f.date DESC, i.name ASC;
        """,
        (shop_id,),
    ).fetchall()

    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow([
        "date", "item_name", "predicted_quantity", "actual_qty",
        "weather_source", "cold_start", "model_version", "created_at"
    ])
    for r in rows:
        writer.writerow([
            r["date"],
            r["item_name"],
            r["predicted_quantity"],
            r["actual_qty"] if r["actual_qty"] is not None else "",
            r["weather_source"] or "",
            r["cold_start"] if r["cold_start"] is not None else 0,
            r["model_version"] or "",
            r["created_at"] or "",
        ])

    csv_data = output.getvalue()
    today_str = date.today().isoformat()
    return Response(
        csv_data,
        mimetype="text/csv",
        headers={"Content-Disposition": f"attachment; filename=forecasts_{today_str}.csv"},
    )


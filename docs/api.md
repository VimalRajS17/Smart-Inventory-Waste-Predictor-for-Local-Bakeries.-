# Loaf Ledger REST API & Authentication Documentation

Base URL: `/api`

### Authentication & CSRF Protection
- **Session Authentication**: All `/api/*` endpoints (except `/api/health`) require authentication via session cookies (`HttpOnly`, `SameSite=Lax`).
- **CSRF Protection**: All state-changing methods (`POST`, `PUT`, `DELETE`) require the `X-CSRFToken` header containing the session's CSRF token.
- **Tenant Scoping**: All operations automatically scope to the authenticated user's registered bakery shop. `shop_id` cannot be overridden via query or body parameters.

All JSON error responses follow a standardized schema:
```json
{
  "error": {
    "code": "unauthorized | csrf_error | no_shop | invalid_input | duplicate_item | not_found | weather_fetch_failed | server_error",
    "message": "Clear explanation of the error."
  }
}
```

---

## 1. System & Health Endpoints (Public)

### `GET /api/health`
Check application availability, database connectivity, background scheduler status, and RL feature flag.
- **Authentication**: None (Public)
- **Response (200 OK):**
  ```json
  {
    "status": "ok",
    "database": "connected",
    "scheduler": "running",
    "rl_enabled": false,
    "timestamp": "2026-11-01T12:00:00+00:00"
  }
  ```
- **Response (503 Service Unavailable):**
  ```json
  {
    "status": "degraded",
    "database": "disconnected",
    "error": "OperationalError: unable to open database file",
    "scheduler": "stopped",
    "rl_enabled": false,
    "timestamp": "2026-11-01T12:00:00+00:00"
  }
  ```

---

## 2. Authentication Endpoints (Web & JSON)

### `POST /register`
Create a new user account. Supports both JSON and form data.
- **Request Body (JSON):**
  ```json
  {
    "email": "baker@example.com",
    "password": "securepassword123",
    "confirm_password": "securepassword123"
  }
  ```
- **Response (201 Created):**
  ```json
  {
    "status": "ok",
    "user": { "id": 2, "email": "baker@example.com" },
    "redirect_url": "/onboarding"
  }
  ```

### `POST /login`
Authenticate existing user and establish session cookie. Protected by sliding-window rate limiting (10 attempts/min/IP).
- **Request Body (JSON):**
  ```json
  {
    "email": "baker@example.com",
    "password": "securepassword123",
    "remember": true
  }
  ```
- **Response (200 OK):**
  ```json
  {
    "status": "ok",
    "user": { "id": 2, "email": "baker@example.com" },
    "redirect_url": "/plan"
  }
  ```

### `POST /logout`
Invalidate user session.
- **Response (200 OK):**
  ```json
  {
    "status": "ok",
    "redirect_url": "/login"
  }
  ```

### `POST /onboarding`
Create or configure bakery shop details (with coordinates and initial items) for the authenticated user.
- **Request Body (JSON):**
  ```json
  {
    "name": "Sunrise Artisan Bakery",
    "city": "Bristol",
    "lat": 51.4545,
    "lon": -2.5879,
    "items": [
      { "name": "Cardamom Bun", "current_stock": 15, "shelf_life_days": 1, "unit_price": 3.5, "unit_cost": 1.2 },
      { "name": "Focaccia", "current_stock": 10, "shelf_life_days": 2, "unit_price": 4.0, "unit_cost": 1.5 }
    ]
  }
  ```
- **Response (201 Created):**
  ```json
  {
    "status": "ok",
    "shop": { "id": 2, "name": "Sunrise Artisan Bakery", "city": "Bristol", "lat": 51.4545, "lon": -2.5879 },
    "items_created": 2,
    "redirect_url": "/plan"
  }
  ```

---

## 3. Items Catalog Endpoints

### `GET /api/items`
List all bakery items for the authenticated user's shop.
- **Response (200 OK):**
  ```json
  [
    {
      "id": 1,
      "shop_id": 1,
      "name": "Croissant",
      "current_stock": 20,
      "shelf_life_days": 1,
      "unit_price": 2.50,
      "unit_cost": 0.80
    }
  ]
  ```

### `POST /api/items`
Add a new item to the bakery catalog.
- **Request Body:**
  ```json
  {
    "name": "Almond Croissant",
    "current_stock": 15,
    "shelf_life_days": 1,
    "unit_price": 3.20,
    "unit_cost": 1.10
  }
  ```
- **Response (201 Created):** Item JSON object.

### `GET /api/items/<id>`
Fetch an item by ID.

### `PUT /api/items/<id>`
Update item properties (`name`, `current_stock`, `shelf_life_days`, `unit_price`, `unit_cost`).

### `DELETE /api/items/<id>`
Delete an item from catalog (cascades to logs and forecasts).

---

## 4. Daily Logging Endpoints

### `GET /api/daily-log`
Fetch daily logs for the authenticated shop.
- **Query Parameters:** `date` (YYYY-MM-DD), `start_date`, `end_date`, `item_id`.
- **Response (200 OK):** Array of daily log records.

### `POST /api/daily-log`
Upsert a daily log entry for an item on a given date. Automatically synchronizes `actual_qty` on matching `forecasts` rows.
- **Request Body:**
  ```json
  {
    "item_id": 1,
    "date": "2026-11-01",
    "sold": 45,
    "baked": 50,
    "wasted": 5,
    "leftover": 0,
    "sold_out": false,
    "reason": "Rainy afternoon slowdown"
  }
  ```
- **Response (200 OK):**
  ```json
  {
    "status": "ok",
    "message": "Daily log saved successfully.",
    "actual_qty_synced": true
  }
  ```

---

## 5. Demand Forecast Endpoints

### `GET /api/forecast`
Generates demand predictions, safety buffers, carry-over adjustments, and recommended bake quantities for the specified target date.
- **Query Parameters:**
  - `date` (optional, default: tomorrow): Forecast target date (YYYY-MM-DD).
  - `item_id` (optional): Filter to a specific catalog item.
- **Response (200 OK):**
  ```json
  {
    "forecast_date": "2026-11-02",
    "shop": { "id": 1, "name": "Demo Bakery", "city": "London" },
    "weather": {
      "temperature_c": 14.5,
      "precip_mm": 2.4,
      "is_rainy": true,
      "is_holiday": false,
      "source": "forecast"
    },
    "items": [
      {
        "item_id": 1,
        "item_name": "Croissant",
        "current_stock": 5,
        "predicted_demand": 42.0,
        "buffer_qty": 4.2,
        "shelf_life_days": 1,
        "recommended_bake": 41,
        "model_version": 2,
        "rl_multiplier": 1.0
      }
    ]
  }
  ```

---

## 6. CSV History Import

### `POST /api/import/csv`
Upload and import a historical sales CSV file (up to 2 MB) into the authenticated user's `daily_log`. Automatically detects or maps columns, validates rows, auto-creates catalog items if configured, and triggers weather synchronization.
- **Content-Type**: `multipart/form-data`
- **Parameters**: `file`, `col_date`, `col_item`, `col_sold`, `col_baked`, `col_wasted`, `create_missing_items`.
- **Response (200 OK):**
  ```json
  {
    "status": "success",
    "rows_imported": 45,
    "rows_skipped": 0,
    "created_items": ["Brioche"],
    "date_range": { "start": "2026-01-01", "end": "2026-01-31" }
  }
  ```

---

## 7. Model Retraining & Admin Endpoints

### `POST /api/admin/retrain`
Trigger on-demand retraining of machine learning regression models for the authenticated user's shop. Models are evaluated against the 7-day baseline moving average; only models that beat the baseline WAPE are promoted to active status.
- **Authentication**: Required (`@login_required`, Shop Owner only)
- **Response (200 OK):**
  ```json
  {
    "status": "success",
    "message": "Model retraining completed: 5 items evaluated, 5 promoted, 0 unpromoted.",
    "summary": { "items_evaluated": 5, "promoted": 5, "unpromoted": 0 },
    "details": [
      {
        "item_id": 1,
        "item_name": "Croissant",
        "new_version": 2,
        "model_type": "random_forest",
        "wape": 0.082,
        "baseline_wape": 0.145,
        "promoted": true
      }
    ]
  }
  ```

### `GET /api/settings/retrain-status`
Retrieve the latest retraining version, timestamp, and active model status for all items in the shop.
- **Response (200 OK):**
  ```json
  {
    "last_trained_at": "2026-11-01T08:00:00Z",
    "total_models": 5,
    "active_models": 5,
    "models": [...]
  }
  ```

---

## 8. Analytics, Reports & Simulation Endpoints

### `GET /api/reports/accuracy`
Returns MAE, WAPE, and baseline WAPE over the past 28 days for each catalog item, calculated strictly from forecasts that have corresponding recorded `actual_qty`.
- **Query Parameters:** `days` (optional, default: 28).
- **Response (200 OK):**
  ```json
  {
    "lookback_days": 28,
    "overall_wape": 0.095,
    "overall_baseline_wape": 0.162,
    "items": [
      {
        "item_id": 1,
        "item_name": "Croissant",
        "days_evaluated": 28,
        "mae": 3.2,
        "wape": 0.085,
        "baseline_wape": 0.150,
        "beats_baseline": true,
        "active_model_version": 2
      }
    ]
  }
  ```

### `GET /api/reports/waste`
Returns aggregated waste and leftover quantities and costs for the preceding 4 weeks.
- **Query Parameters:** `days` (optional, default: 28).
- **Response (200 OK):**
  ```json
  {
    "lookback_days": 28,
    "total_waste_units": 42,
    "total_waste_cost": 21.0,
    "items": [...]
  }
  ```

### `GET /api/reports/evaluation`
Returns policy simulation results across the past 60 days comparing: Same as Last Week (baseline), Forecast Only, Forecast + Buffer, and RL Contextual Bandit. Labelled explicitly as a simulation on past data.
- **Response (200 OK):**
  ```json
  {
    "notice": "Simulation on past data",
    "days_evaluated": 60,
    "policies": [
      {
        "policy_name": "forecast_plus_buffer",
        "display_name": "Forecast + Buffer",
        "total_baked": 3120,
        "total_sold": 2980,
        "waste_units": 140,
        "waste_cost": 70.0,
        "lost_sales_units": 12,
        "lost_sales_cost": 24.0,
        "net_profit": 2836.0,
        "savings_vs_baseline": 340.50
      }
    ]
  }
  ```

---

## 9. Data Export Endpoints (CSV)

### `GET /api/reports/export/daily-log`
Exports the complete historical daily log (`daily_log`) of the authenticated bakery shop as a downloadable CSV file.
- **Response**: `Content-Type: text/csv; charset=utf-8`, attachment filename `daily_log_<date>.csv`.

### `GET /api/reports/export/forecasts`
Exports historical forecast predictions, buffers, and actual sales (`forecasts`) as a downloadable CSV file.
- **Response**: `Content-Type: text/csv; charset=utf-8`, attachment filename `forecasts_<date>.csv`.

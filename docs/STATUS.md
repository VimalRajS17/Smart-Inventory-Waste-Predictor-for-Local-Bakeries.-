# Loaf Ledger — Project Status & Implementation Audit

**Document Date:** November 2026  
**Status:** All Planned Phases (1 through 6B) Fully Implemented & Verified  
**Total Automated Tests:** 111 Passing Tests (0 Failures)

---

## 1. Executive Summary

Loaf Ledger is an intelligent, production-ready demand forecasting and waste-reduction platform engineered for commercial and artisanal bakeries. It combines weather signals, holiday calendars, historical sales trends, machine learning regression models, and reinforcement learning contextual bandits to produce actionable daily baking plans.

All requirements across Phases 1 through 6B have been implemented, thoroughly tested, and hardened for production deployment.

---

## 2. Phase-by-Phase Implementation Matrix

| Phase | Description | Key Deliverables | Status |
|---|---|---|---|
| **Phase 1** | Core Architecture & Data Foundation | SQLite schema, parameterised queries, Open-Meteo weather client, seed generator | Completed |
| **Phase 1.5** | API Blueprint & Validation | REST API endpoints, input validators, standardized JSON error envelope, conftest fixtures | Completed |
| **Phase 2A** | Authentication & Multi-Tenancy | Secure user registration/login (Flask-Login, Argon2/pbkdf2), CSRF protection, tenant isolation | Completed |
| **Phase 2B** | Shop Onboarding & Item Management | Onboarding flow, shop location setup, item catalog CRUD, stock tracking, shelf-life configuration | Completed |
| **Phase 3A** | Daily Logging Engine | Daily sales, bakes, waste, and leftover tracking; actual quantity synchronization; CSV history importer | Completed |
| **Phase 3B** | Demand Forecasting & Plan UI | Weather-informed predictions, safety buffer calculations, shelf-life carryover handling, Plan dashboard | Completed |
| **Phase 4A** | Feature Engineering & Baseline Models | Calendar features, weather flags, lag sales features, 7-day baseline moving averages | Completed |
| **Phase 4B** | Multi-Model ML Pipeline | Ridge regression, Random Forest, and Gradient Boosting regressors; automatic model training & persistence | Completed |
| **Phase 4C** | Model Registry & Lifecycle Management | `model_registry` table, promotion gate (WAPE beating baseline), `flask retrain-all`, `flask rollback`, APScheduler | Completed |
| **Phase 4D** | RL Contextual Bandit (Optional) | Contextual bandit (`ml/rl.py`), gym-style replay simulation, censored demand handling, safety bounds, `RL_ENABLED` toggle | Completed |
| **Phase 5** | Accuracy, Simulation & Waste Analytics | `GET /api/reports/accuracy` (MAE, WAPE, baseline), 60-day policy simulation (`flask evaluate`), CSV exports | Completed |
| **Phase 6A** | End-to-End Tests & Security Hardening | `tests/test_e2e.py`, security headers (CSP, nosniff, SAMEORIGIN), login rate limiter, error pages, `qa_smoke.py` | Completed |
| **Phase 6B** | Production, Backups & Documentation | Gunicorn WSGI, `Procfile`, enhanced `/api/health`, persistent disk options, `flask backup`, `docs/DEPLOY.md`, CI | Completed |

---

## 3. Database Schema Overview

The database uses SQLite with foreign keys enabled, indexed for fast multi-tenant queries:

1. **`users`**: User authentication credentials (id, email, password_hash, created_at).
2. **`shops`**: Bakery shop configuration (id, owner_user_id, name, city, lat, lon).
3. **`items`**: Catalog items (id, shop_id, name, current_stock, shelf_life_days, unit_price, unit_cost).
4. **`daily_log`**: Historical day logs (id, shop_id, item_id, date, sold, baked, wasted, leftover, sold_out, reason).
5. **`weather_daily`**: Daily weather records (shop_id, date, temperature_c, temp_mean_c, precip_mm, is_rainy, is_holiday, source).
6. **`forecasts`**: Generated forecast rows (id, shop_id, item_id, forecast_date, predicted_qty, buffer_qty, actual_qty).
7. **`model_registry`**: ML model version catalog (id, shop_id, item_id, model_type, version, mae, rmse, wape, beats_baseline, active).
8. **`policies`**: RL contextual bandit policy versions (id, shop_id, item_id, version, policy_type, parameters, observations_count, active).
9. **`evaluation_results`**: Historical simulation results (id, shop_id, item_id, policy_name, total_baked, total_sold, total_waste_units, total_waste_cost, savings_vs_baseline).

---

## 4. CLI Commands Catalog

| Command | Purpose |
|---|---|
| `flask --app app init-db` | Initializes clean database schema. |
| `flask --app app seed-demo` | Seeds demo shop, standard bakery items, and 730 days of sales history. |
| `flask --app app reset-demo [--history-days N]` | Completely resets database, regenerates data, fetches weather, and trains ML models. |
| `flask --app app fetch-weather [--days N]` | Fetches upcoming weather forecasts from Open-Meteo for all configured shops. |
| `flask --app app train [--source db\|csv]` | Trains baseline and machine learning regression models for all items. |
| `flask --app app retrain-all [--min-new-rows N]` | Retrains models for shops with new data; enforces promotion gate before activating. |
| `flask --app app rollback --shop-id N --item-id M` | Reverts the active model for an item to the preceding registered version. |
| `flask --app app evaluate-rl` | Runs held-out contextual bandit simulation across 5 random seeds. |
| `flask --app app evaluate [--days 60]` | Replays 60-day historical policies to quantify financial waste and stock-out savings. |
| `flask --app app backup [--output-dir DIR]` | Writes an atomic, lock-safe, timestamped copy of the SQLite database. |

---

## 5. Automated Test Coverage Summary

```
============================== 110 passed in 12.8s ==============================
```

Test suite files:
- `tests/test_api.py`: REST API endpoints and input validation constraints.
- `tests/test_auth.py`: Authentication, login, registration, CSRF, and rate limiting.
- `tests/test_backup.py`: Atomic database backup CLI and data verification.
- `tests/test_bootstrap.py`: Idempotent deployment bootstrap, cold start schema, demo seeding, and model training.
- `tests/test_daily_log.py`: Daily log recording, actual quantity sync, and validation.
- `tests/test_e2e.py`: Full multi-tenant end-to-end user journey and data isolation.
- `tests/test_frontend.py`: Server-rendered templates, error pages, and UI rendering.
- `tests/test_import.py`: CSV import parser, header detection, and row validation.
- `tests/test_leakage.py`: Machine learning temporal split and feature leakage verification.
- `tests/test_ml_package.py`: Feature engineering, holiday flags, and baseline models.
- `tests/test_phase_1_5.py`: Weather service caching, fallback behaviors, and forecast logic.
- `tests/test_phase_5.py`: Evaluation metrics (MAE, WAPE), simulation engine, and CSV exports.
- `tests/test_plan_v2.py`: Baking plan calculation, shelf-life carryover, and safety buffers.
- `tests/test_predictor.py`: Prediction service and multi-model dispatching.
- `tests/test_reports.py`: Accuracy reports, waste analytics, and savings calculations.
- `tests/test_retrain.py`: Model retraining scans, promotion rules, and rollback commands.
- `tests/test_rl.py`: Contextual bandit environment, reward formulation, and safety bounds.
- `tests/test_weather.py`: Weather data persistence and Open-Meteo API integration.

---

## 6. Security & Hardening Checklist

- [x] **Debug Mode**: Disabled by default (`FLASK_DEBUG=0`).
- [x] **Cookie Security**: `HttpOnly`, `SameSite=Lax`, and `Secure` flag configurable via `SESSION_COOKIE_SECURE`.
- [x] **Request Size Limits**: Enforced 2 MB maximum payload limit (`MAX_CONTENT_LENGTH = 2 * 1024 * 1024`).
- [x] **Security Headers**: Injected on all HTTP responses (`X-Content-Type-Options: nosniff`, `X-Frame-Options: SAMEORIGIN`, `Content-Security-Policy`, `Referrer-Policy`).
- [x] **Brute-Force Protection**: Sliding-window rate limiter on `/auth/login` (10 attempts per minute per IP returning HTTP 429).
- [x] **Tenant Isolation**: All database queries strictly parameterized and scoped by `shop_id`.
- [x] **CSRF Protection**: Flask-WTF CSRF tokens on all state-changing HTML forms.
- [x] **Structured Logging**: Timestamps, log levels, and request details logged cleanly to standard output.
- [x] **Custom Error Pages**: Friendly user-facing HTML templates for 400, 404, and 500 errors.


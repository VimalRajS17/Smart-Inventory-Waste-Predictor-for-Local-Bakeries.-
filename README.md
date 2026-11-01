# Loaf Ledger — Smart Demand Forecasting & Waste Reduction for Bakeries

[![CI](https://github.com/vjvr1/loaf-ledger/actions/workflows/ci.yml/badge.svg)](https://github.com/vjvr1/loaf-ledger/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python: 3.11+](https://img.shields.io/badge/Python-3.11%2B-blue.svg)](https://www.python.org/)

An intelligent, low-cost demand forecasting and inventory optimization platform engineered for commercial and artisanal bakeries. Loaf Ledger learns from historical sales, day-of-week demand patterns, Open-Meteo weather forecasts, and national holidays to predict tomorrow's item-level demand — helping bakery owners bake the optimal quantity to cut spoilage waste and avoid costly stock-outs.

---

## 1. Architecture Overview

```mermaid
flowchart TD
    subgraph Client & UI Layer
        Browser([Web Browser / Mobile View])
        PWA["HTML5 / Mobile-First CSS / Vanilla JS\n(Plan, Today Log, Reports, Catalog, Import, Settings)"]
        Browser <--> PWA
    end

    subgraph Application & Security Gateway
        PWA <--> WSGI["Gunicorn WSGI / Flask Application Factory\n(Security Headers, CSRF, Sliding-Window Rate Limiter)"]
        WSGI <--> Auth["Session Auth & Multi-Tenant Scoping\n(Flask-Login, Argon2 / Werkzeug)"]
        WSGI <--> Blueprints["/api Blueprint & View Routes"]
    end

    subgraph Data & Persistence Layer
        DBRouter{"DATABASE_URL Router"}
        Blueprints <--> DBRouter
        DBRouter -->|Default: Local / Disk Volume| SQLite[("SQLite Engine\nbakery.db\n(Render / Railway / Fly.io Volume)")]
        DBRouter -->|Optional: DATABASE_URL| Postgres[("PostgreSQL\n(Neon / Supabase / RDS)")]
        BackupJob["flask --app app backup"] -.->|Atomic Online Backup| BackupFiles[("backups/backup_*.db")]
        SQLite -.-> BackupJob
    end

    subgraph ML & Forecasting Engine
        Blueprints <--> Predictor["ml.predictor\n(Multi-Model Dispatcher)"]
        Predictor <--> Registry[("Model Registry\n(model_registry table)")]
        Registry <--> Models[("Active Models\n(Random Forest, HistGB, Ridge)")]
        APScheduler["APScheduler (Nightly)"] --> RetrainService["ml.train (Retrain Pipeline)"]
        RetrainService --> PromotionGate{"Promotion Gate\n(Beats Baseline WAPE?)"}
        PromotionGate -->|Yes| Registry
        PromotionGate -->|No| KeepCurrent["Retain Previous Active Model"]
        RLBandit["ml.rl (Contextual Bandit)\n(RL_ENABLED=true)"] -.->|Bake Multipliers| Predictor
    end

    subgraph External Signals
        WeatherSvc["app.services.weather"] <-->|REST API (CC BY 4.0)| OpenMeteo["Open-Meteo Weather API\n(Forecasts & Archive)"]
        HolidaysSvc["python-holidays"] --> FeatEng["Feature Engineering"]
        WeatherSvc --> FeatEng
        FeatEng --> RetrainService
    end
```

---

## 2. Quickstart Guide (Run from a Clean Clone)

Follow these steps to run Loaf Ledger locally on Windows PowerShell or Linux / macOS:

### Windows PowerShell

```powershell
# 1. Clone the repository and enter the folder
git clone https://github.com/vjvr1/loaf-ledger.git
cd "loaf-ledger"

# 2. Create and activate a Python virtual environment
python -m venv venv
.\venv\Scripts\Activate.ps1

# 3. Install dependencies
pip install -r requirements.txt

# 4. Copy environment configuration
Copy-Item .env.example .env

# 5. Initialize the database and seed the demo bakery
flask --app app init-db
flask --app app seed-demo

# 6. Fetch upcoming weather forecasts from Open-Meteo
flask --app app fetch-weather --forecast-days 7

# 7. Run the test suite to verify installation
python -m pytest

# 8. Start the application
flask --app app run
```

### Linux / macOS

```bash
# 1. Clone the repository and enter the folder
git clone https://github.com/vjvr1/loaf-ledger.git
cd loaf-ledger

# 2. Create and activate a Python virtual environment
python3 -m venv venv
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Copy environment configuration
cp .env.example .env

# 5. Initialize the database and seed the demo bakery
flask --app app init-db
flask --app app seed-demo

# 6. Fetch upcoming weather forecasts from Open-Meteo
flask --app app fetch-weather --forecast-days 7

# 7. Run the test suite to verify installation
pytest -v

# 8. Start with Gunicorn (Production with automated bootstrap) or Flask (Development)
python scripts/bootstrap.py && gunicorn "app:create_app()" --workers 4 --bind 0.0.0.0:5000
# or for local dev: flask --app app run
```

Once running, navigate to `http://127.0.0.1:5000` in your web browser.  
Log in with the pre-seeded demo account:
* **Email:** `demo@example.com`
* **Password:** `demo123`

---

## 3. Screen-by-Screen User Guide

### 1. Plan & Forecast Screen (`/plan`)
The central command dashboard for bakery managers planning tomorrow's baking schedule.
* **Weather & Holiday Widget:** Displays target date conditions (temperature, precipitation mm, rain icon, holiday banner).
* **Item Baking Table:**
  * **Current Stock:** Tracked shelf inventory.
  * **Predicted Demand:** Base machine learning forecast.
  * **Safety Buffer:** Statistical dynamic buffer (80th percentile backtest error).
  * **Carryover Deduction:** Deducts safe inventory within shelf life.
  * **Recommended Bake:** Clear, actionable whole units to prepare (`bake X more`).
* **Explainability Badges:** "Rain slowdown (-15%)", "Saturday morning surge (+35%)", "Bank holiday lift".

### 2. Today's Daily Log Screen (`/today`)
A touch-friendly POS/closing log designed for fast evening entry on mobile or tablet:
* Input fields for each product: **Units Sold**, **Units Baked**, **Units Wasted**, **Leftovers**, and **Sold Out checkbox**.
* Dropdown reason tags for excess waste (e.g., "Overbaked", "Inclement weather", "Burnt batch").
* Automatically synchronizes `actual_qty` into matching forecasts for rolling accuracy tracking.

### 3. Reports & Analytics Screen (`/reports`)
Auditable business intelligence and model performance tracking:
* **Accuracy KPI Cards:** 28-day Mean Absolute Error (MAE) and Weighted Absolute Percentage Error (WAPE) vs baseline moving average.
* **Forecast vs Actual Visual Chart:** Interactive SVG chart comparing predicted bakes against actual sales.
* **Policy Simulation ("Simulation on past data"):** Historical 60-day replay comparing Same as Last Week, Forecast Only, Forecast + Buffer, and Contextual Bandit to quantify net profit, waste costs, and stock-out savings.
* **Waste & Spoilage Breakdown:** 4-week logged waste quantities and financial cost.
* **CSV Data Export:** One-click downloads for `daily_log` and `forecasts` datasets.

### 4. Items Catalog (`/items`)
Complete bakery catalog management:
* Add, edit, and deactivate catalog items.
* Configure **Shelf Life (days)** (1-day perishable items vs 2-3 day artisan sourdough).
* Configure **Unit Selling Price** and **Unit Production Cost** for gross margin and waste value calculations.

### 5. Sales CSV Import (`/import`)
Historical onboarding and batch data loading:
* Upload raw CSV exports from POS systems (Square, Toast, Shopify, Clover).
* Automatic column name matching (`date`, `item`, `sold`, `baked`, `wasted`).
* Optional toggle to automatically register newly discovered items in the catalog.
* Automatic background weather synchronization for the imported date range via Open-Meteo Archive API.

### 6. Settings & Model Retraining (`/settings`)
Administrative controls and ML lifecycle dashboard:
* Displays active model versions, training timestamps, and backtest WAPE.
* **"Retrain Now" Button:** Evaluates all candidate models (Random Forest, HistGradientBoosting, Ridge) and promotes new versions only if they pass the promotion rule (beating the naive moving average baseline).
* **Rollback Support:** One-click rollback to prior model versions via CLI or API.

---

## 4. CLI Commands Reference

Loaf Ledger provides built-in Flask CLI commands for administrative and automation tasks:

```bash
# Database management
flask --app app init-db                     # Clear and recreate database tables
flask --app app seed-demo                   # Seed demo bakery with 730 days of sales history
flask --app app reset-demo --history-days N # Full rebuild with synthetic generation & weather sync

# Automated Backups
flask --app app backup                      # Create timestamped copy in backups/
flask --app app backup --output-dir /data   # Custom destination directory

# Weather Synchronization
flask --app app fetch-weather --forecast-days 7  # Pull latest 7-day forecast from Open-Meteo

# Machine Learning Lifecycle
flask --app app train --shop-id 1           # Train models for shop 1 with 4-fold backtest
flask --app app retrain-all                 # Scan & retrain all shops with >=30 new log entries
flask --app app rollback --shop-id 1 --item-id 2 # Revert item model to previous active version

# Policy Simulation & RL Evaluation
flask --app app evaluate --days 60          # Run 60-day policy simulation and persist results
flask --app app evaluate-rl                 # Benchmark RL Contextual Bandit across 5 seeds

# Quality Assurance
python scripts/qa_smoke.py http://127.0.0.1:5000 # Standalone live health and journey smoke test
```

---

## 5. REST API Overview

Comprehensive documentation of all API routes, authentication headers, error shapes, and payloads is provided in [docs/api.md](docs/api.md).

| Endpoint | Method | Auth | Description |
|---|---|---|---|
| `/api/health` | GET | Public | Health status, database connectivity, scheduler, and RL status |
| `/auth/login` | POST | Public | User authentication with sliding-window rate limit |
| `/api/items` | GET / POST | User | List and create bakery catalog items |
| `/api/daily-log` | GET / POST | User | Retrieve and record daily sales, bakes, waste, and leftovers |
| `/api/forecast` | GET | User | Retrieve weather-informed predictions and recommended bakes |
| `/api/import/csv` | POST | User | Ingest sales history CSV with automatic weather sync |
| `/api/admin/retrain` | POST | Owner | Trigger on-demand retraining with promotion gate |
| `/api/settings/retrain-status` | GET | User | View current active model versions and performance |
| `/api/reports/accuracy` | GET | User | 28-day MAE, WAPE, and baseline comparison metrics |
| `/api/reports/waste` | GET | User | 4-week waste and leftover quantities and financial cost |
| `/api/reports/evaluation` | GET | User | 60-day historical policy simulation results |
| `/api/reports/export/daily-log` | GET | User | Download historical daily log as CSV |
| `/api/reports/export/forecasts` | GET | User | Download forecasts and actual sales as CSV |

---

## 6. Production Deployment & Persistence

Detailed production deployment guides, Gunicorn process management, systemd units, and Nginx SSL configuration are documented in [docs/DEPLOY.md](docs/DEPLOY.md).

> [!IMPORTANT]
> **Database Persistence on Cloud Hosts:**  
> Cloud container platforms (Render, Railway, Fly.io, Heroku) use ephemeral filesystems by default. To preserve your SQLite database across deployments, attach a **Persistent Disk Volume** (e.g., `/var/data` on Render or `/data` on Railway/Fly) and configure `DATABASE_PATH=/data/bakery.db`. Alternatively, configure `DATABASE_URL=postgresql://...` to connect to an external PostgreSQL database.

---

## 7. System Limitations

1. **Multi-Tenancy Model:** Each user account is currently scoped to a single bakery shop. Multi-location enterprise franchise support is planned for future releases.
2. **Weather Horizon:** High-accuracy meteorological forecasts via Open-Meteo extend up to 14 days. Predictions beyond this horizon fallback to seasonal historical medians.
3. **Model Cold-Start Period:** Machine learning models require at least 30 recorded daily logs for an item before qualifying for the promotion gate. Until 30 observations are recorded, the system automatically uses the robust 7-day same-weekday moving average baseline.
4. **Censored Demand When Sold Out:** When an item sells out before closing, actual sales represent an unknown lower bound on true customer demand. Loaf Ledger explicitly tags `sold_out=true` and applies a conservative demand adjustment in the contextual bandit environment.

---

## 8. Data Attributions & Acknowledgments

* **Weather Data:** Weather forecasts and historical meteorological records are provided by [Open-Meteo](https://open-meteo.com/) under the [Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/) license.
* **Bakery Sales Datasets:** Sample data schemas and test distributions are inspired by public point-of-sale datasets, including the [Bread Basket Bakery Dataset](https://github.com/mehreengillani/BreadBasket) by Mehreen Gillani and retail forecasting benchmarks on Kaggle under Open Data Commons / CC0.

---

## 9. License

This project is licensed under the [MIT License](LICENSE).

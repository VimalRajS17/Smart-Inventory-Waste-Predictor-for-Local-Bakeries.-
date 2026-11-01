PRAGMA foreign_keys = ON;

-- Drop legacy and current tables for a clean slate
DROP TABLE IF EXISTS evaluation_results;
DROP TABLE IF EXISTS policies;
DROP TABLE IF EXISTS model_registry;
DROP TABLE IF EXISTS forecasts;
DROP TABLE IF EXISTS weather_daily;
DROP TABLE IF EXISTS daily_log;
DROP TABLE IF EXISTS items;
DROP TABLE IF EXISTS shops;
DROP TABLE IF EXISTS users;
DROP TABLE IF EXISTS sales;
DROP TABLE IF EXISTS waste_log;

-- 0. Users
CREATE TABLE users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    email TEXT NOT NULL UNIQUE,
    password_hash TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 1. Shops / Bakery locations
CREATE TABLE shops (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    owner_user_id INTEGER REFERENCES users(id) ON DELETE CASCADE,
    name TEXT NOT NULL,
    city TEXT,
    lat REAL NOT NULL,
    lon REAL NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(owner_user_id, name)
);

-- 2. Items sold by each bakery
CREATE TABLE items (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    shop_id INTEGER NOT NULL REFERENCES shops(id) ON DELETE CASCADE,
    name TEXT NOT NULL,
    current_stock INTEGER DEFAULT 0,
    shelf_life_days INTEGER DEFAULT 1,
    unit_price REAL DEFAULT NULL,
    unit_cost REAL DEFAULT NULL,
    UNIQUE(shop_id, name)
);

-- 3. Unified daily log per item (baked, sold, wasted)
CREATE TABLE daily_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    shop_id INTEGER NOT NULL REFERENCES shops(id) ON DELETE CASCADE,
    item_id INTEGER NOT NULL REFERENCES items(id) ON DELETE CASCADE,
    date TEXT NOT NULL,
    baked INTEGER DEFAULT NULL,
    sold INTEGER DEFAULT NULL,
    wasted INTEGER DEFAULT NULL,
    leftover INTEGER DEFAULT NULL,
    sold_out INTEGER DEFAULT 0,
    reason TEXT DEFAULT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(shop_id, item_id, date)
);

-- 4. Weather & holiday tracking per shop and date
CREATE TABLE weather_daily (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    shop_id INTEGER NOT NULL REFERENCES shops(id) ON DELETE CASCADE,
    date TEXT NOT NULL,
    temperature_c REAL,
    temp_mean_c REAL,
    temp_max_c REAL,
    temp_min_c REAL,
    precip_mm REAL,
    is_rainy INTEGER DEFAULT 0,
    is_holiday INTEGER DEFAULT 0,
    source TEXT,
    fetched_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(shop_id, date)
);

-- 5. Forecast history
CREATE TABLE forecasts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    shop_id INTEGER NOT NULL REFERENCES shops(id) ON DELETE CASCADE,
    item_id INTEGER NOT NULL REFERENCES items(id) ON DELETE CASCADE,
    date TEXT NOT NULL,
    predicted_quantity REAL NOT NULL,
    actual_qty REAL DEFAULT NULL,
    weather_source TEXT,
    cold_start INTEGER DEFAULT 0,
    lag_features_stale INTEGER DEFAULT 0,
    model_version TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(shop_id, item_id, date)
);

-- 6. Model registry per shop and item
CREATE TABLE model_registry (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    shop_id INTEGER NOT NULL REFERENCES shops(id) ON DELETE CASCADE,
    item_id INTEGER NOT NULL REFERENCES items(id) ON DELETE CASCADE,
    model_type TEXT NOT NULL,
    version INTEGER NOT NULL,
    path TEXT,
    trained_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    mae REAL,
    rmse REAL,
    wape REAL,
    beats_baseline INTEGER DEFAULT 0,
    active INTEGER DEFAULT 0,
    UNIQUE(shop_id, item_id, version)
);

-- 7. RL Contextual Bandit Policies (Phase 4D)
CREATE TABLE IF NOT EXISTS policies (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    shop_id INTEGER NOT NULL REFERENCES shops(id) ON DELETE CASCADE,
    item_id INTEGER NOT NULL REFERENCES items(id) ON DELETE CASCADE,
    version INTEGER NOT NULL,
    policy_type TEXT NOT NULL,
    parameters TEXT NOT NULL,
    observations_count INTEGER DEFAULT 0,
    beat_buffer_rule INTEGER DEFAULT 0,
    active INTEGER DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(shop_id, item_id, version)
);

-- 8. 60-Day Replay Evaluation Results (Phase 5)
CREATE TABLE IF NOT EXISTS evaluation_results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    shop_id INTEGER NOT NULL REFERENCES shops(id) ON DELETE CASCADE,
    item_id INTEGER REFERENCES items(id) ON DELETE CASCADE,
    evaluated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    period_start TEXT NOT NULL,
    period_end TEXT NOT NULL,
    days_count INTEGER DEFAULT 60,
    policy_name TEXT NOT NULL,
    total_baked INTEGER DEFAULT 0,
    total_sold INTEGER DEFAULT 0,
    total_waste_units INTEGER DEFAULT 0,
    total_waste_cost REAL DEFAULT 0.0,
    total_lost_sales_units INTEGER DEFAULT 0,
    total_lost_sales_cost REAL DEFAULT 0.0,
    net_profit REAL DEFAULT 0.0,
    savings_vs_baseline REAL DEFAULT 0.0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Indices for performance
CREATE INDEX idx_shops_owner ON shops(owner_user_id);
CREATE INDEX idx_daily_log_shop_date ON daily_log(shop_id, date);
CREATE INDEX idx_daily_log_item ON daily_log(item_id);
CREATE INDEX idx_weather_shop_date ON weather_daily(shop_id, date);
CREATE INDEX idx_forecasts_shop_date ON forecasts(shop_id, date);
CREATE INDEX idx_model_registry_active ON model_registry(shop_id, item_id, active);
CREATE INDEX idx_policies_active ON policies(shop_id, item_id, active);
CREATE INDEX idx_evaluation_results_shop ON evaluation_results(shop_id, policy_name);



"""
Generates a synthetic (but realistic-looking) daily sales dataset for a small
bakery, with weekday effects, weather effects, and item-level seasonality.

Replace this with real POS/sales data once you have it — the rest of the
pipeline (train_model.py, app.py) doesn't care where the data came from as
long as the columns match: date, item, quantity_sold, temperature_c, is_rainy,
is_holiday, day_of_week.
"""

import numpy as np
import pandas as pd
from datetime import date, timedelta

np.random.seed(42)

ITEMS = {
    "Croissant": {"base": 40, "weekend_boost": 1.4, "weather_sensitive": False},
    "Sourdough Loaf": {"base": 25, "weekend_boost": 1.6, "weather_sensitive": False},
    "Chocolate Muffin": {"base": 35, "weekend_boost": 1.2, "weather_sensitive": True},
    "Cinnamon Roll": {"base": 20, "weekend_boost": 1.3, "weather_sensitive": True},
    "Cupcake": {"base": 30, "weekend_boost": 1.5, "weather_sensitive": True},
}

START_DATE = date(2024, 1, 1)
NUM_DAYS = 365 * 2  # two years of daily data

rows = []
for i in range(NUM_DAYS):
    current_date = START_DATE + timedelta(days=i)
    day_of_week = current_date.weekday()  # 0=Mon ... 6=Sun
    is_weekend = day_of_week >= 5

    # simple seasonal temperature curve + noise (Celsius)
    day_of_year = current_date.timetuple().tm_yday
    temperature_c = 22 + 8 * np.sin(2 * np.pi * (day_of_year - 80) / 365) + np.random.normal(0, 2)
    is_rainy = np.random.rand() < (0.35 if 150 < day_of_year < 250 else 0.15)  # more rain mid-year

    # a handful of fixed local "festival/holiday" dates that spike demand
    is_holiday = current_date.strftime("%m-%d") in {"01-01", "08-15", "10-02", "12-25", "12-31"}

    for item, cfg in ITEMS.items():
        demand = cfg["base"]
        if is_weekend:
            demand *= cfg["weekend_boost"]
        if is_holiday:
            demand *= 1.8
        if cfg["weather_sensitive"] and is_rainy:
            demand *= 0.8  # rainy days -> fewer walk-ins for indulgent treats
        if cfg["weather_sensitive"] and temperature_c > 30:
            demand *= 0.85  # very hot days -> fewer bakery treats bought

        # weekly slow upward trend (growing customer base) + random noise
        trend = 1 + (i / NUM_DAYS) * 0.25
        demand = demand * trend + np.random.normal(0, demand * 0.12)
        demand = max(0, round(demand))

        rows.append({
            "date": current_date.isoformat(),
            "item": item,
            "quantity_sold": int(demand),
            "day_of_week": day_of_week,
            "is_weekend": int(is_weekend),
            "temperature_c": round(temperature_c, 1),
            "is_rainy": int(is_rainy),
            "is_holiday": int(is_holiday),
        })

df = pd.DataFrame(rows)
df.to_csv("data/sales_history.csv", index=False)
print(f"Generated {len(df)} rows across {len(ITEMS)} items and {NUM_DAYS} days.")
print(df.head())

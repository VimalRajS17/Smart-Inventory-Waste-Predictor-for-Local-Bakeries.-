"""
Generates a synthetic (but realistic-looking) daily sales dataset for a small
bakery, with weekday effects, weather effects, and item-level seasonality.

Replace this with real POS/sales data once you have it — the rest of the
pipeline (train_model.py, app.py) doesn't care where the data came from as
long as the columns match: date, item, quantity_sold, day_of_week, is_weekend,
temperature_c, is_rainy, is_holiday.
"""

from datetime import date, timedelta
import os
import numpy as np
import pandas as pd

ITEMS = {
    "Croissant": {"base": 40, "weekend_boost": 1.4, "weather_sensitive": False},
    "Sourdough Loaf": {"base": 25, "weekend_boost": 1.6, "weather_sensitive": False},
    "Chocolate Muffin": {"base": 35, "weekend_boost": 1.2, "weather_sensitive": True},
    "Cinnamon Roll": {"base": 20, "weekend_boost": 1.3, "weather_sensitive": True},
    "Cupcake": {"base": 30, "weekend_boost": 1.5, "weather_sensitive": True},
}


def generate_sample_data(csv_path: str = "data/sales_history.csv", num_days: int = 730) -> pd.DataFrame:
    """Generate synthetic bakery sales history ending yesterday and spanning num_days."""
    np.random.seed(42)

    # End yesterday, start num_days earlier computed from today
    today = date.today()
    end_date = today - timedelta(days=1)
    start_date = today - timedelta(days=num_days)

    rows = []
    for i in range(num_days):
        current_date = start_date + timedelta(days=i)
        day_of_week = current_date.weekday()  # 0=Mon ... 6=Sun
        is_weekend = day_of_week >= 5

        # Seasonal temperature curve + noise (Celsius)
        day_of_year = current_date.timetuple().tm_yday
        temperature_c = 22 + 8 * np.sin(2 * np.pi * (day_of_year - 80) / 365) + np.random.normal(0, 2)
        is_rainy = np.random.rand() < (0.35 if 150 < day_of_year < 250 else 0.15)

        # Fixed local holiday dates that spike demand
        is_holiday = current_date.strftime("%m-%d") in {"01-01", "08-15", "10-02", "12-25", "12-31"}

        for item, cfg in ITEMS.items():
            demand = cfg["base"]
            if is_weekend:
                demand *= cfg["weekend_boost"]
            if is_holiday:
                demand *= 1.8
            if cfg["weather_sensitive"] and is_rainy:
                demand *= 0.8
            if cfg["weather_sensitive"] and temperature_c > 30:
                demand *= 0.85

            # Growth trend over the 730 days + noise
            trend = 1 + (i / num_days) * 0.25
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
    os.makedirs(os.path.dirname(os.path.abspath(csv_path)), exist_ok=True)
    df.to_csv(csv_path, index=False)
    print(f"Generated {len(df)} rows across {len(ITEMS)} items and {num_days} days (from {start_date} to {end_date}).")
    return df


if __name__ == "__main__":
    generate_sample_data()

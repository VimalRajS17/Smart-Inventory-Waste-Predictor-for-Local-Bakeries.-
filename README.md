# Loaf Ledger — Smart Inventory & Waste Predictor for Local Bakeries

A final-year mini project that predicts daily demand for a small bakery's
products so the owner can order/bake the right quantity — reducing both
spoilage (overstocking) and lost sales (understocking).

Unlike a plain inventory CRUD app, the core of this project is a machine
learning forecasting pipeline that learns from historical sales, day-of-week
patterns, weather, and holidays to predict tomorrow's demand per item.

## Problem statement

Small food businesses lose revenue daily either by overstocking perishables
that spoil, or understocking items that sell out. This project builds a
lightweight, low-cost forecasting tool aimed at independent bakeries/grocers
who can't afford enterprise tools like Afresh or Shelf Engine, which are
built for large chains and require POS integration.

## Project status (Review 1 — Prototype)

- [x] Synthetic dataset generator (to be replaced with a real POS export or
      a public dataset such as the Kaggle "Bread Basket" bakery dataset)
- [x] Feature engineering: lag features, 7-day rolling average, weekday,
      weather, holiday flags
- [x] Three forecasting models built and compared: Moving Average (baseline),
      Linear Regression, Random Forest
- [x] Chronological train/test split (no data leakage) + MAE/RMSE/MAPE
      evaluation
- [x] Flask backend serving predictions via a REST endpoint
- [x] Minimal dashboard: predicted demand, suggested order quantity per
      item, waste log, sale/waste entry forms
- [ ] Swap synthetic data for a real dataset
- [ ] Add weather API integration (currently a manual/static input)
- [ ] Model retraining pipeline triggered by new logged sales
- [ ] Auth / multi-bakery support

## Architecture

```
                ┌─────────────────────┐
                │   sales_history.csv  │  (or real POS export)
                └──────────┬───────────┘
                           │
                    train_model.py
              (feature engineering + model
               comparison: MA / LR / RF)
                           │
                           ▼
              models/best_model.joblib
                           │
                           ▼
            models/predictor.py  ──────►  app.py (Flask)
                                              │
                             ┌────────────────┼─────────────────┐
                             ▼                ▼                 ▼
                     dashboard.html   /api/predict (JSON)   /log-sale,
                     (predicted        for external/mobile   /log-waste
                     demand + order     use                  (SQLite via
                     suggestions)                             database.py)
```

## Model results (on synthetic data, chronological split)

| Model              | MAE  | RMSE | MAPE   |
|---------------------|------|------|--------|
| Moving Average (7d) | 7.26 | 9.32 | 17.74% |
| Linear Regression   | 4.18 | 5.45 | 10.09% |
| Random Forest       | 4.14 | 5.32 |  9.82% |

Random Forest is selected and saved automatically by `train_model.py`.
Numbers will change once trained on a real bakery's data — the point of
this table for the review is to show the *comparison methodology*
(chronological split, multiple metrics, baseline vs ML), not the exact
figures.

## Tech stack

- **Backend**: Python, Flask
- **ML**: scikit-learn (Linear Regression, Random Forest), pandas, numpy
- **Storage**: SQLite (sales log, waste log, item stock)
- **Frontend**: server-rendered HTML/CSS (no JS framework, kept simple for
  a mini project — can be swapped for a JS dashboard later)

## Setup

```bash
git clone <your-repo-url>
cd bakery-inventory-predictor
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt

# 1. Generate the sample dataset (or drop your own sales_history.csv here)
python3 data/generate_sample_data.py

# 2. Train the models and save the best one
python3 train_model.py

# 3. Run the app
python3 app.py
```

Then open `http://127.0.0.1:5000` in your browser.

### API usage

```
GET /api/predict?item=Croissant&date=2026-09-10&temperature_c=28&is_rainy=0
```

Returns:
```json
{"item": "Croissant", "date": "2026-09-10", "predicted_quantity": 45.3}
```

## Dataset

The included `data/generate_sample_data.py` creates a synthetic two-year
daily sales history for five bakery items with realistic weekday, weather,
and holiday effects, purely so the pipeline is demoable without real data.
To replace it with real data, provide a CSV with these columns:

```
date, item, quantity_sold, day_of_week, is_weekend, temperature_c, is_rainy, is_holiday
```

Public datasets that work well as substitutes:
- [Bread Basket bakery dataset](https://github.com/mehreengillani/BreadBasket) — real transaction-level data from an Edinburgh bakery
- [Kaggle Store Item Demand Forecasting](https://www.kaggle.com/c/demand-forecasting-kernels-only/) — 5 years of store-item sales

## Related work / research grounding

- Hübner et al., *"Machine-learning-based demand forecasting against food
  waste: Life cycle environmental impacts and benefits of a bakery case
  study,"* Journal of Industrial Ecology, 2024 — real-world bakery ML
  forecasting service reporting a 30% reduction in bakery returns.
- Comparative study of ARIMA/SARIMA vs Random Forest/XGBoost/LightGBM vs
  LSTM for food-waste-oriented demand forecasting — *Journal of Science,
  Innovation & Social Impact*, 2026.
- Literature review on ML for perishable food demand forecasting in retail
  supermarkets (Priyadarshi et al. 2019; Arunraj et al. 2016, cited
  therein).

Commercial products solving a related problem at enterprise scale: **Afresh**
and **Shelf Engine** (grocery chain-level automated ordering), **Winnow**
and **KITRO** (camera-based waste measurement). This project differentiates
by targeting small independent bakeries with no POS integration
requirement and an explainable, low-cost forecasting approach.

## Repository structure

```
bakery-inventory-predictor/
├── app.py                     # Flask app: routes, dashboard, API
├── database.py                 # SQLite schema + helpers
├── train_model.py              # Feature engineering + model training/comparison
├── data/
│   ├── generate_sample_data.py # Synthetic dataset generator
│   └── sales_history.csv       # Generated sample data (gitignored if you swap in real data)
├── models/
│   ├── predictor.py            # Loads trained model, builds features, predicts
│   ├── best_model.joblib       # Saved trained model (generated)
│   ├── feature_columns.joblib  # Feature column order (generated)
│   └── metrics.json            # Model comparison results (generated)
├── templates/
│   └── dashboard.html
├── static/
│   └── style.css
├── requirements.txt
└── README.md
```

## License

MIT — free to use and adapt for academic purposes.

"""Backward-compatible shim delegating to ml.predictor."""

from ml.predictor import (
    predict_demand,
    predict_item_demand,
    predict_all_items,
    get_seasonal_avg_temp,
)

__all__ = [
    "predict_demand",
    "predict_item_demand",
    "predict_all_items",
    "get_seasonal_avg_temp",
]

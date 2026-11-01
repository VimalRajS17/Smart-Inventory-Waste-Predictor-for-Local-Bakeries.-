"""Machine learning package for Loaf Ledger."""

from ml.features import FEATURE_COLUMNS, build_features, build_inference_features
from ml.prepare import load_training_data
from ml.train import create_model, train_shop_models, train_command
from ml.evaluate import compute_metrics, evaluate_item_rolling_origin, print_comparison_tables
from ml.registry import get_active_model, promote_and_register_item_model
from ml.predictor import predict_item_demand, predict_all_items, predict_demand

__all__ = [
    "FEATURE_COLUMNS",
    "build_features",
    "build_inference_features",
    "load_training_data",
    "create_model",
    "train_shop_models",
    "train_command",
    "compute_metrics",
    "evaluate_item_rolling_origin",
    "print_comparison_tables",
    "get_active_model",
    "promote_and_register_item_model",
    "predict_item_demand",
    "predict_all_items",
    "predict_demand",
]



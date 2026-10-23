"""
Model training entry point.
Trains Random Forest baseline, XGBoost classifier, 1D CNN, and Isolation Forest
on leakage-free split datasets.
"""

import argparse
from pathlib import Path
import sys
import numpy as np
import pandas as pd

root_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir / "src"))

from predictive_maintenance.config import settings
from predictive_maintenance.logging_config import setup_logger
from predictive_maintenance.models.anomaly_detection import VibrationAnomalyDetector
from predictive_maintenance.models.cnn_1d import CNN1DClassifier
from predictive_maintenance.models.random_forest import RandomForestBaseline
from predictive_maintenance.models.xgboost_model import XGBoostClassifierModel

logger = setup_logger("train")


def train_models(models_to_train: list):
    proc_dir = root_dir / "data" / "processed"
    artifacts_dir = root_dir / "models" / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load feature datasets
    train_feat_path = proc_dir / "train_features.csv"
    val_feat_path = proc_dir / "val_features.csv"

    if not train_feat_path.exists():
        raise FileNotFoundError(
            f"Processed training features not found at {train_feat_path}. "
            "Please run 'python scripts/prepare_data.py' first."
        )

    train_df = pd.read_csv(train_feat_path)
    val_df = pd.read_csv(val_feat_path)

    # Separate feature columns from metadata
    meta_cols = ["recording_id", "label", "load_percent", "window_idx"]
    feature_cols = [c for c in train_df.columns if c not in meta_cols]

    X_train_df = train_df[feature_cols]
    y_train = train_df["label"]

    X_val_df = val_df[feature_cols]
    y_val = val_df["label"]

    cfg = settings.model_config

    # --- Train Random Forest ---
    if "random_forest" in models_to_train or "all" in models_to_train:
        logger.info(">>> Training Random Forest Baseline...")
        rf_cfg = cfg.get("random_forest", {})
        rf = RandomForestBaseline(
            n_estimators=rf_cfg.get("n_estimators", 150),
            max_depth=rf_cfg.get("max_depth", 12),
            min_samples_split=rf_cfg.get("min_samples_split", 4),
            min_samples_leaf=rf_cfg.get("min_samples_leaf", 2),
            class_weight=rf_cfg.get("class_weight", "balanced"),
            random_state=rf_cfg.get("random_state", 42),
        )
        rf.fit(X_train_df, y_train, feature_names=feature_cols)
        rf.save(artifacts_dir / "random_forest.joblib")

    # --- Train XGBoost ---
    if "xgboost" in models_to_train or "all" in models_to_train:
        logger.info(">>> Training XGBoost Classifier...")
        xgb_cfg = cfg.get("xgboost", {})
        xgb = XGBoostClassifierModel(
            n_estimators=xgb_cfg.get("n_estimators", 200),
            max_depth=xgb_cfg.get("max_depth", 6),
            learning_rate=xgb_cfg.get("learning_rate", 0.05),
            subsample=xgb_cfg.get("subsample", 0.8),
            colsample_bytree=xgb_cfg.get("colsample_bytree", 0.8),
            random_state=xgb_cfg.get("random_state", 42),
            early_stopping_rounds=xgb_cfg.get("early_stopping_rounds", 15),
        )
        xgb.fit(X_train_df, y_train, X_val=X_val_df, y_val=y_val, feature_names=feature_cols)
        xgb.save(artifacts_dir / "xgboost.joblib")

    # --- Train Anomaly Detector ---
    if "anomaly" in models_to_train or "all" in models_to_train:
        logger.info(">>> Training Vibration Anomaly Detector (Healthy baseline only)...")
        ad_cfg = cfg.get("anomaly_detection", {})
        healthy_train_df = train_df[train_df["label"] == "Healthy"][feature_cols]
        detector = VibrationAnomalyDetector(
            contamination=ad_cfg.get("contamination", 0.05),
            n_estimators=ad_cfg.get("n_estimators", 100),
            random_state=ad_cfg.get("random_state", 42),
        )
        detector.fit_healthy_only(healthy_train_df, feature_names=feature_cols, threshold_percentile=95.0)
        detector.save(artifacts_dir / "anomaly_detector.joblib")

    # --- Train 1D CNN ---
    if "cnn" in models_to_train or "all" in models_to_train:
        logger.info(">>> Training 1D CNN on Raw Vibration Waveforms...")
        train_win_path = proc_dir / "train_windows.npy"
        val_win_path = proc_dir / "val_windows.npy"

        if not train_win_path.exists():
            raise FileNotFoundError(f"Training windows not found at {train_win_path}")

        train_windows = np.load(train_win_path)
        val_windows = np.load(val_win_path)

        cnn_cfg = cfg.get("cnn_1d", {})
        cnn = CNN1DClassifier(
            num_classes=2,
            input_channels=1,
            learning_rate=cnn_cfg.get("learning_rate", 0.001),
            weight_decay=float(cnn_cfg.get("weight_decay", 1e-4)),
            batch_size=cnn_cfg.get("batch_size", 32),
            max_epochs=cnn_cfg.get("max_epochs", 25),
            patience=cnn_cfg.get("early_stopping_patience", 6),
            device=cnn_cfg.get("device", "auto"),
        )
        cnn.fit(
            X_train=train_windows,
            y_train=train_df["label"].values,
            X_val=val_windows,
            y_val=val_df["label"].values,
            class_names=["Healthy", "BrokenTooth"],
        )
        cnn.save(artifacts_dir / "cnn_1d.pt")

    logger.info(">>> Model training pipeline completed successfully!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train predictive maintenance models.")
    parser.add_argument(
        "--models",
        nargs="+",
        default=["all"],
        choices=["all", "random_forest", "xgboost", "cnn", "anomaly"],
        help="List of models to train",
    )
    args = parser.parse_args()
    train_models(args.models)

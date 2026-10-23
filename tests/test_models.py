"""
Unit tests for model training, forward passes, and serialization.
Uses synthetic fixtures to verify model logic without requiring full dataset training.
"""

from pathlib import Path
import sys
import tempfile
import numpy as np
import pandas as pd
import pytest
import torch

root_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir / "src"))

from predictive_maintenance.models.anomaly_detection import VibrationAnomalyDetector
from predictive_maintenance.models.cnn_1d import CNN1DClassifier, Conv1DNet
from predictive_maintenance.models.random_forest import RandomForestBaseline
from predictive_maintenance.models.xgboost_model import XGBoostClassifierModel


def test_random_forest_fit_predict_save():
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create small synthetic dataset
        X = pd.DataFrame({
            "rms": [1.0, 1.2, 5.0, 5.5],
            "peak_to_peak": [3.0, 3.2, 12.0, 13.0],
            "kurtosis": [0.1, 0.2, 4.5, 4.8],
        })
        y = pd.Series(["Healthy", "Healthy", "BrokenTooth", "BrokenTooth"])

        rf = RandomForestBaseline(n_estimators=10, random_state=42)
        rf.fit(X, y)

        preds = rf.predict(X)
        assert len(preds) == 4

        # Save and reload
        art_path = Path(tmpdir) / "rf.joblib"
        rf.save(art_path)

        loaded_rf = RandomForestBaseline.load(art_path)
        loaded_preds = loaded_rf.predict(X)
        assert np.array_equal(preds, loaded_preds)


def test_xgboost_fit_predict_save():
    with tempfile.TemporaryDirectory() as tmpdir:
        X = pd.DataFrame({
            "rms": [1.0, 1.2, 5.0, 5.5],
            "peak_to_peak": [3.0, 3.2, 12.0, 13.0],
        })
        y = pd.Series(["Healthy", "Healthy", "BrokenTooth", "BrokenTooth"])

        xgb = XGBoostClassifierModel(n_estimators=10, random_state=42)
        xgb.fit(X, y)

        preds = xgb.predict(X)
        assert len(preds) == 4

        art_path = Path(tmpdir) / "xgb.joblib"
        xgb.save(art_path)

        loaded_xgb = XGBoostClassifierModel.load(art_path)
        assert np.array_equal(preds, loaded_xgb.predict(X))


def test_cnn_1d_architecture_forward():
    net = Conv1DNet(num_classes=2, input_channels=1)
    # Batch size 2, 1 channel, 2048 sequence length
    dummy_input = torch.randn(2, 1, 2048)
    logits = net(dummy_input)
    assert logits.shape == (2, 2)


def test_cnn_1d_fit_predict():
    X = np.random.randn(8, 2048).astype(np.float32)
    y = np.array(["Healthy", "Healthy", "Healthy", "Healthy", "BrokenTooth", "BrokenTooth", "BrokenTooth", "BrokenTooth"])

    cnn = CNN1DClassifier(num_classes=2, max_epochs=2, batch_size=4, device="cpu")
    cnn.fit(X, y)
    preds = cnn.predict(X)
    assert len(preds) == 8


def test_anomaly_detector_healthy_baseline():
    rng = np.random.RandomState(42)
    X_healthy = pd.DataFrame({
        "rms": rng.normal(1.0, 0.05, 30),
        "kurtosis": rng.normal(0.0, 0.02, 30),
    })

    detector = VibrationAnomalyDetector(contamination=0.1, n_estimators=50, random_state=42)
    detector.fit_healthy_only(X_healthy)

    # Extreme outlier (Broken tooth impact)
    outlier = pd.DataFrame({"rms": [8.5], "kurtosis": [9.0]})
    is_anom = detector.is_anomaly(outlier)
    assert is_anom[0] == True

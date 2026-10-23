"""
Unsupervised Anomaly Detection module using Isolation Forest.
Trained strictly on healthy vibration signals to establish normal operating baselines
and output anomaly severity scores.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from ..logging_config import setup_logger

logger = setup_logger(__name__)


class VibrationAnomalyDetector:
    """Isolation Forest anomaly detector for mechanical vibration features."""

    def __init__(
        self,
        contamination: float = 0.05,
        n_estimators: int = 100,
        random_state: int = 42,
    ):
        self.contamination = contamination
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.model = IsolationForest(
            contamination=contamination,
            n_estimators=n_estimators,
            random_state=random_state,
        )
        self.feature_names: List[str] = []
        self.threshold: float = 0.0
        self.metadata: Dict[str, Any] = {}

    def fit_healthy_only(
        self,
        X_healthy: pd.DataFrame,
        feature_names: Optional[List[str]] = None,
        threshold_percentile: float = 95.0,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> "VibrationAnomalyDetector":
        """
        Fits the detector strictly on confirmed Healthy training recordings.
        Establishes an anomaly score threshold at the specified percentile.
        """
        if isinstance(X_healthy, np.ndarray):
            self.feature_names = feature_names or [f"feat_{i}" for i in range(X_healthy.shape[1])]
            X = pd.DataFrame(X_healthy, columns=self.feature_names)
        else:
            self.feature_names = list(X_healthy.columns)
            X = X_healthy

        self.model.fit(X[self.feature_names])

        # Raw scores from decision_function: lower means more anomalous
        scores = -self.model.decision_function(X[self.feature_names])
        self.threshold = float(np.percentile(scores, threshold_percentile))
        self.metadata = metadata or {}
        self.metadata["threshold"] = self.threshold
        self.metadata["threshold_percentile"] = threshold_percentile

        logger.info(
            f"AnomalyDetector fitted on {len(X)} healthy samples. "
            f"Established anomaly threshold: {self.threshold:.4f} (p{threshold_percentile})"
        )
        return self

    def score_anomaly(self, X: pd.DataFrame) -> np.ndarray:
        """
        Returns anomaly severity scores.
        Higher score indicates greater deviation from healthy operating profile.
        """
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X, columns=self.feature_names)
        # Negative decision function: positive values above threshold are anomalous
        return -self.model.decision_function(X[self.feature_names])

    def is_anomaly(self, X: pd.DataFrame) -> np.ndarray:
        """Returns boolean array: True if score exceeds established healthy threshold."""
        scores = self.score_anomaly(X)
        return scores > self.threshold

    def save(self, filepath: Path) -> Path:
        """Saves artifact to disk."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "model": self.model,
            "feature_names": self.feature_names,
            "threshold": self.threshold,
            "contamination": self.contamination,
            "metadata": self.metadata,
        }
        joblib.dump(payload, filepath)
        logger.info(f"AnomalyDetector saved to {filepath}")
        return filepath

    @classmethod
    def load(cls, filepath: Path) -> "VibrationAnomalyDetector":
        """Loads detector artifact."""
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"Detector artifact not found: {filepath}")
        payload = joblib.load(filepath)
        instance = cls(
            contamination=payload.get("contamination", 0.05),
        )
        instance.model = payload["model"]
        instance.feature_names = payload["feature_names"]
        instance.threshold = payload["threshold"]
        instance.metadata = payload.get("metadata", {})
        return instance

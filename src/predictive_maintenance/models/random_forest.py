"""
Random Forest classifier for gearbox condition classification.
Baseline model consuming tabular time/frequency vibration features.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, f1_score
from ..logging_config import setup_logger

logger = setup_logger(__name__)


class RandomForestBaseline:
    """Random Forest condition classifier with feature importance and probability calibration."""

    def __init__(
        self,
        n_estimators: int = 150,
        max_depth: Optional[int] = 12,
        min_samples_split: int = 4,
        min_samples_leaf: int = 2,
        class_weight: str = "balanced",
        random_state: int = 42,
        n_jobs: int = -1,
    ):
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            class_weight=class_weight,
            random_state=random_state,
            n_jobs=n_jobs,
        )
        self.feature_names: List[str] = []
        self.classes: List[str] = []
        self.metadata: Dict[str, Any] = {}

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        feature_names: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> "RandomForestBaseline":
        """Trains the Random Forest model on vibration features."""
        if isinstance(X_train, np.ndarray):
            self.feature_names = feature_names or [f"feat_{i}" for i in range(X_train.shape[1])]
            X = pd.DataFrame(X_train, columns=self.feature_names)
        else:
            self.feature_names = list(X_train.columns)
            X = X_train

        self.classes = sorted(list(np.unique(y_train)))
        self.model.fit(X, y_train)
        self.metadata = metadata or {}
        logger.info(f"RandomForest trained successfully on {len(X)} samples across classes: {self.classes}")
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predicts class labels."""
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X, columns=self.feature_names)
        return self.model.predict(X[self.feature_names])

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Returns class probabilities."""
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X, columns=self.feature_names)
        return self.model.predict_proba(X[self.feature_names])

    def get_feature_importances(self) -> Dict[str, float]:
        """Returns sorted feature importances."""
        importances = self.model.feature_importances_
        feat_imp = {feat: float(imp) for feat, imp in zip(self.feature_names, importances)}
        return dict(sorted(feat_imp.items(), key=lambda x: x[1], reverse=True))

    def save(self, filepath: Path) -> Path:
        """Serializes model and metadata to disk."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "model": self.model,
            "feature_names": self.feature_names,
            "classes": self.classes,
            "metadata": self.metadata,
        }
        joblib.dump(payload, filepath)
        logger.info(f"RandomForest artifact saved to {filepath}")
        return filepath

    @classmethod
    def load(cls, filepath: Path) -> "RandomForestBaseline":
        """Loads serialized model and metadata."""
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"Artifact not found: {filepath}")
        payload = joblib.load(filepath)
        instance = cls()
        instance.model = payload["model"]
        instance.feature_names = payload["feature_names"]
        instance.classes = payload["classes"]
        instance.metadata = payload.get("metadata", {})
        return instance

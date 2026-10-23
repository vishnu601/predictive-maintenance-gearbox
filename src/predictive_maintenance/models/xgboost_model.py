"""
XGBoost condition classifier for mechanical fault diagnosis.
Trained on engineered vibration features with validation-guided early stopping.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional
import joblib
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from ..logging_config import setup_logger

logger = setup_logger(__name__)


class XGBoostClassifierModel:
    """XGBoost classifier with feature tracking and artifact persistence."""

    def __init__(
        self,
        n_estimators: int = 200,
        max_depth: int = 6,
        learning_rate: float = 0.05,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        random_state: int = 42,
        early_stopping_rounds: int = 15,
        eval_metric: str = "logloss",
    ):
        self.early_stopping_rounds = early_stopping_rounds
        self.model = XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            random_state=random_state,
            eval_metric=eval_metric,
        )
        self.feature_names: List[str] = []
        self.classes: List[str] = []
        self.class_to_idx: Dict[str, int] = {}
        self.idx_to_class: Dict[int, str] = {}
        self.metadata: Dict[str, Any] = {}

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
        feature_names: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> "XGBoostClassifierModel":
        """Fits XGBoost with early stopping on validation split."""
        if isinstance(X_train, np.ndarray):
            self.feature_names = feature_names or [f"feat_{i}" for i in range(X_train.shape[1])]
            X_tr = pd.DataFrame(X_train, columns=self.feature_names)
        else:
            self.feature_names = list(X_train.columns)
            X_tr = X_train

        self.classes = sorted(list(np.unique(y_train)))
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        self.idx_to_class = {idx: cls for cls, idx in self.class_to_idx.items()}

        y_tr_num = np.array([self.class_to_idx[y] for y in y_train])

        eval_set = None
        if X_val is not None and y_val is not None:
            if isinstance(X_val, np.ndarray):
                X_v = pd.DataFrame(X_val, columns=self.feature_names)
            else:
                X_v = X_val
            y_v_num = np.array([self.class_to_idx[y] for y in y_val])
            eval_set = [(X_tr[self.feature_names], y_tr_num), (X_v[self.feature_names], y_v_num)]

        if eval_set is not None and self.early_stopping_rounds:
            self.model.set_params(early_stopping_rounds=self.early_stopping_rounds)
        else:
            self.model.set_params(early_stopping_rounds=None)

        self.model.fit(
            X_tr[self.feature_names],
            y_tr_num,
            eval_set=eval_set,
            verbose=False,
        )
        self.metadata = metadata or {}
        logger.info(f"XGBoost trained on {len(X_tr)} samples. Best iteration: {getattr(self.model, 'best_iteration', 'N/A')}")
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predicts class labels."""
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X, columns=self.feature_names)
        num_preds = self.model.predict(X[self.feature_names])
        return np.array([self.idx_to_class[p] for p in num_preds])

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
            "class_to_idx": self.class_to_idx,
            "idx_to_class": self.idx_to_class,
            "metadata": self.metadata,
        }
        joblib.dump(payload, filepath)
        logger.info(f"XGBoost artifact saved to {filepath}")
        return filepath

    @classmethod
    def load(cls, filepath: Path) -> "XGBoostClassifierModel":
        """Loads serialized model and metadata."""
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"Artifact not found: {filepath}")
        payload = joblib.load(filepath)
        instance = cls()
        instance.model = payload["model"]
        instance.feature_names = payload["feature_names"]
        instance.classes = payload["classes"]
        instance.class_to_idx = payload["class_to_idx"]
        instance.idx_to_class = payload["idx_to_class"]
        instance.metadata = payload.get("metadata", {})
        return instance

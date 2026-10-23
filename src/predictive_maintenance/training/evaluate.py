"""
Model evaluation and benchmarking module.
Computes standardized classification metrics, confusion matrices, and inference latency.
"""

from pathlib import Path
import time
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from ..logging_config import setup_logger

logger = setup_logger(__name__)


def evaluate_classifier(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    classes: List[str],
    model_name: str,
    latency_ms: Optional[float] = None,
    model_size_kb: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Computes a comprehensive performance report for a condition classifier.
    """
    y_t = np.asarray(y_true, dtype=str)
    y_p = np.asarray(y_pred, dtype=str)

    acc = float(accuracy_score(y_t, y_p))
    bal_acc = float(balanced_accuracy_score(y_t, y_p))
    macro_f1 = float(f1_score(y_t, y_p, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(y_t, y_p, average="weighted", zero_division=0))

    # Per-class metrics
    class_metrics = {}
    for cls in classes:
        p = float(precision_score(y_t == cls, y_p == cls, zero_division=0))
        r = float(recall_score(y_t == cls, y_p == cls, zero_division=0))
        f = float(f1_score(y_t == cls, y_p == cls, zero_division=0))
        class_metrics[cls] = {"precision": p, "recall": r, "f1": f}

    cm = confusion_matrix(y_t, y_p, labels=classes)
    with np.errstate(divide="ignore", invalid="ignore"):
        cm_norm = np.nan_to_num(cm.astype("float") / cm.sum(axis=1)[:, np.newaxis])

    report = {
        "model_name": model_name,
        "accuracy": acc,
        "balanced_accuracy": bal_acc,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "per_class": class_metrics,
        "confusion_matrix": cm.tolist(),
        "normalized_confusion_matrix": cm_norm.tolist(),
        "classes": classes,
        "inference_latency_ms": latency_ms,
        "model_size_kb": model_size_kb,
    }
    return report


def benchmark_inference_latency(
    model: Any,
    sample_input: Any,
    n_iterations: int = 100,
) -> float:
    """
    Measures average inference latency in milliseconds over repeated calls.
    """
    # Warmup
    for _ in range(5):
        _ = model.predict(sample_input)

    latencies = []
    for _ in range(n_iterations):
        t0 = time.perf_counter()
        _ = model.predict(sample_input)
        latencies.append((time.perf_counter() - t0) * 1000.0)

    return float(np.mean(latencies))

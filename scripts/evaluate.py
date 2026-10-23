"""
Model evaluation entry point.
Evaluates all trained models on held-out test groups, benchmarks latency,
and exports comprehensive performance comparisons and confusion matrices.
"""

import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd

root_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir / "src"))

from predictive_maintenance.logging_config import setup_logger
from predictive_maintenance.models.cnn_1d import CNN1DClassifier
from predictive_maintenance.models.random_forest import RandomForestBaseline
from predictive_maintenance.models.xgboost_model import XGBoostClassifierModel
from predictive_maintenance.training.evaluate import benchmark_inference_latency, evaluate_classifier

logger = setup_logger("evaluate")


def run_evaluation():
    proc_dir = root_dir / "data" / "processed"
    artifacts_dir = root_dir / "models" / "artifacts"

    test_feat_path = proc_dir / "test_features.csv"
    test_win_path = proc_dir / "test_windows.npy"

    if not test_feat_path.exists() or not test_win_path.exists():
        raise FileNotFoundError("Held-out test dataset not found in data/processed/. Run prepare_data.py first.")

    test_df = pd.read_csv(test_feat_path)
    test_windows = np.load(test_win_path)

    meta_cols = ["recording_id", "label", "load_percent", "window_idx"]
    feature_cols = [c for c in test_df.columns if c not in meta_cols]
    y_test = test_df["label"].values
    classes = ["Healthy", "BrokenTooth"]

    comparison_results = []

    # 1. Evaluate Random Forest
    rf_path = artifacts_dir / "random_forest.joblib"
    if rf_path.exists():
        logger.info("Evaluating Random Forest...")
        rf = RandomForestBaseline.load(rf_path)
        sample_input = test_df[feature_cols].iloc[:1]
        lat = benchmark_inference_latency(rf, sample_input, n_iterations=50)
        size_kb = rf_path.stat().st_size / 1024.0
        preds = rf.predict(test_df[feature_cols])
        res = evaluate_classifier(y_test, preds, classes, "Random Forest", latency_ms=lat, model_size_kb=size_kb)
        comparison_results.append(res)
    else:
        logger.warning(f"Random Forest artifact not found at {rf_path}")

    # 2. Evaluate XGBoost
    xgb_path = artifacts_dir / "xgboost.joblib"
    if xgb_path.exists():
        logger.info("Evaluating XGBoost...")
        xgb = XGBoostClassifierModel.load(xgb_path)
        sample_input = test_df[feature_cols].iloc[:1]
        lat = benchmark_inference_latency(xgb, sample_input, n_iterations=50)
        size_kb = xgb_path.stat().st_size / 1024.0
        preds = xgb.predict(test_df[feature_cols])
        res = evaluate_classifier(y_test, preds, classes, "XGBoost", latency_ms=lat, model_size_kb=size_kb)
        comparison_results.append(res)
    else:
        logger.warning(f"XGBoost artifact not found at {xgb_path}")

    # 3. Evaluate 1D CNN
    cnn_path = artifacts_dir / "cnn_1d.pt"
    if cnn_path.exists():
        logger.info("Evaluating 1D CNN...")
        cnn = CNN1DClassifier.load(cnn_path, device="cpu")
        sample_win = test_windows[:1]
        lat = benchmark_inference_latency(cnn, sample_win, n_iterations=50)
        size_kb = cnn_path.stat().st_size / 1024.0
        preds = cnn.predict(test_windows)
        res = evaluate_classifier(y_test, preds, classes, "1D CNN", latency_ms=lat, model_size_kb=size_kb)
        comparison_results.append(res)
    else:
        logger.warning(f"1D CNN checkpoint not found at {cnn_path}")

    # Save summary
    with open(artifacts_dir / "model_comparison.json", "w", encoding="utf-8") as f:
        json.dump(comparison_results, f, indent=2)

    # Create comparison table
    summary_rows = []
    for r in comparison_results:
        summary_rows.append({
            "Model": r["model_name"],
            "Accuracy": f"{r['accuracy']:.4f}",
            "Balanced Accuracy": f"{r['balanced_accuracy']:.4f}",
            "Macro F1": f"{r['macro_f1']:.4f}",
            "Weighted F1": f"{r['weighted_f1']:.4f}",
            "Healthy F1": f"{r['per_class']['Healthy']['f1']:.4f}",
            "BrokenTooth F1": f"{r['per_class']['BrokenTooth']['f1']:.4f}",
            "Latency (ms)": f"{r['inference_latency_ms']:.2f}",
            "Size (KB)": f"{r['model_size_kb']:.1f}",
        })
    df_summary = pd.DataFrame(summary_rows)
    df_summary.to_csv(artifacts_dir / "model_comparison.csv", index=False)

    print("\n" + "=" * 70)
    print("MODEL COMPARISON REPORT (Held-Out Test Groups)")
    print("=" * 70)
    print(df_summary.to_string(index=False))
    print("=" * 70 + "\n")


if __name__ == "__main__":
    run_evaluation()

# Model Artifacts Directory

This directory stores trained machine learning and deep learning artifacts:
- `random_forest.joblib`: Trained Scikit-learn Random Forest model, feature schema, class labels, and metadata.
- `xgboost.joblib`: Trained XGBoost classifier with early stopping state.
- `cnn_1d.pt`: PyTorch checkpoint containing 1D CNN weights, architecture parameters, and validation F1 history.
- `anomaly_detector.joblib`: Trained Isolation Forest model calibrated strictly on healthy baseline data.
- `model_comparison.json` & `model_comparison.csv`: Comprehensive evaluation benchmarks across all models on held-out test groups.

# Build State — AI Predictive Maintenance Platform

> **Snapshot Date:** 2026-10-23  
> **Python:** 3.11.15 (Apple Silicon / arm64)  
> **Project Root:** `predictive-maintenance/` (`/Users/vishnuganesh/dev/predictive analyst gearbox`)  
> **Virtual Environment:** `.venv/` (activated via `source .venv/bin/activate`)

---

## 1. Overall Completion Status

| Phase | Description | Status |
|-------|-------------|--------|
| Phase 1 | Repository scaffolding & environment | ✅ Complete |
| Phase 2 | Dataset download & extraction | ✅ Complete (OpenEI data downloaded & extracted) |
| Phase 3 | Signal processing & feature engineering | ✅ Complete |
| Phase 4 | Data preparation & leakage-free splitting | ✅ Complete (1946 windows extracted, group-split) |
| Phase 5 | Model training (RF, XGBoost, CNN, Anomaly) | ✅ Complete (All 4 models trained & saved) |
| Phase 6 | Model evaluation & benchmarking | ✅ Complete (Evaluated on held-out test groups) |
| Phase 7 | Unified inference engine | ✅ Complete |
| Phase 8 | Monitoring state machine & alerts | ✅ Complete |
| Phase 9 | FastAPI REST API | ✅ Complete |
| Phase 10 | Streamlit dashboard (4 pages) | ✅ Complete |
| Phase 11 | Test suite | ✅ Complete (32/32 tests passing, 100% green) |
| Phase 12 | Documentation | ✅ Complete (6 docs + README) |

---

## 2. Bugs Resolved

### ✅ BUG-001: `Settings` class incompatible with pydantic-settings v2
- **File:** `src/predictive_maintenance/config.py`
- **Fix:** Converted `Settings` to a clean standard Python class loading YAML files from `configs/`, removing `pydantic-settings` metaclass conflicts with `@property`.

### ✅ BUG-002: XGBoost early stopping crash without validation split
- **File:** `src/predictive_maintenance/models/xgboost_model.py`
- **Fix:** Dynamically configure `early_stopping_rounds` on `self.model` only when an `eval_set` is provided to `fit()`.

### ✅ BUG-003: macOS Apple Silicon OpenMP duplicate runtime conflict
- **File:** `src/predictive_maintenance/__init__.py`
- **Fix:** Added `KMP_DUPLICATE_LIB_OK=TRUE` and `OMP_NUM_THREADS=1` defaults at package initialization to eliminate segfaults when PyTorch and XGBoost coexist.

### ✅ BUG-004: Monitoring state machine recovery logic prematurely cleared recovery mode
- **File:** `src/predictive_maintenance/monitoring/rules.py`
- **Fix:** Added `_in_recovery` per-machine flag so that machines recovering from `FAULT` strictly require `recovery_confirmation_windows` (e.g. M=3 or 5) consecutive healthy windows before resetting to `NORMAL`.

---

## 3. Dataset & Model Artifact State

| Item | Status | Path | Details |
|------|--------|------|---------|
| Raw ZIP archive | ✅ Present | `data/raw/gearboxdata.zip` | 36.8 MB |
| Extracted recordings | ✅ Present | `data/raw/gearbox_extracted/` | 10 Healthy, 10 BrokenTooth recordings (loads 0%-90%) |
| Processed features | ✅ Generated | `data/processed/{train,val,test}_features.csv` | 1946 total windows, 22 features each |
| Processed windows | ✅ Generated | `data/processed/{train,val,test}_windows.npy` | 2048 samples per window |
| Splits summary | ✅ Generated | `data/processed/splits_summary.json` | Leakage-free GroupShuffleSplit summary |
| Random Forest model | ✅ Trained | `models/artifacts/random_forest.joblib` | 122.1 KB |
| XGBoost model | ✅ Trained | `models/artifacts/xgboost.joblib` | 153.2 KB |
| 1D CNN checkpoint | ✅ Trained | `models/artifacts/cnn_1d.pt` | 262.8 KB (PyTorch, early stopping at epoch 17) |
| Anomaly Detector | ✅ Trained | `models/artifacts/anomaly_detector.joblib` | Isolation Forest fitted strictly on healthy baseline |
| Model Comparison Report | ✅ Generated | `models/artifacts/model_comparison.{json,csv}` | Benchmarks on held-out test groups |

---

## 4. Held-Out Test Evaluation Results

| Model | Accuracy | Balanced Accuracy | Macro F1 | Weighted F1 | Healthy F1 | BrokenTooth F1 | Latency (ms) | Size (KB) |
|---|---|---|---|---|---|---|---|---|
| Random Forest | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 14.08 | 122.1 |
| XGBoost | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.50 | 153.2 |
| 1D CNN | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.24 | 262.8 |

---

## 5. Automated Test Suite

32 out of 32 tests passing across all 9 test suites:
- `tests/test_api.py` (5 tests)
- `tests/test_data_validation.py` (5 tests)
- `tests/test_features.py` (5 tests)
- `tests/test_inference.py` (2 tests)
- `tests/test_models.py` (5 tests)
- `tests/test_monitoring.py` (3 tests)
- `tests/test_preprocessing.py` (3 tests)
- `tests/test_splitting.py` (2 tests)
- `tests/test_storage.py` (2 tests)

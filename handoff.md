# Handoff Guide — AI Predictive Maintenance Platform

> **Author:** Vishnu Ganesh (via AI assistant)  
> **Handoff Date:** 2026-10-23  
> **Priority:** Continue from where this session left off

---

## 1. Project Summary

An end-to-end AI-powered **Predictive Maintenance and Mechanical Failure Prevention** platform for gearbox vibration monitoring. Uses the US Data.gov / OpenEI Gearbox Fault Diagnosis Dataset to train Random Forest, XGBoost, 1D CNN, and Isolation Forest models. Provides a Streamlit industrial dashboard and FastAPI REST API for real-time condition monitoring.

**Project Location:**
```
/Users/vishnuganesh/.gemini/antigravity-ide/scratch/predictive-maintenance/
```

---

## 2. What Has Been Done

### ✅ Fully Implemented & Executed
- Complete repository scaffolding with proper Python packaging (`pyproject.toml`, `.gitignore`, `.env.example`)
- Python 3.11 virtual environment with all dependencies installed (`.venv/`)
- **Dataset:** Downloaded and extracted the OpenEI Gearbox Fault Diagnosis Dataset (36.8 MB, 20 recordings)
- **Data Preparation Executed:** 1,946 segmented windows extracted (2,048 samples, 50% overlap), 22 statistical time/frequency features computed, zero-leakage `GroupShuffleSplit` by `recording_id`
- **Signal Processing Pipeline:** Validators (flatline, clipping, NaN/Inf), preprocessors (detrend + z-score), 2048-sample segmentation, 22 features
- **All 4 ML Models Trained & Saved:**
  - Random Forest (`models/artifacts/random_forest.joblib`)
  - XGBoost (`models/artifacts/xgboost.joblib`)
  - 1D CNN with MPS/CPU acceleration (`models/artifacts/cnn_1d.pt`)
  - Isolation Forest Anomaly Detector (`models/artifacts/anomaly_detector.joblib`)
- **Model Evaluation on Held-Out Test Groups:** Benchmarked accuracy, balanced accuracy, F1, latency, and model sizes (`model_comparison.json` and `model_comparison.csv`)
- **Unified Inference Engine:** End-to-end pipeline: validate → preprocess → classify → anomaly score → state machine → persist → recommend
- **Monitoring:** Consecutive-window state machine (NORMAL → WARNING → FAULT with multi-window recovery), alert manager with deduplication, SQLite history
- **4 Sensor Sources:** Hardware serial, CSV replay, synthetic demo, file upload
- **FastAPI REST API:** `/health`, `/predict`, `/models`, `/machines/{id}/history`, `/alerts` CRUD (verified and functional)
- **Streamlit Dashboard:** 4 pages (Machine Monitor, Data Explorer, Model Performance, Alerts & History) with Plotly charts (verified)
- **Test Suite:** 32 tests across 9 files — **100% passing (32/32 PASSED)**
- **All Bugs Resolved:** BUG-001 (Pydantic Settings), BUG-002 (XGBoost early stopping), BUG-003 (macOS OpenMP conflicts), BUG-004 (Monitoring recovery state retention)
- **Documentation:** README, architecture, dataset provenance, evaluation, hardware integration, limitations, model card, build-state

### 🚀 Ready to Run
1. **Launch Streamlit dashboard:** `streamlit run app.py` (Runs on http://localhost:8501)
2. **Launch FastAPI REST service:** `uvicorn src.predictive_maintenance.api.main:app --port 8000 --reload` (Swagger UI on http://localhost:8000/docs)

---

## 3. Critical Bug to Fix First

### BUG-001: `config.py` Settings class breaks with pydantic-settings v2

**File:** `src/predictive_maintenance/config.py` (Line 25)

**Error:** `TypeError: 'property' object is not iterable`

**The Problem:** The `Settings(BaseSettings)` class uses Python `@property` decorators, which conflict with Pydantic v2's metaclass processing.

**Recommended Fix — Replace `@property` with regular attributes initialized in `__init__`:**

```python
# BEFORE (broken):
class Settings(BaseSettings):
    root_dir: Path = get_project_root()
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._config = load_yaml(self.config_yaml_path)
    
    @property
    def system(self) -> Dict[str, Any]:
        return self._config.get("system", {})

# AFTER (fixed):
class Settings:
    """Application and pipeline settings. Plain class (no BaseSettings)."""
    def __init__(self):
        self.root_dir = get_project_root()
        self.config_yaml_path = self.root_dir / "configs" / "config.yaml"
        self.model_config_yaml_path = self.root_dir / "configs" / "model_config.yaml"
        self._config = load_yaml(self.config_yaml_path)
        self._model_config = load_yaml(self.model_config_yaml_path)
    
    @property
    def system(self) -> Dict[str, Any]:
        return self._config.get("system", {})
    
    @property
    def dataset(self) -> Dict[str, Any]:
        return self._config.get("dataset", {})
    
    # ... rest of properties unchanged ...

settings = Settings()
```

The simplest fix is to **stop inheriting from `BaseSettings`** since we're not using env-var binding — just make it a plain Python class.

---

## 4. Step-by-Step Execution Order

```bash
# 1. Navigate to project
cd /Users/vishnuganesh/.gemini/antigravity-ide/scratch/predictive-maintenance

# 2. Activate virtual environment
source .venv/bin/activate

# 3. Fix BUG-001 in src/predictive_maintenance/config.py (see Section 3)

# 4. Verify fix
python -c "from predictive_maintenance.config import settings; print(settings.system)"

# 5. Run data preparation (segments recordings, extracts features, group-splits)
python scripts/prepare_data.py

# 6. Train all models
python scripts/train.py --models all

# 7. Evaluate on held-out test groups
python scripts/evaluate.py

# 8. Run test suite
pytest tests/ -v

# 9. Launch Streamlit dashboard
streamlit run app.py
# Open http://localhost:8501

# 10. Launch FastAPI (in separate terminal)
uvicorn src.predictive_maintenance.api.main:app --port 8000 --reload
# Open http://localhost:8000/docs
```

---

## 5. Architecture Quick Reference

```
Vibration Sensor → Ring Buffer → Signal Validator → Preprocessor
    ↓
Feature Extractor (22 features) → RF / XGBoost / Anomaly Detector
    ↓                                       ↓
1D CNN (raw waveform) ──────────→ Unified Inference Engine
    ↓
State Machine (NORMAL/WARNING/FAULT) → Maintenance Recommendation
    ↓
SQLite Persistence → Streamlit Dashboard / FastAPI API
```

### Key Design Decisions
- **Data Leakage Prevention:** `GroupShuffleSplit` by `recording_id` — no window from the same physical recording appears in multiple splits
- **State Machine:** Requires N=3 consecutive fault windows to confirm FAULT, M=5 consecutive normals to recover
- **Fault Classes:** Only `Healthy` and `BrokenTooth` (from OpenEI dataset). No fabricated classes
- **Window Size:** 2048 samples, 50% overlap, at 5 kHz sampling rate

---

## 6. Key Files to Understand First

| Priority | File | Why |
|----------|------|-----|
| 1 | `src/predictive_maintenance/config.py` | **Fix this first** (BUG-001) |
| 2 | `src/predictive_maintenance/schemas.py` | All data types and Pydantic models |
| 3 | `src/predictive_maintenance/models/inference.py` | Core inference pipeline |
| 4 | `src/predictive_maintenance/signal/features.py` | Feature engineering |
| 5 | `src/predictive_maintenance/monitoring/rules.py` | State machine logic |
| 6 | `scripts/prepare_data.py` | Data pipeline entry point |
| 7 | `scripts/train.py` | Training pipeline entry point |
| 8 | `app.py` + `dashboard/pages/` | Dashboard UI |

---

## 7. Dataset Details

- **Source:** US Data.gov / OpenEI Gearbox Fault Diagnosis Dataset
- **URL:** `https://data.openei.org/files/623/gearboxdata.zip`
- **Physical Rig:** SpectraQuest Gearbox Diagnostics Simulator (GDS)
- **Sensors:** 4 accelerometers
- **Load Conditions:** 0% to 90%
- **Classes:** `Healthy` (normal spur gear) and `BrokenTooth` (artificially induced tooth crack)
- **Format:** Multi-column tab-delimited `.txt` files (one column per sensor channel)

---

## 8. API Endpoints Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | System health + model status |
| `GET` | `/models` | List available/trained models |
| `POST` | `/predict` | Run inference on vibration data |
| `GET` | `/machines` | List active machine IDs |
| `GET` | `/machines/{id}/history` | Observation history |
| `GET` | `/alerts` | List alerts (filterable) |
| `POST` | `/alerts/{id}/acknowledge` | Acknowledge an alert |
| `POST` | `/alerts/{id}/resolve` | Resolve an alert |

### Example `/predict` Request Body:
```json
{
  "samples": [0.1, -0.2, 0.15, ...],  // 2048+ float values
  "working_parameters": {
    "machine_id": "GEARBOX_01",
    "load_percent": 50.0,
    "rotational_speed_rpm": 1800,
    "sampling_rate_hz": 5000.0
  },
  "source": "live_sensor"
}
```

---

## 9. Potential Improvements (Backlog)

1. Multi-channel synchronous processing (4 accelerometer channels simultaneously)
2. CWRU Bearing Dataset integration for bearing-specific fault classification
3. ONNX/TFLite Micro export for edge microcontroller deployment
4. Time-series trending and RUL estimation (requires run-to-failure data)
5. WebSocket streaming in dashboard for true real-time updates
6. Docker containerization for deployment

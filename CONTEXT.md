# Project Context — Quick Reference Card

## What Is This?
AI-powered gearbox predictive maintenance system. Analyzes vibration signals → classifies Healthy vs BrokenTooth → recommends maintenance actions.

## Tech Stack
- Python 3.11, PyTorch, scikit-learn, XGBoost
- FastAPI (REST API), Streamlit (dashboard), SQLite (persistence)
- Plotly (charts), Pydantic v2 (validation), pySerial (hardware)

## One-Liner Architecture
```
Vibration Data → Validate → Extract 22 Features → ML Classification → State Machine → Alert/Recommend
```

## Current State (2026-10-23)
- ✅ All source code written (40+ files)
- ✅ Dataset downloaded (OpenEI gearbox data)
- 🔴 MUST FIX: `config.py` Pydantic BaseSettings bug (change to plain class)
- ❌ Pipeline not yet run: `prepare_data.py` → `train.py` → `evaluate.py`

## Run Order After Bug Fix
```bash
source .venv/bin/activate
python scripts/prepare_data.py    # Extract features & split data
python scripts/train.py --models all  # Train RF, XGBoost, CNN, Anomaly
python scripts/evaluate.py        # Benchmark test-set performance
pytest tests/ -v                  # Run test suite
streamlit run app.py              # Launch dashboard (port 8501)
```

## Supported Fault Classes
| Class | Description | Source |
|-------|-------------|--------|
| Healthy | Normal spur gear meshing | OpenEI dataset |
| BrokenTooth | Artificially induced tooth crack | OpenEI dataset |

## Key Models
| Model | Input | Purpose |
|-------|-------|---------|
| Random Forest | 22 statistical features | Fast baseline classifier |
| XGBoost | 22 statistical features | High-accuracy classifier |
| 1D CNN (PyTorch) | Raw 2048-sample waveform | Deep learning classifier |
| Isolation Forest | 22 features (healthy only) | Anomaly detection |

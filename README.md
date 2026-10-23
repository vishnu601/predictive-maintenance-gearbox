# AI-Based Predictive Maintenance and Fault Diagnosis of Gearboxes

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-teal.svg)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30+-red.svg)](https://streamlit.io/)

An industrial-grade mechanical condition monitoring and failure prevention platform analyzing gearbox vibration signals to detect abnormal operating conditions, diagnose gear tooth damage at an early stage, and provide cautious inspection recommendations.

---

## 1. System Overview

Traditional maintenance relies on fixed time-based inspection intervals or reactive repairs after machinery fails. This platform implements condition-based monitoring using vibration accelerometers, feature engineering, and artificial intelligence:

1. **Mechanical Gearbox:** Operates under normal or faulty loading conditions (0% to 90% load).
2. **Vibration Sensor:** Piezoelectric accelerometers measure vibration acceleration waveforms.
3. **Signal Processing & Feature Extraction:** Computes statistical time-domain features (RMS, Peak-to-Peak, Kurtosis, Crest Factor) and frequency-domain FFT power spectra.
4. **Machine Learning & Deep Learning:** Random Forest, XGBoost, and a PyTorch 1D Convolutional Neural Network classify conditions and detect anomalies without data leakage.
5. **Operational State Machine & Alerts:** Enforces consecutive-window verification (`NORMAL`, `WARNING`, `FAULT`, `SENSOR_ERROR`) and generates actionable maintenance recommendations.

---

## 2. Supported Conditions & Ground Truth

- **Healthy:** Normal spur gear meshing dynamics under load.
- **Gear Wear / Tooth Damage (`BrokenTooth`):** Artificially induced tooth crack / chipped tooth condition.
- **Unsupported Classes:** Bearing faults, misalignment, and looseness are tracked as `UNAVAILABLE_FOR_DATASET` per our strict non-fabrication policy.

---

## 3. Repository Architecture

```text
predictive-maintenance/
├── app.py                      # Main Streamlit dashboard entrypoint
├── README.md                   # System documentation & quickstart
├── requirements.txt            # Pinned dependencies
├── pyproject.toml              # Build & package configuration
├── configs/
│   ├── config.yaml             # System & operational status configuration
│   └── model_config.yaml       # ML & CNN training hyperparameters
├── data/
│   ├── raw/                    # Downloaded & extracted OpenEI dataset
│   ├── processed/              # Group-split features and raw windows
│   └── interim/                # Temporary processing artifacts
├── models/
│   └── artifacts/              # Serialized RF, XGBoost, CNN checkpoints & metrics
├── src/predictive_maintenance/
│   ├── config.py               # Settings loader
│   ├── schemas.py              # Pydantic v2 schemas for data, features, inference
│   ├── logging_config.py       # Structured logging
│   ├── data/
│   │   ├── loaders.py          # Multi-channel text/CSV loader
│   │   ├── dataset_adapters.py # OpenEI dataset adapter
│   │   └── splitting.py        # Leakage-free GroupShuffleSplit
│   ├── signal/
│   │   ├── validators.py       # Sanity checks (flatline, clipping, NaNs)
│   │   ├── preprocessing.py    # Detrending and normalization
│   │   ├── segmentation.py     # 2048-sample sliding window generator
│   │   └── features.py         # Time & FFT frequency feature extraction
│   ├── models/
│   │   ├── random_forest.py    # Scikit-learn Random Forest baseline
│   │   ├── xgboost_model.py    # XGBoost classifier with early stopping
│   │   ├── cnn_1d.py           # PyTorch 1D Deep CNN
│   │   ├── anomaly_detection.py# Isolation Forest anomaly detector
│   │   └── inference.py        # Unified inference engine
│   ├── monitoring/
│   │   ├── rules.py            # State machine & maintenance recommendations
│   │   ├── alerts.py           # Alert manager with deduplication
│   │   └── history.py          # SQLite observation storage
│   ├── sensors/
│   │   ├── base.py             # Bounded ring buffer abstraction
│   │   ├── csv_source.py       # Offline file replay
│   │   ├── serial_source.py    # Hardware UART/USB serial integration
│   │   └── demo_source.py      # Synthetic demonstration source
│   └── api/
│       └── main.py             # FastAPI REST service
├── dashboard/
│   ├── pages/
│   │   ├── 1_Machine_Monitor.py# Live waveform, FFT spectrum, status cards
│   │   ├── 2_Data_Explorer.py  # File upload & on-demand diagnosis
│   │   ├── 3_Model_Performance.py # Comparison tables & confusion matrices
│   │   └── 4_Alerts_History.py # Observation audit log & CSV export
│   └── components/
│       └── charts.py           # Plotly interactive industrial visualizations
├── scripts/
│   ├── download_dataset.py     # Downloads OpenEI dataset archive
│   ├── prepare_data.py         # Extracts features & creates group splits
│   ├── train.py                # Trains all models
│   └── evaluate.py             # Benchmarks test-set metrics and latencies
├── tests/                      # Automated unit and integration test suite
└── docs/                       # In-depth architectural & technical documentation
```

---

## 4. Installation & Quickstart

### Prerequisites
- Python 3.11
- macOS, Linux, or Windows

### Step 1: Clone & Create Virtual Environment
```bash
git clone <repo_url>
cd predictive-maintenance
python3.11 -m venv .venv

# On macOS / Linux:
source .venv/bin/activate

# On Windows:
.venv\Scripts\activate
```

### Step 2: Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 5. End-to-End Pipeline Execution

### Step 3: Download OpenEI Gearbox Dataset
```bash
python scripts/download_dataset.py
```

### Step 4: Prepare Data & Extract Leakage-Free Features
```bash
python scripts/prepare_data.py
```

### Step 5: Train Models (Random Forest, XGBoost, 1D CNN, Anomaly Detector)
```bash
python scripts/train.py --models all
```

### Step 6: Benchmark & Evaluate Models on Held-Out Test Set
```bash
python scripts/evaluate.py
```

---

## 6. Running the Applications

### Launch the Streamlit Industrial Dashboard
```bash
streamlit run app.py
```
Open [http://localhost:8501](http://localhost:8501) in your browser.

### Launch the FastAPI REST Service
```bash
uvicorn src.predictive_maintenance.api.main:app --port 8000 --reload
```
Interactive API documentation is available at [http://localhost:8000/docs](http://localhost:8000/docs).

---

## 7. Running Automated Tests

```bash
pytest tests/ -v
```

---

## 8. Hardware Serial Integration

Connect any microcontroller (Arduino, STM32, ESP32) sending framed vibration readings over USB serial:
- Baud Rate: `115200`
- Format: `timestamp,acceleration_value\n`
- In the dashboard, navigate to **Machine Monitor**, select **Hardware Serial Port**, enter your port (e.g. `/dev/tty.usbserial-0001` or `COM3`), and click **Connect**.

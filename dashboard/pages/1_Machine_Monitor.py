"""
Page 1: Machine Monitor.
Real-time condition monitoring, vibration waveforms, FFT power spectra,
and operational status state machine.
"""

from pathlib import Path
import sys
import time
import numpy as np
import streamlit as st

root_dir = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(root_dir / "src"))

from predictive_maintenance.config import settings
from predictive_maintenance.models.inference import UnifiedInferenceEngine
from predictive_maintenance.sensors.csv_source import CSVReplaySensorSource
from predictive_maintenance.sensors.demo_source import SyntheticDemoSensorSource
from predictive_maintenance.sensors.serial_source import SerialSensorSource
from dashboard.components.charts import create_fft_spectrum_chart, create_waveform_chart

st.set_page_config(page_title="Machine Monitor | Gearbox AI", page_icon="📈", layout="wide")

from dashboard.components.navigation import render_sidebar_navigation
render_sidebar_navigation()

st.title("📈 Real-Time Machine Monitor")
st.markdown("Live vibration waveform acquisition, spectral analysis, and operational state classification.")

# Initialize or retrieve inference engine
if "inference_engine" not in st.session_state:
    st.session_state.inference_engine = UnifiedInferenceEngine(model_name="random_forest")

engine = st.session_state.inference_engine

# Sidebar Configuration
st.sidebar.header("⚙️ Monitoring Settings")
machine_id = st.sidebar.selectbox("Select Asset / Machine ID", ["GEARBOX_01", "GEARBOX_02", "GEARBOX_03"])

# Model Selection
model_choice = st.sidebar.selectbox(
    "Active Classification Model",
    ["random_forest", "xgboost", "cnn"],
    index=0,
)
if model_choice != engine.model_name:
    engine.switch_model(model_choice)

# Data Source Selection
source_type = st.sidebar.radio(
    "Telemetry Data Source",
    ["Dataset Replay (OpenEI)", "Synthetic Demonstration", "Hardware Serial Port"],
    index=0,
)

# Manage Sensor Source Instance
if "sensor_source" not in st.session_state or st.session_state.get("current_source_type") != source_type:
    # Cleanup previous source
    if "sensor_source" in st.session_state and st.session_state.sensor_source:
        try:
            st.session_state.sensor_source.disconnect()
        except Exception:
            pass

    st.session_state.current_source_type = source_type

    if source_type == "Dataset Replay (OpenEI)":
        extracted_dir = root_dir / "data" / "raw" / "gearbox_extracted"
        h_files = list(extracted_dir.glob("Healthy Data/*.txt"))
        b_files = list(extracted_dir.glob("BrokenTooth Data/*.txt"))
        all_files = h_files + b_files

        if all_files:
            file_names = [f"{p.parent.name}/{p.name}" for p in all_files]
            selected_file_str = st.sidebar.selectbox("Select Recording to Replay", file_names, index=0)
            selected_path = extracted_dir / selected_file_str
            st.session_state.sensor_source = CSVReplaySensorSource(
                file_path=selected_path,
                sampling_rate_hz=5000.0,
                selected_channel=1,
                chunk_size=512,
                loop=True,
            )
            st.session_state.sensor_source.connect()
        else:
            st.sidebar.warning("Dataset not extracted. Falling back to Synthetic Demo.")
            st.session_state.sensor_source = SyntheticDemoSensorSource(mode="Healthy")
            st.session_state.sensor_source.connect()

    elif source_type == "Synthetic Demonstration":
        demo_mode = st.sidebar.radio("Simulated Condition", ["Healthy", "BrokenTooth"], index=0)
        st.session_state.sensor_source = SyntheticDemoSensorSource(mode=demo_mode)
        st.session_state.sensor_source.connect()

    elif source_type == "Hardware Serial Port":
        port = st.sidebar.text_input("Serial Port", value="/dev/tty.usbserial-0001")
        baud = st.sidebar.selectbox("Baud Rate", [9600, 57600, 115200, 230400], index=2)
        st.session_state.sensor_source = SerialSensorSource(port=port, baud_rate=baud)
        connect_btn = st.sidebar.button("Connect to Serial Device")
        if connect_btn:
            success = st.session_state.sensor_source.connect()
            if success:
                st.sidebar.success(f"Connected to {port}")
            else:
                st.sidebar.error(f"Failed to connect to {port}")

sensor = st.session_state.get("sensor_source")
if sensor and not sensor.is_connected():
    sensor.connect()

# Read latest window from bounded buffer
window_size = 2048
latest_window = sensor.get_window(window_size=window_size) if sensor else None

if latest_window is None:
    # Buffer filling
    time.sleep(0.3)
    latest_window = sensor.get_window(window_size=window_size) if sensor else None

# Synthetic Demo Indicator Banner
if sensor and sensor.get_metadata().get("is_synthetic"):
    st.warning("⚠️ **SYNTHETIC DEMO MODE ACTIVE**: Telemetry is generated via mathematical simulation. Not physical machine measurements.")

# Run Inference
if latest_window is not None:
    source_label = sensor.get_metadata().get("source_type", "sensor")
    # Working parameters
    load_pct = sensor.get_metadata().get("inferred_load_percent", 0.0)
    working_params = {
        "machine_id": machine_id,
        "load_percent": load_pct,
        "rotational_speed_rpm": 1800.0,
        "sampling_rate_hz": 5000.0,
    }

    result = engine.predict(
        input_data=latest_window,
        machine_id=machine_id,
        working_parameters=working_params,
        source=source_label,
        persist=True,
    )

    # --- TOP STATUS & METRICS ROW ---
    st.markdown("### Operational Condition & Telemetry Status")
    stat_col1, stat_col2, stat_col3, stat_col4 = st.columns([2, 1, 1, 1])

    with stat_col1:
        status_val = result.operational_status.value
        if status_val == "NORMAL":
            st.success(f"🟢 **OPERATIONAL STATUS: NORMAL**\n\nCondition: {result.predicted_class}")
        elif status_val == "WARNING":
            st.warning(f"🟡 **OPERATIONAL STATUS: WARNING**\n\nCondition: {result.predicted_class} (Investigate)")
        elif status_val == "FAULT":
            st.error(f"🔴 **OPERATIONAL STATUS: FAULT**\n\nCondition: {result.predicted_class} (Action Required)")
        else:
            st.error(f"⚠️ **SENSOR STATUS: {status_val}**")

    with stat_col2:
        conf_str = f"{result.confidence * 100:.1f}%" if result.confidence else "N/A"
        st.metric("Model Confidence", conf_str, help="Predicted class certainty from trained model")

    with stat_col3:
        rms_val = result.features.get("rms", 0.0)
        st.metric("Vibration RMS", f"{rms_val:.3f} g", help="Root Mean Square vibration severity")

    with stat_col4:
        peak_val = result.features.get("peak_to_peak", 0.0)
        st.metric("Peak-to-Peak", f"{peak_val:.3f} g", help="Total peak-to-peak acceleration swing")

    # Recommendation Alert Box
    st.info(f"💡 **Maintenance Advice:** {result.maintenance_recommendation}")

    # --- CHARTS SECTION ---
    tab1, tab2 = st.tabs(["📊 Time-Domain Waveform", "📈 Frequency Spectrum (FFT)"])

    with tab1:
        waveform_fig = create_waveform_chart(
            signal=latest_window,
            sampling_rate_hz=5000.0,
            title=f"Live Acceleration Waveform ({machine_id} - Window Size: {window_size} samples)",
            peak_threshold=8.0,
        )
        st.plotly_chart(waveform_fig, use_container_width=True)

    with tab2:
        fft_fig = create_fft_spectrum_chart(
            signal=latest_window,
            sampling_rate_hz=5000.0,
            title=f"One-Sided FFT Power Spectrum ({machine_id})",
            max_freq_hz=1500.0,
        )
        st.plotly_chart(fft_fig, use_container_width=True)

    # Detailed Feature Metrics Expander
    with st.expander("🔍 Detailed Vibration Feature Measurements"):
        f_cols = st.columns(4)
        feats = result.features
        f_cols[0].metric("Crest Factor", f"{feats.get('crest_factor', 0.0):.2f}")
        f_cols[0].metric("Kurtosis (Fisher)", f"{feats.get('kurtosis', 0.0):.2f}")
        f_cols[1].metric("Shape Factor", f"{feats.get('shape_factor', 0.0):.2f}")
        f_cols[1].metric("Impulse Factor", f"{feats.get('impulse_factor', 0.0):.2f}")
        f_cols[2].metric("Dominant Frequency", f"{feats.get('dominant_frequency', 0.0):.1f} Hz")
        f_cols[2].metric("Spectral Centroid", f"{feats.get('spectral_centroid', 0.0):.1f} Hz")
        f_cols[3].metric("Spectral Energy", f"{feats.get('spectral_energy', 0.0):.2e}")
        f_cols[3].metric("Spectral Entropy", f"{feats.get('spectral_entropy', 0.0):.2f}")

else:
    st.info("⏳ Waiting for sensor buffer to accumulate 2,048 samples... Please wait a moment.")

# Continuous polling refresher button
st.sidebar.markdown("---")
if st.sidebar.button("🔄 Refresh Telemetry"):
    st.rerun()

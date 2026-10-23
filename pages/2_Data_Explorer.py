"""
Page 2: Data Explorer.
File upload, interactive signal inspection, feature extraction, and window diagnosis.
"""

from pathlib import Path
import sys
import tempfile
import numpy as np
import pandas as pd
import streamlit as st

root_dir = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(root_dir / "src"))

from predictive_maintenance.data.loaders import load_vibration_file
from predictive_maintenance.models.inference import UnifiedInferenceEngine
from predictive_maintenance.signal.features import extract_features
from predictive_maintenance.signal.preprocessing import preprocess_window
from dashboard.components.charts import create_fft_spectrum_chart, create_waveform_chart

st.set_page_config(page_title="Data Explorer | Gearbox AI", page_icon="🔍", layout="wide")

from dashboard.components.navigation import render_sidebar_navigation
render_sidebar_navigation()

st.title("🔍 Vibration Data Explorer")
st.markdown("Upload, inspect, and analyze gearbox vibration recordings across time and frequency domains.")

if "inference_engine" not in st.session_state:
    st.session_state.inference_engine = UnifiedInferenceEngine(model_name="random_forest")

engine = st.session_state.inference_engine

# Selection Method
mode = st.radio("Select Ingestion Mode", ["Choose from Local Dataset", "Upload Custom Vibration File (.txt, .csv)"], horizontal=True)

signal_data = None
file_meta = {}

if mode == "Choose from Local Dataset":
    raw_dir = root_dir / "data" / "raw" / "gearbox_extracted"
    all_files = sorted(list(raw_dir.glob("*/*.txt")))
    if all_files:
        options = [f"{p.parent.name}/{p.name}" for p in all_files]
        chosen = st.selectbox("Select Gearbox Recording", options, index=0)
        chosen_path = raw_dir / chosen
        channel = st.selectbox("Accelerometer Channel", [1, 2, 3, 4], index=0)
        try:
            signal_data, file_meta = load_vibration_file(chosen_path, selected_channel=channel)
        except Exception as e:
            st.error(f"Error loading file: {e}")
    else:
        st.warning("No local recordings found. Please upload a file below.")

else:
    uploaded = st.file_uploader("Upload Vibration File", type=["txt", "csv"])
    if uploaded is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{uploaded.name}") as tmp:
            tmp.write(uploaded.getvalue())
            tmp_path = Path(tmp.name)

        channel = st.selectbox("Accelerometer Channel", [1, 2, 3, 4], index=0)
        try:
            signal_data, file_meta = load_vibration_file(tmp_path, selected_channel=channel)
        except Exception as e:
            st.error(f"Could not parse uploaded file: {e}")

if signal_data is not None and len(signal_data) > 0:
    st.markdown("---")
    st.markdown("### 📋 Recording Metadata & Summary")
    m_col1, m_col2, m_col3, m_col4 = st.columns(4)
    m_col1.metric("Total Samples", f"{len(signal_data):,}")
    m_col2.metric("Inferred Label", file_meta.get("inferred_label", "Unknown"))
    load_val = file_meta.get("inferred_load_percent")
    m_col3.metric("Operating Load", f"{load_val}%" if load_val is not None else "Unknown")
    m_col4.metric("Active Channel", f"Channel {channel} of {file_meta.get('total_channels', 1)}")

    # Window Selection
    window_size = 2048
    max_start = max(0, len(signal_data) - window_size)
    st.markdown("### 🎯 Interactive Window Selection")
    start_sample = st.slider("Window Start Sample", min_value=0, max_value=max_start, value=0, step=512)
    current_window = signal_data[start_sample : start_sample + window_size]

    # Preprocessing Toggle
    apply_proc = st.checkbox("Show Preprocessed Signal (Detrended & Normalized)", value=False)
    display_signal = preprocess_window(current_window, detrend=True, normalize=True) if apply_proc else current_window

    # Visualizations
    tab_wave, tab_fft, tab_diag = st.tabs(["📊 Waveform", "📈 Spectrum", "🧠 Diagnostic Inference"])

    with tab_wave:
        fig_wave = create_waveform_chart(
            signal=display_signal,
            sampling_rate_hz=5000.0,
            title=f"Vibration Window ({start_sample} to {start_sample + window_size})",
        )
        st.plotly_chart(fig_wave, use_container_width=True)

    with tab_fft:
        fig_fft = create_fft_spectrum_chart(
            signal=display_signal,
            sampling_rate_hz=5000.0,
            title=f"One-Sided FFT Spectrum ({start_sample} to {start_sample + window_size})",
            max_freq_hz=1500.0,
        )
        st.plotly_chart(fig_fft, use_container_width=True)

    with tab_diag:
        st.markdown("#### Run Diagnostic Inference on Current Window")
        if st.button("🚀 Analyze Window Condition"):
            wp = {
                "machine_id": "EXPLORER_GEARBOX",
                "load_percent": load_val,
                "sensor_channel": channel,
                "sampling_rate_hz": 5000.0,
            }
            res = engine.predict(current_window, working_parameters=wp, source="explorer_tool", persist=False)

            if res.operational_status.value == "NORMAL":
                st.success(f"🟢 **STATUS: NORMAL** — Condition: **{res.predicted_class}** (Confidence: {res.confidence * 100:.1f}%)")
            elif res.operational_status.value == "FAULT":
                st.error(f"🔴 **STATUS: FAULT** — Condition: **{res.predicted_class}** (Confidence: {res.confidence * 100:.1f}%)")
            else:
                st.warning(f"🟡 **STATUS: {res.operational_status.value}** — Condition: **{res.predicted_class}**")

            st.info(f"💡 **Recommendation:** {res.maintenance_recommendation}")

            st.markdown("##### Extracted Features")
            df_feats = pd.DataFrame([res.features]).T
            df_feats.columns = ["Value"]
            st.dataframe(df_feats, use_container_width=True)

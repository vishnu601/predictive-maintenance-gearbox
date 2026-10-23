"""
AI-Based Predictive Maintenance and Mechanical Failure Prevention Platform.
Main Streamlit Application Entrypoint.
"""

from pathlib import Path
import sys
import streamlit as st

# Add src to Python path
root_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(root_dir / "src"))

st.set_page_config(
    page_title="Gearbox AI Condition Monitor",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

from dashboard.components.navigation import render_sidebar_navigation
render_sidebar_navigation()

# Custom Industrial Dark Theme CSS
st.markdown("""
<style>
    /* Dark graphite theme styling */
    .stApp {
        background-color: #0e1117;
        color: #e5e7eb;
    }
    .metric-card {
        background-color: #181d28;
        border: 1px solid #242b3d;
        border-radius: 8px;
        padding: 16px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.3);
    }
    .status-normal {
        color: #10b981;
        font-weight: 700;
        font-size: 1.25rem;
    }
    .status-warning {
        color: #f59e0b;
        font-weight: 700;
        font-size: 1.25rem;
    }
    .status-fault {
        color: #ef4444;
        font-weight: 700;
        font-size: 1.25rem;
    }
    .status-sensor-error {
        color: #dc2626;
        font-weight: 700;
        font-size: 1.25rem;
    }
    .demo-badge {
        background-color: #854d0e;
        color: #fef08a;
        padding: 4px 8px;
        border-radius: 4px;
        font-size: 0.75rem;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

st.title("⚙️ Gearbox AI Condition Monitoring & Failure Prevention")
st.markdown("### Industrial Predictive Maintenance Platform for Rotating Machinery")

st.markdown("""
Welcome to the **AI-Based Gearbox Predictive Maintenance & Condition Monitoring Platform**.

This industrial-grade system analyzes vibration signals from gearbox accelerometers to detect abnormal operating conditions, diagnose mechanical faults at an early stage, and provide cautious inspection recommendations before catastrophic failure occurs.

---
""")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown("""
    <div class="metric-card">
        <h4 style="color:#00d4ff; margin-bottom:4px;">1. Machine Monitor</h4>
        <p style="font-size:0.875rem; color:#9ca3af;">
            Live waveform streaming, real-time FFT spectrum, RMS/P2P meters, operational state machine, and maintenance advice.
        </p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="metric-card">
        <h4 style="color:#00d4ff; margin-bottom:4px;">2. Data Explorer</h4>
        <p style="font-size:0.875rem; color:#9ca3af;">
            Upload vibration CSV/TXT files, inspect multi-channel signals, compute time & frequency features, and run diagnostic inference.
        </p>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="metric-card">
        <h4 style="color:#00d4ff; margin-bottom:4px;">3. Model Performance</h4>
        <p style="font-size:0.875rem; color:#9ca3af;">
            Compare Random Forest, XGBoost, and 1D CNN evaluated on held-out recordings with confusion matrices and latency benchmarks.
        </p>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown("""
    <div class="metric-card">
        <h4 style="color:#00d4ff; margin-bottom:4px;">4. Alerts & History</h4>
        <p style="font-size:0.875rem; color:#9ca3af;">
            Audit observation logs, filter active alerts, acknowledge/resolve maintenance events, and export historical CSV records.
        </p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")
st.info("👈 **Select a page from the left sidebar to begin monitoring or exploring data.**")

# System Status Summary
with st.expander("ℹ️ System Configuration & Dataset Provenance", expanded=False):
    st.markdown("""
    - **Primary Dataset:** US Data.gov / OpenEI Gearbox Fault Diagnosis Dataset (PIEMR / Dept of Energy).
    - **Physical Rig:** SpectraQuest Gearbox Diagnostics Simulator (GDS), 4-sensor accelerometer layout across 0% to 90% load.
    - **Supported Validated Classes:** `Healthy` (normal spur gear), `BrokenTooth` (artificially induced tooth crack/damage).
    - **Data Leakage Prevention:** Windows grouped strictly by original recording run (`GroupShuffleSplit`).
    - **Hardware Interface:** Configurable serial streaming receiver (baud 115200, framed ASCII/JSON).
    """)

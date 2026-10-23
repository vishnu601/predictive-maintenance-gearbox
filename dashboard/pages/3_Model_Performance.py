"""
Page 3: Model Performance & Comparison.
Evaluates Random Forest, XGBoost, and 1D CNN on held-out test groups.
Displays confusion matrices, latency benchmarks, learning curves, and model card.
"""

import json
from pathlib import Path
import sys
import pandas as pd
import streamlit as st

root_dir = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(root_dir / "src"))

from dashboard.components.charts import create_confusion_matrix_chart, create_model_comparison_bar

st.set_page_config(page_title="Model Performance | Gearbox AI", page_icon="🏆", layout="wide")

from dashboard.components.navigation import render_sidebar_navigation
render_sidebar_navigation()

st.title("🏆 Model Performance & Trustworthiness Evaluation")
st.markdown("Rigorous benchmark of classification algorithms evaluated strictly on **held-out recording groups**.")

artifacts_dir = root_dir / "models" / "artifacts"
comp_json_path = artifacts_dir / "model_comparison.json"
comp_csv_path = artifacts_dir / "model_comparison.csv"
splits_json_path = root_dir / "data" / "processed" / "splits_summary.json"

if not comp_json_path.exists():
    st.info("ℹ️ No model comparison metrics found yet. Run `python scripts/train.py --models all` and `python scripts/evaluate.py` to populate performance benchmarks.")
    st.code("python scripts/prepare_data.py\npython scripts/train.py --models all\npython scripts/evaluate.py", language="bash")
else:
    with open(comp_json_path, "r", encoding="utf-8") as f:
        metrics = json.load(f)

    st.markdown("### 📊 Model Comparison Overview")
    df_comp = pd.read_csv(comp_csv_path)
    st.dataframe(df_comp, use_container_width=True)

    # Bar Chart: Macro F1 vs Latency
    model_names = [m["model_name"] for m in metrics]
    f1s = [m["macro_f1"] for m in metrics]
    latencies = [m["inference_latency_ms"] for m in metrics]
    fig_bar = create_model_comparison_bar(model_names, f1s, latencies)
    st.plotly_chart(fig_bar, use_container_width=True)

    # Confusion Matrices
    st.markdown("### 🔲 Confusion Matrices (Held-Out Test Set)")
    cm_cols = st.columns(len(metrics))
    for i, m in enumerate(metrics):
        with cm_cols[i]:
            cm_fig = create_confusion_matrix_chart(
                cm=m["confusion_matrix"],
                classes=m["classes"],
                title=f"{m['model_name']} (Acc: {m['accuracy'] * 100:.1f}%)"
            )
            st.plotly_chart(cm_fig, use_container_width=True)

    # Dataset & Group-Aware Split Provenance
    if splits_json_path.exists():
        with open(splits_json_path, "r", encoding="utf-8") as f:
            splits_info = json.load(f)

        st.markdown("### 🛡️ Data Leakage Prevention & Split Hierarchy")
        st.markdown("""
        > **Mandatory Leakage Protection:** Vibration windows sliced from the same physical recording run are highly correlated. 
        > We partitioned the dataset using `GroupShuffleSplit` across recording runs so that **0% of recording groups overlap** between train, validation, and test sets.
        """)

        s_col1, s_col2, s_col3 = st.columns(3)
        with s_col1:
            st.markdown(f"**Train Partition:** {splits_info['train_windows']} windows")
            st.caption(", ".join(splits_info["train_groups"]))
        with s_col2:
            st.markdown(f"**Validation Partition:** {splits_info['val_windows']} windows")
            st.caption(", ".join(splits_info["val_groups"]))
        with s_col3:
            st.markdown(f"**Held-Out Test Partition:** {splits_info['test_windows']} windows")
            st.caption(", ".join(splits_info["test_groups"]))

# Model Card Expander
with st.expander("📄 Official Model Card & Technical Limitations", expanded=False):
    st.markdown("""
    #### Intended Use
    - **Asset Domain:** Two-stage spur gearboxes operating under steady load (0% to 90% load).
    - **Objective:** Condition monitoring, early fault detection, and maintenance scheduling.
    - **Supported Classes:** `Healthy` (normal gear condition), `BrokenTooth` (gear tooth damage / crack).
    - **Unsupported Classes:** Bearing faults, shaft misalignment, mechanical looseness (marked as unavailable for this dataset).
    
    #### Limitations & Trustworthiness
    1. **Not a Safety-Critical Automated Interlock:** Model recommendations are decision aids for maintenance engineers. Do not automatically trip or engage safety equipment without human review.
    2. **Transferability:** A model trained on a specific rig will experience domain drift if deployed on a different gearbox without calibration.
    3. **Remaining Useful Life (RUL):** This platform provides fault diagnosis and condition classification. It does not claim unvalidated run-to-failure RUL prediction.
    """)

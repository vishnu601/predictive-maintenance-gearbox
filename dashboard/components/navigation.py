"""
Reusable navigation sidebar component for Streamlit multi-page dashboard.
"""

import streamlit as st


def render_sidebar_navigation():
    """Renders prominent navigation links in the Streamlit left sidebar."""
    st.sidebar.markdown("## ⚙️ VIBRA·PULSE AI")
    st.sidebar.caption("Industrial Predictive Maintenance")
    st.sidebar.markdown("---")
    
    st.sidebar.markdown("### 🧭 Page Navigation")
    st.sidebar.page_link("app.py", label="System Overview", icon="🏠")
    st.sidebar.page_link("pages/1_Machine_Monitor.py", label="1. Machine Monitor", icon="📈")
    st.sidebar.page_link("pages/2_Data_Explorer.py", label="2. Data Explorer", icon="🔬")
    st.sidebar.page_link("pages/3_Model_Performance.py", label="3. Model Performance", icon="🏆")
    st.sidebar.page_link("pages/4_Alerts_History.py", label="4. Alerts & History", icon="🚨")
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📡 Engine Status")
    st.sidebar.success("● AI Pipeline: Ready")
    st.sidebar.info("● Model: Random Forest / CNN")
    st.sidebar.caption("SpectraQuest GDS (0%–90% Load)")

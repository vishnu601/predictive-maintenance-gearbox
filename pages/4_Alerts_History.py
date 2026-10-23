"""
Page 4: Alerts & Historical Monitoring.
Inspect observation logs, filter active alerts, acknowledge/resolve maintenance events,
and export records to CSV.
"""

from pathlib import Path
import sys
import pandas as pd
import streamlit as st

root_dir = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(root_dir / "src"))

from predictive_maintenance.config import settings
from predictive_maintenance.monitoring.alerts import AlertManager
from predictive_maintenance.monitoring.history import HistoryRepository

st.set_page_config(page_title="Alerts & History | Gearbox AI", page_icon="🚨", layout="wide")

from dashboard.components.navigation import render_sidebar_navigation
render_sidebar_navigation()

st.title("🚨 Alerts & Historical Observation Log")
st.markdown("Audit trail of machine health observations, persistent alerts, and maintenance tracking.")

db_path = root_dir / settings.storage.get("database_path", "data/monitoring.db")
history_repo = HistoryRepository(db_path)
alert_manager = AlertManager(db_path)

# Filter Controls
st.sidebar.header("🔍 Filters")
machine_filter = st.sidebar.selectbox("Filter Machine ID", ["All", "GEARBOX_01", "GEARBOX_02", "EXPLORER_GEARBOX"])
machine_id_arg = None if machine_filter == "All" else machine_filter

status_filter = st.sidebar.selectbox("Alert Status", ["All", "ACTIVE", "ACKNOWLEDGED", "RESOLVED"])
status_arg = None if status_filter == "All" else status_filter

# Fetch Alerts
df_alerts = alert_manager.get_alerts(machine_id=machine_id_arg, status=status_arg, limit=100)

st.markdown("### 🔔 Active & Historical Maintenance Alerts")

if df_alerts.empty:
    st.info("No alerts found matching the current filters.")
else:
    for _, row in df_alerts.iterrows():
        sev_color = "#ef4444" if row["severity"] == "CRITICAL" else "#f59e0b"
        status_badge = f"**[{row['status']}]**"
        with st.container():
            st.markdown(f"""
            <div style="border-left: 4px solid {sev_color}; padding: 10px 16px; background-color: #181d28; margin-bottom: 8px; border-radius: 4px;">
                <span style="font-weight:700; color:{sev_color};">{row['severity']} ALERT</span> - {row['alert_id']}
                <span style="float:right; color:#9ca3af; font-size:0.875rem;">{row['created_at']}</span>
                <p style="margin: 4px 0; font-size:0.95rem;"><b>Reason:</b> {row['trigger_reason']}</p>
                <p style="margin: 4px 0; color:#d1d5db; font-size:0.875rem;"><b>Action:</b> {row['recommendation']}</p>
            </div>
            """, unsafe_allow_html=True)

            col_a, col_b, _ = st.columns([1, 1, 4])
            if row["status"] == "ACTIVE":
                if col_a.button(f"Acknowledge", key=f"ack_{row['alert_id']}"):
                    alert_manager.acknowledge_alert(row["alert_id"])
                    st.rerun()
                if col_b.button(f"Resolve", key=f"res_{row['alert_id']}"):
                    alert_manager.resolve_alert(row["alert_id"])
                    st.rerun()
            elif row["status"] == "ACKNOWLEDGED":
                if col_a.button(f"Resolve", key=f"res_{row['alert_id']}"):
                    alert_manager.resolve_alert(row["alert_id"])
                    st.rerun()

st.markdown("---")

# Fetch Observations
st.markdown("### 📋 Historical Vibration Observations")
df_obs = history_repo.get_recent_observations(machine_id=machine_id_arg, limit=200)

if df_obs.empty:
    st.info("No observation history recorded yet.")
else:
    st.dataframe(df_obs, use_container_width=True)

    # CSV Export Button
    csv_data = df_obs.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="📥 Export Observation History (CSV)",
        data=csv_data,
        file_name=f"gearbox_observations_{machine_filter.lower()}.csv",
        mime="text/csv",
    )

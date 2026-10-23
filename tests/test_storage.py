"""
Unit tests for SQLite database persistence: observation logging and alert deduplication.
"""

from pathlib import Path
import sys
import tempfile
import pytest

root_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir / "src"))

from predictive_maintenance.monitoring.alerts import AlertManager
from predictive_maintenance.monitoring.history import HistoryRepository


def test_sqlite_observation_recording():
    with tempfile.NamedTemporaryFile(suffix=".db") as tmp:
        db_path = Path(tmp.name)
        repo = HistoryRepository(db_path)

        # Record observations
        row_id = repo.record_observation(
            machine_id="TEST_GEARBOX",
            predicted_class="Healthy",
            operational_status="NORMAL",
            rms=1.45,
            peak_to_peak=6.2,
            confidence=0.98,
            model_name="random_forest",
        )
        assert row_id == 1

        df = repo.get_recent_observations(machine_id="TEST_GEARBOX", limit=10)
        assert len(df) == 1
        assert df.iloc[0]["machine_id"] == "TEST_GEARBOX"
        assert df.iloc[0]["operational_status"] == "NORMAL"


def test_alert_deduplication_and_resolution():
    with tempfile.NamedTemporaryFile(suffix=".db") as tmp:
        db_path = Path(tmp.name)
        mgr = AlertManager(db_path)

        # 1. Trigger alert
        alt1 = mgr.trigger_alert_if_needed(
            machine_id="GEAR_01",
            operational_status="FAULT",
            predicted_class="BrokenTooth",
            recommendation="Inspect teeth",
        )
        assert alt1 is not None

        # 2. Trigger again while still active: must return same alert ID (deduplicated)
        alt2 = mgr.trigger_alert_if_needed(
            machine_id="GEAR_01",
            operational_status="FAULT",
            predicted_class="BrokenTooth",
            recommendation="Inspect teeth",
        )
        assert alt1 == alt2

        # 3. Acknowledge alert
        ack_res = mgr.acknowledge_alert(alt1)
        assert ack_res is True

        # 4. Resolve alert
        res_res = mgr.resolve_alert(alt1)
        assert res_res is True

        # Check status
        df_alerts = mgr.get_alerts(machine_id="GEAR_01")
        assert df_alerts.iloc[0]["status"] == "RESOLVED"

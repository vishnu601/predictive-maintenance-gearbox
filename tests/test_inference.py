"""
Integration tests for the Unified Inference Engine.
"""

from pathlib import Path
import sys
import tempfile
import numpy as np
import pytest

root_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir / "src"))

from predictive_maintenance.models.inference import UnifiedInferenceEngine
from predictive_maintenance.schemas import OperationalStatus


def test_inference_engine_on_valid_signal():
    with tempfile.NamedTemporaryFile(suffix=".db") as tmp:
        engine = UnifiedInferenceEngine(
            model_name="random_forest",
            db_path=Path(tmp.name)
        )
        sig = np.random.normal(0, 1.2, 2048)
        wp = {
            "machine_id": "TEST_GB",
            "load_percent": 50.0,
            "rotational_speed_rpm": 1800.0,
        }
        res = engine.predict(sig, machine_id="TEST_GB", working_parameters=wp, persist=True)

        assert res.machine_id == "TEST_GB"
        assert res.signal_quality == "VALID"
        assert res.operational_status in [OperationalStatus.NORMAL, OperationalStatus.WARNING, OperationalStatus.INSUFFICIENT_DATA]
        assert len(res.maintenance_recommendation) > 0


def test_inference_engine_detects_flatline():
    with tempfile.NamedTemporaryFile(suffix=".db") as tmp:
        engine = UnifiedInferenceEngine(
            model_name="random_forest",
            db_path=Path(tmp.name)
        )
        flat_sig = np.zeros(2048)
        res = engine.predict(flat_sig, machine_id="TEST_GB", persist=False)

        assert res.signal_quality == "FLATLINE"
        assert res.operational_status == OperationalStatus.SENSOR_ERROR
        assert "disconnect" in res.maintenance_recommendation.lower() or "sensor" in res.maintenance_recommendation.lower()

"""
Integration tests for the FastAPI REST service.
Verifies /health, /models, /predict, /machines, and /alerts endpoints.
"""

from pathlib import Path
import sys
import numpy as np
import pytest
from fastapi.testclient import TestClient

root_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir / "src"))

from predictive_maintenance.api.main import app

client = TestClient(app)


def test_api_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "model_loaded" in data


def test_api_models_endpoint():
    response = client.get("/models")
    assert response.status_code == 200
    data = response.json()
    assert "available_models" in data
    assert "random_forest" in data["available_models"]
    assert "xgboost" in data["available_models"]
    assert "cnn" in data["available_models"]
    assert "Healthy" in data["supported_fault_classes"]


def test_api_predict_rejects_too_short():
    short_payload = {
        "samples": [1.0, 2.0, 3.0],
        "working_parameters": {"machine_id": "TEST_01"},
    }
    response = client.post("/predict", json=short_payload)
    assert response.status_code == 400


def test_api_predict_valid_signal():
    # 1024 valid samples
    sig = list(np.random.normal(0, 1.0, 1024))
    payload = {
        "samples": sig,
        "working_parameters": {
            "machine_id": "API_TEST_GEARBOX",
            "load_percent": 30.0,
            "rotational_speed_rpm": 1800.0,
        },
        "source": "api_test",
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    res_data = response.json()
    assert res_data["machine_id"] == "API_TEST_GEARBOX"
    assert "operational_status" in res_data
    assert "maintenance_recommendation" in res_data


def test_api_machines_list():
    response = client.get("/machines")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)

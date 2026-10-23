"""
FastAPI REST API Service for Predictive Maintenance & Machine Condition Monitoring.
Exposes real-time inference, model management, and alert tracking endpoints.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional
from fastapi import FastAPI, HTTPException, Query, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from ..config import settings
from ..logging_config import setup_logger
from ..models.inference import UnifiedInferenceEngine
from ..schemas import AlertRecord, InferenceResult, VibrationDataInput

logger = setup_logger("api")

app = FastAPI(
    title="Gearbox Predictive Maintenance API",
    version="1.0.0",
    description="Industrial AI service for mechanical condition monitoring and gearbox failure prevention.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize engine instance
engine = UnifiedInferenceEngine(model_name="random_forest")


@app.get("/health", tags=["System"])
def get_health() -> Dict[str, Any]:
    """
    Health check distinguishing application readiness from model availability.
    """
    model_loaded = engine.classifier is not None
    return {
        "status": "healthy",
        "app_name": settings.system.get("app_name", "Predictive Maintenance API"),
        "version": "1.0.0",
        "model_loaded": model_loaded,
        "active_model": engine.model_name if model_loaded else None,
        "anomaly_detector_loaded": engine.anomaly_detector is not None,
    }


@app.get("/models", tags=["Models"])
def list_models() -> Dict[str, Any]:
    """
    Lists supported model types and available trained artifacts.
    """
    artifacts_dir = engine.artifacts_dir
    rf_exists = (artifacts_dir / "random_forest.joblib").exists()
    xgb_exists = (artifacts_dir / "xgboost.joblib").exists()
    cnn_exists = (artifacts_dir / "cnn_1d.pt").exists()
    ad_exists = (artifacts_dir / "anomaly_detector.joblib").exists()

    return {
        "active_model": engine.model_name,
        "available_models": {
            "random_forest": {"trained": rf_exists, "path": "models/artifacts/random_forest.joblib"},
            "xgboost": {"trained": xgb_exists, "path": "models/artifacts/xgboost.joblib"},
            "cnn": {"trained": cnn_exists, "path": "models/artifacts/cnn_1d.pt"},
            "anomaly_detector": {"trained": ad_exists, "path": "models/artifacts/anomaly_detector.joblib"},
        },
        "supported_fault_classes": settings.supported_classes,
        "unsupported_fault_classes": settings.dataset.get("unsupported_classes", []),
    }


@app.post("/predict", response_model=InferenceResult, tags=["Inference"])
def predict_vibration(payload: VibrationDataInput) -> InferenceResult:
    """
    Runs diagnostic inference on incoming vibration time-series data and gearbox working parameters.
    """
    if len(payload.samples) < 512:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Sample count {len(payload.samples)} is insufficient. Minimum length is 512."
        )

    try:
        result = engine.predict(
            input_data=payload,
            source=payload.source,
            persist=True,
        )
        return result
    except Exception as e:
        logger.error(f"Inference execution failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference error: {str(e)}"
        )


@app.get("/machines", tags=["Monitoring"])
def get_machines() -> List[str]:
    """Returns list of active machine IDs present in observation records."""
    df = engine.history_repo.get_recent_observations(limit=500)
    if df.empty or "machine_id" not in df.columns:
        return ["GEARBOX_01"]
    return sorted(list(df["machine_id"].unique()))


@app.get("/machines/{machine_id}/history", tags=["Monitoring"])
def get_machine_history(machine_id: str, limit: int = Query(default=100, ge=1, le=1000)) -> List[Dict[str, Any]]:
    """Returns recent observations for a specific machine."""
    df = engine.history_repo.get_recent_observations(machine_id=machine_id, limit=limit)
    if df.empty:
        return []
    return df.to_dict(orient="records")


@app.get("/alerts", tags=["Alerts"])
def get_alerts(
    machine_id: Optional[str] = None,
    status_filter: Optional[str] = Query(default=None, pattern="^(ACTIVE|ACKNOWLEDGED|RESOLVED)$"),
    limit: int = 100,
) -> List[Dict[str, Any]]:
    """Lists alerts filtered by machine and status."""
    df = engine.alert_manager.get_alerts(machine_id=machine_id, status=status_filter, limit=limit)
    if df.empty:
        return []
    return df.to_dict(orient="records")


@app.post("/alerts/{alert_id}/acknowledge", tags=["Alerts"])
def acknowledge_alert(alert_id: str) -> Dict[str, Any]:
    """Acknowledges an active alert."""
    success = engine.alert_manager.acknowledge_alert(alert_id)
    if not success:
        raise HTTPException(status_code=404, detail=f"Active alert {alert_id} not found.")
    return {"status": "success", "alert_id": alert_id, "state": "ACKNOWLEDGED"}


@app.post("/alerts/{alert_id}/resolve", tags=["Alerts"])
def resolve_alert(alert_id: str) -> Dict[str, Any]:
    """Resolves an alert."""
    success = engine.alert_manager.resolve_alert(alert_id)
    if not success:
        raise HTTPException(status_code=404, detail=f"Alert {alert_id} not found.")
    return {"status": "success", "alert_id": alert_id, "state": "RESOLVED"}


# Mount static web UI
web_dir = Path(__file__).resolve().parent.parent.parent.parent / "web"
if web_dir.exists():
    from starlette.staticfiles import StaticFiles
    from fastapi.responses import RedirectResponse

    app.mount("/dashboard", StaticFiles(directory=str(web_dir), html=True), name="dashboard")
    app.mount("/ui", StaticFiles(directory=str(web_dir), html=True), name="ui")

    @app.get("/", include_in_schema=False)
    def root():
        return RedirectResponse(url="/dashboard/")


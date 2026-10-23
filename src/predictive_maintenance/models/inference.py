"""
Unified Inference Engine for Gearbox Condition Monitoring.
Processes vibration waveforms, executes model inference, manages state transitions,
persists observations, and returns structured diagnosis and maintenance actions.
"""

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import numpy as np
import pandas as pd
from ..config import settings
from ..logging_config import setup_logger
from ..schemas import (
    GearboxWorkingParameters,
    InferenceResult,
    OperationalStatus,
    SignalQualityReport,
    VibrationDataInput,
)
from ..signal.features import extract_features
from ..signal.preprocessing import preprocess_window
from ..signal.validators import validate_vibration_signal
from .anomaly_detection import VibrationAnomalyDetector
from .cnn_1d import CNN1DClassifier
from .random_forest import RandomForestBaseline
from .xgboost_model import XGBoostClassifierModel
from ..monitoring.alerts import AlertManager
from ..monitoring.history import HistoryRepository
from ..monitoring.rules import MonitoringStateMachine, generate_maintenance_recommendation

logger = setup_logger(__name__)


class UnifiedInferenceEngine:
    """
    Production-grade inference coordinator powering both the Dashboard and FastAPI API.
    """

    def __init__(
        self,
        model_name: str = "random_forest",
        artifacts_dir: Optional[Path] = None,
        db_path: Optional[Path] = None,
        state_machine: Optional[MonitoringStateMachine] = None,
    ):
        self.model_name = model_name
        self.artifacts_dir = Path(artifacts_dir or settings.root_dir / "models" / "artifacts")
        self.db_path = Path(db_path or settings.root_dir / settings.storage.get("database_path", "data/monitoring.db"))

        self.classifier: Optional[Union[RandomForestBaseline, XGBoostClassifierModel, CNN1DClassifier]] = None
        self.anomaly_detector: Optional[VibrationAnomalyDetector] = None
        self.state_machine = state_machine or MonitoringStateMachine(
            fault_confirmation_windows=settings.operational_status.get("rules", {}).get("fault_confirmation_windows", 3),
            recovery_confirmation_windows=settings.operational_status.get("rules", {}).get("recovery_confirmation_windows", 5),
            fault_confidence_threshold=settings.operational_status.get("rules", {}).get("fault_confidence_threshold", 0.70),
        )
        self.history_repo = HistoryRepository(self.db_path)
        self.alert_manager = AlertManager(self.db_path)
        self.load_models()

    def load_models(self):
        """Loads trained classification and anomaly artifacts if available."""
        rf_path = self.artifacts_dir / "random_forest.joblib"
        xgb_path = self.artifacts_dir / "xgboost.joblib"
        cnn_path = self.artifacts_dir / "cnn_1d.pt"
        ad_path = self.artifacts_dir / "anomaly_detector.joblib"

        # Load classifier
        try:
            if self.model_name == "random_forest" and rf_path.exists():
                self.classifier = RandomForestBaseline.load(rf_path)
                logger.info(f"Loaded Random Forest model from {rf_path}")
            elif self.model_name == "xgboost" and xgb_path.exists():
                self.classifier = XGBoostClassifierModel.load(xgb_path)
                logger.info(f"Loaded XGBoost model from {xgb_path}")
            elif self.model_name == "cnn" and cnn_path.exists():
                self.classifier = CNN1DClassifier.load(cnn_path, device="cpu")
                logger.info(f"Loaded 1D CNN model from {cnn_path}")
            else:
                logger.warning(f"Requested model '{self.model_name}' artifact not found in {self.artifacts_dir}.")
        except Exception as e:
            logger.error(f"Failed to load classifier: {e}")
            self.classifier = None

        # Load anomaly detector
        try:
            if ad_path.exists():
                self.anomaly_detector = VibrationAnomalyDetector.load(ad_path)
                logger.info(f"Loaded Anomaly Detector from {ad_path}")
        except Exception as e:
            logger.warning(f"Could not load anomaly detector: {e}")
            self.anomaly_detector = None

    def switch_model(self, model_name: str):
        """Switches the active classification model."""
        self.model_name = model_name
        self.load_models()

    def predict(
        self,
        input_data: Union[VibrationDataInput, np.ndarray, list],
        machine_id: str = "GEARBOX_01",
        working_parameters: Optional[Dict[str, Any]] = None,
        source: str = "live_sensor",
        persist: bool = True,
    ) -> InferenceResult:
        """
        Executes end-to-end vibration inference.
        """
        now_iso = datetime.now(timezone.utc).isoformat()

        # Unpack input
        if isinstance(input_data, VibrationDataInput):
            samples = np.array(input_data.samples, dtype=np.float64)
            source = input_data.source
            if input_data.working_parameters:
                working_parameters = input_data.working_parameters.model_dump()
                machine_id = input_data.working_parameters.machine_id
        else:
            samples = np.array(input_data, dtype=np.float64)

        wp_dict = working_parameters or {"machine_id": machine_id}
        sampling_rate = wp_dict.get("sampling_rate_hz", 5000.0)

        # 1. Signal Validation
        quality: SignalQualityReport = validate_vibration_signal(samples, min_length=1024)
        if not quality.is_valid:
            status = self.state_machine.update_state(
                machine_id=machine_id,
                predicted_class="Unknown",
                confidence=None,
                is_anomaly=False,
                signal_valid=False,
            )
            rec = generate_maintenance_recommendation(status, "Unknown", None, wp_dict)
            return InferenceResult(
                machine_id=machine_id,
                timestamp=now_iso,
                source=source,
                signal_quality=quality.status,
                predicted_class="Unknown",
                operational_status=status,
                confidence=None,
                anomaly_score=None,
                model_name=self.model_name,
                model_version="1.0.0",
                features={},
                working_parameters=wp_dict,
                maintenance_recommendation=rec,
                limitations=["Input signal failed quality validation checks."],
            )

        # 2. Preprocessing & Feature Extraction
        features = extract_features(samples, sampling_rate_hz=sampling_rate)
        processed_window = preprocess_window(samples, detrend=True, normalize=True)

        # 3. Model Classification
        if self.classifier is None:
            status = OperationalStatus.INSUFFICIENT_DATA
            return InferenceResult(
                machine_id=machine_id,
                timestamp=now_iso,
                source=source,
                signal_quality="VALID",
                predicted_class="MODEL_UNAVAILABLE",
                operational_status=status,
                confidence=None,
                anomaly_score=None,
                model_name=self.model_name,
                model_version="unloaded",
                features=features,
                working_parameters=wp_dict,
                maintenance_recommendation="Model artifacts not yet trained. Run python scripts/train.py to initialize models.",
                limitations=["No trained model artifact is currently loaded."],
            )

        feat_df = pd.DataFrame([features])
        predicted_class = "Healthy"
        confidence: Optional[float] = None

        if isinstance(self.classifier, (RandomForestBaseline, XGBoostClassifierModel)):
            # Align features
            available_feats = [f for f in self.classifier.feature_names if f in feat_df.columns]
            preds = self.classifier.predict(feat_df[available_feats])
            predicted_class = str(preds[0])
            probs = self.classifier.predict_proba(feat_df[available_feats])[0]
            confidence = float(np.max(probs))
        elif isinstance(self.classifier, CNN1DClassifier):
            # 1D CNN takes normalized sequence
            preds = self.classifier.predict(processed_window)
            predicted_class = str(preds[0])
            probs = self.classifier.predict_proba(processed_window)[0]
            confidence = float(np.max(probs))

        # 4. Anomaly Detection
        anomaly_score: Optional[float] = None
        is_anomaly = False
        if self.anomaly_detector is not None:
            try:
                available_ad_feats = [f for f in self.anomaly_detector.feature_names if f in feat_df.columns]
                score = self.anomaly_detector.score_anomaly(feat_df[available_ad_feats])[0]
                anomaly_score = float(score)
                is_anomaly = bool(score > self.anomaly_detector.threshold)
            except Exception as e:
                logger.warning(f"Error computing anomaly score: {e}")

        # 5. Operational Status via State Machine
        op_status = self.state_machine.update_state(
            machine_id=machine_id,
            predicted_class=predicted_class,
            confidence=confidence,
            is_anomaly=is_anomaly,
            signal_valid=True,
        )

        # 6. Recommendation
        recommendation = generate_maintenance_recommendation(
            operational_status=op_status,
            predicted_class=predicted_class,
            confidence=confidence,
            working_parameters=wp_dict,
        )

        # 7. Persistence & Alerting
        if persist:
            try:
                self.history_repo.record_observation(
                    machine_id=machine_id,
                    predicted_class=predicted_class,
                    operational_status=op_status.value,
                    source=source,
                    rms=features.get("rms"),
                    peak_to_peak=features.get("peak_to_peak"),
                    confidence=confidence,
                    anomaly_score=anomaly_score,
                    model_name=self.model_name,
                    model_version="1.0.0",
                    load_percent=wp_dict.get("load_percent"),
                    speed_rpm=wp_dict.get("rotational_speed_rpm"),
                    timestamp=now_iso,
                )
                self.alert_manager.trigger_alert_if_needed(
                    machine_id=machine_id,
                    operational_status=op_status.value,
                    predicted_class=predicted_class,
                    recommendation=recommendation,
                    model_version="1.0.0",
                )
            except Exception as e:
                logger.error(f"Persistence error: {e}")

        return InferenceResult(
            machine_id=machine_id,
            timestamp=now_iso,
            source=source,
            signal_quality=quality.status,
            predicted_class=predicted_class,
            operational_status=op_status,
            confidence=confidence,
            anomaly_score=anomaly_score,
            model_name=self.model_name,
            model_version="1.0.0",
            features=features,
            working_parameters=wp_dict,
            maintenance_recommendation=recommendation,
            limitations=[
                "Model trained on two-stage gearbox setup under steady load; dynamic transients may affect confidence.",
                "Softmax probabilities represent relative classification certainty, not a calibrated failure probability."
            ],
        )

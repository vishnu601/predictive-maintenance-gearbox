"""
Data schemas and type definitions for predictive maintenance and condition monitoring.
Accommodates vibration data time-series and mechanical working parameters.
"""

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class OperationalStatus(str, Enum):
    NORMAL = "NORMAL"
    WARNING = "WARNING"
    FAULT = "FAULT"
    SENSOR_ERROR = "SENSOR_ERROR"
    INSUFFICIENT_DATA = "INSUFFICIENT_DATA"


class FaultClass(str, Enum):
    HEALTHY = "Healthy"
    BROKEN_TOOTH = "BrokenTooth"
    UNAVAILABLE = "UNAVAILABLE"


class GearboxWorkingParameters(BaseModel):
    """Operational parameters of the monitored gearbox equipment."""
    machine_id: str = Field(default="GEARBOX_01", description="Identifier of the mechanical asset")
    load_percent: Optional[float] = Field(default=None, description="Operating load percentage (e.g. 0 to 90%)")
    rotational_speed_rpm: Optional[float] = Field(default=None, description="Input/Shaft rotational speed in RPM")
    sensor_channel: Optional[int] = Field(default=1, description="Accelerometer channel index (1 to 4)")
    sampling_rate_hz: Optional[float] = Field(default=5000.0, description="Sensor sampling rate in Hertz")
    operating_temperature_c: Optional[float] = Field(default=None, description="Bearing/gearbox temperature in Celsius")


class SignalQualityReport(BaseModel):
    """Quality and validation report for a raw vibration signal segment."""
    is_valid: bool
    status: str = Field(description="VALID, FLATLINE, CLIPPED, HAS_NANS, or TOO_SHORT")
    details: str
    sample_count: int
    mean_val: float
    std_val: float


class VibrationFeatures(BaseModel):
    """Extracted time-domain and frequency-domain vibration features."""
    # Time domain
    mean: float
    std: float
    rms: float
    min_val: float
    max_val: float
    peak_to_peak: float
    absolute_peak: float
    crest_factor: float
    skewness: float
    kurtosis: float
    shape_factor: float
    impulse_factor: float

    # Frequency domain
    dominant_frequency: Optional[float] = None
    spectral_centroid: Optional[float] = None
    spectral_bandwidth: Optional[float] = None
    spectral_energy: Optional[float] = None
    spectral_entropy: Optional[float] = None
    peak_spectral_magnitude: Optional[float] = None


class VibrationDataInput(BaseModel):
    """Complete inference payload containing vibration samples and operational parameters."""
    samples: List[float] = Field(..., description="Vibration time series raw acceleration data")
    working_parameters: Optional[GearboxWorkingParameters] = Field(
        default_factory=GearboxWorkingParameters,
        description="Gearbox operational parameters (load, speed, channel)"
    )
    source: str = Field(default="live_sensor", description="Source: serial_sensor, csv_replay, demo_source, uploaded_file")
    timestamp: Optional[datetime] = Field(default_factory=lambda: datetime.now(timezone.utc))


class InferenceResult(BaseModel):
    """Standardized output produced by the unified inference engine."""
    machine_id: str
    timestamp: str
    source: str
    signal_quality: str
    predicted_class: str
    operational_status: OperationalStatus
    confidence: Optional[float] = None
    anomaly_score: Optional[float] = None
    model_name: str
    model_version: str
    features: Dict[str, float] = Field(default_factory=dict)
    working_parameters: Optional[Dict[str, Any]] = None
    maintenance_recommendation: str
    limitations: List[str] = Field(default_factory=list)


class AlertRecord(BaseModel):
    """Persistent alert record stored in SQLite."""
    alert_id: str
    machine_id: str
    severity: str  # WARNING or CRITICAL
    status: str  # ACTIVE, ACKNOWLEDGED, RESOLVED
    trigger_reason: str
    recommendation: str
    created_at: str
    acknowledged_at: Optional[str] = None
    resolved_at: Optional[str] = None
    model_version: Optional[str] = None

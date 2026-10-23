"""
Signal validation and sanity checking for vibration time-series.
Rejects invalid, disconnected, clipping, or corrupted signals before inference.
"""

from typing import Union
import numpy as np
from ..schemas import SignalQualityReport


def validate_vibration_signal(
    signal: Union[np.ndarray, list],
    min_length: int = 1024,
    flatline_std_threshold: float = 1e-5,
    clipping_percentile: float = 99.9,
    max_amplitude_g: float = 100.0,
) -> SignalQualityReport:
    """
    Validates a vibration signal segment for physical and numerical sanity.
    
    Checks:
    - Minimum length requirement
    - NaN or Inf values
    - Flatline / sensor disconnect (near-zero standard deviation)
    - Sensor saturation / severe clipping
    """
    if isinstance(signal, list):
        signal = np.array(signal, dtype=np.float64)
    elif not isinstance(signal, np.ndarray):
        return SignalQualityReport(
            is_valid=False,
            status="INVALID_TYPE",
            details=f"Signal must be a list or numpy array, got {type(signal)}",
            sample_count=0,
            mean_val=0.0,
            std_val=0.0,
        )

    # Dimensionality check
    if signal.ndim > 1:
        signal = signal.flatten()

    sample_count = len(signal)
    if sample_count < min_length:
        return SignalQualityReport(
            is_valid=False,
            status="TOO_SHORT",
            details=f"Signal length {sample_count} is less than required minimum {min_length}",
            sample_count=sample_count,
            mean_val=0.0,
            std_val=0.0,
        )

    # Check for NaNs or Infs
    if np.isnan(signal).any() or np.isinf(signal).any():
        return SignalQualityReport(
            is_valid=False,
            status="HAS_NANS",
            details="Signal contains NaN or Infinite values indicating sensor transmission error",
            sample_count=sample_count,
            mean_val=0.0,
            std_val=0.0,
        )

    mean_val = float(np.mean(signal))
    std_val = float(np.std(signal))

    # Flatline / Disconnected Sensor check
    if std_val < flatline_std_threshold:
        return SignalQualityReport(
            is_valid=False,
            status="FLATLINE",
            details=f"Signal variation standard deviation ({std_val:.2e}) is near zero, indicating sensor disconnect or frozen ADC.",
            sample_count=sample_count,
            mean_val=mean_val,
            std_val=std_val,
        )

    # Clipping / Sensor Saturation check
    abs_peak = float(np.max(np.abs(signal)))
    if abs_peak > max_amplitude_g:
        return SignalQualityReport(
            is_valid=False,
            status="CLIPPED",
            details=f"Signal absolute peak ({abs_peak:.1f} g) exceeds maximum operating limit ({max_amplitude_g} g).",
            sample_count=sample_count,
            mean_val=mean_val,
            std_val=std_val,
        )

    return SignalQualityReport(
        is_valid=True,
        status="VALID",
        details="Signal passed all physical and numerical validation checks.",
        sample_count=sample_count,
        mean_val=mean_val,
        std_val=std_val,
    )

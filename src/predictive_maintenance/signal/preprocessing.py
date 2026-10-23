"""
Signal preprocessing module: detrending, filtering, and normalization.
"""

from typing import Optional, Tuple
import numpy as np
from scipy import signal as sp_signal


def detrend_signal(data: np.ndarray) -> np.ndarray:
    """Removes DC bias / mean offset from vibration time series."""
    return data - np.mean(data)


def normalize_signal(data: np.ndarray, eps: float = 1e-8) -> Tuple[np.ndarray, float, float]:
    """
    Applies z-score standardization (zero mean, unit variance).
    Returns (normalized_signal, mean, std).
    """
    mean = float(np.mean(data))
    std = float(np.std(data))
    if std < eps:
        std = 1.0
    normalized = (data - mean) / std
    return normalized, mean, std


def apply_bandpass_filter(
    data: np.ndarray,
    lowcut: float,
    highcut: float,
    sampling_rate_hz: float,
    order: int = 4
) -> np.ndarray:
    """Applies a Butterworth bandpass filter to isolate vibration frequency bands."""
    nyquist = 0.5 * sampling_rate_hz
    low = lowcut / nyquist
    high = highcut / nyquist
    if low <= 0 or high >= 1.0 or low >= high:
        # Invalid band parameters; return unfiltered signal
        return data
    b, a = sp_signal.butter(order, [low, high], btype="band")
    return sp_signal.filtfilt(b, a, data)


def preprocess_window(
    window: np.ndarray,
    detrend: bool = True,
    normalize: bool = True,
    filter_bands: Optional[Tuple[float, float, float]] = None
) -> np.ndarray:
    """
    Standard preprocessing pipeline applied to an individual vibration window.
    """
    proc = np.array(window, dtype=np.float64)
    if detrend:
        proc = detrend_signal(proc)
    if filter_bands is not None:
        low, high, fs = filter_bands
        proc = apply_bandpass_filter(proc, low, high, fs)
    if normalize:
        proc, _, _ = normalize_signal(proc)
    return proc

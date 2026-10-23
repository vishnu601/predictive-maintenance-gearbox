"""
Feature engineering module for mechanical condition monitoring and fault diagnosis.
Calculates time-domain statistical metrics and frequency-domain spectral metrics.
"""

from typing import Dict, Optional, Tuple
import numpy as np
from scipy import stats


def calculate_time_domain_features(signal: np.ndarray, eps: float = 1e-9) -> Dict[str, float]:
    """
    Computes time-domain statistical indicators from a vibration window.

    Kurtosis convention:
    Uses Fisher's excess kurtosis (normal distribution = 0.0, heavy tails > 0.0).
    Also includes standard Pearson kurtosis (normal distribution = 3.0).
    """
    signal = np.asarray(signal, dtype=np.float64)
    n = len(signal)
    if n == 0:
        return {}

    mean_val = float(np.mean(signal))
    std_val = float(np.std(signal))
    rms_val = float(np.sqrt(np.mean(signal ** 2)))
    min_val = float(np.min(signal))
    max_val = float(np.max(signal))
    peak_to_peak = max_val - min_val
    abs_peak = float(np.max(np.abs(signal)))
    mean_abs = float(np.mean(np.abs(signal)))

    # Safe division guards
    safe_rms = rms_val if rms_val > eps else eps
    safe_mean_abs = mean_abs if mean_abs > eps else eps
    safe_std = std_val if std_val > eps else eps

    crest_factor = abs_peak / safe_rms
    shape_factor = safe_rms / safe_mean_abs
    impulse_factor = abs_peak / safe_mean_abs

    # Higher order moments
    skewness = float(stats.skew(signal, bias=False)) if safe_std > eps else 0.0
    # Fisher's excess kurtosis (bias=False, fisher=True): Gaussian = 0
    excess_kurtosis = float(stats.kurtosis(signal, fisher=True, bias=False)) if safe_std > eps else 0.0
    pearson_kurtosis = excess_kurtosis + 3.0

    return {
        "mean": mean_val,
        "std": std_val,
        "rms": rms_val,
        "min": min_val,
        "max": max_val,
        "peak_to_peak": peak_to_peak,
        "abs_peak": abs_peak,
        "crest_factor": crest_factor,
        "skewness": skewness,
        "kurtosis": excess_kurtosis,  # Excess kurtosis (Fisher)
        "pearson_kurtosis": pearson_kurtosis,
        "shape_factor": shape_factor,
        "impulse_factor": impulse_factor,
    }


def calculate_frequency_domain_features(
    signal: np.ndarray,
    sampling_rate_hz: float = 5000.0,
    eps: float = 1e-12
) -> Dict[str, float]:
    """
    Computes spectral features using one-sided Fast Fourier Transform (FFT).
    """
    signal = np.asarray(signal, dtype=np.float64)
    n = len(signal)
    if n == 0:
        return {}

    # One-sided FFT
    fft_vals = np.fft.rfft(signal)
    fft_mag = np.abs(fft_vals) / n
    fft_mag[1:] *= 2.0  # Conserve energy for one-sided spectrum
    freqs = np.fft.rfftfreq(n, d=1.0 / sampling_rate_hz)

    # Power Spectral Density approximation
    power_spectrum = fft_mag ** 2
    total_power = np.sum(power_spectrum)
    safe_power = total_power if total_power > eps else eps

    # Dominant peak frequency and magnitude
    peak_idx = int(np.argmax(fft_mag))
    dominant_frequency = float(freqs[peak_idx])
    peak_spectral_magnitude = float(fft_mag[peak_idx])

    # Spectral Centroid (Center of gravity of spectrum)
    spectral_centroid = float(np.sum(freqs * power_spectrum) / safe_power)

    # Spectral Spread / Bandwidth (Variance around centroid)
    spectral_bandwidth = float(np.sqrt(np.sum(((freqs - spectral_centroid) ** 2) * power_spectrum) / safe_power))

    # Spectral Energy
    spectral_energy = float(total_power)

    # Spectral Entropy (Distribution flatness of spectrum)
    norm_psd = power_spectrum / safe_power
    norm_psd = norm_psd[norm_psd > 0]
    spectral_entropy = float(-np.sum(norm_psd * np.log2(norm_psd + eps)))

    # Frequency band energy ratios (Low: 0-20% Nyquist, Mid: 20-60%, High: 60-100%)
    nyquist = 0.5 * sampling_rate_hz
    low_band_mask = freqs <= (0.2 * nyquist)
    mid_band_mask = (freqs > (0.2 * nyquist)) & (freqs <= (0.6 * nyquist))
    high_band_mask = freqs > (0.6 * nyquist)

    low_band_ratio = float(np.sum(power_spectrum[low_band_mask]) / safe_power)
    mid_band_ratio = float(np.sum(power_spectrum[mid_band_mask]) / safe_power)
    high_band_ratio = float(np.sum(power_spectrum[high_band_mask]) / safe_power)

    return {
        "dominant_frequency": dominant_frequency,
        "peak_spectral_magnitude": peak_spectral_magnitude,
        "spectral_centroid": spectral_centroid,
        "spectral_bandwidth": spectral_bandwidth,
        "spectral_energy": spectral_energy,
        "spectral_entropy": spectral_entropy,
        "low_band_ratio": low_band_ratio,
        "mid_band_ratio": mid_band_ratio,
        "high_band_ratio": high_band_ratio,
    }


def extract_features(
    signal: np.ndarray,
    sampling_rate_hz: Optional[float] = 5000.0
) -> Dict[str, float]:
    """
    Extracts all supported time-domain and frequency-domain features into a unified dictionary.
    """
    features = calculate_time_domain_features(signal)
    if sampling_rate_hz is not None and sampling_rate_hz > 0:
        freq_features = calculate_frequency_domain_features(signal, sampling_rate_hz=sampling_rate_hz)
        features.update(freq_features)
    return features

"""
Unit tests for time-domain and frequency-domain feature engineering.
Verifies mathematical correctness on known synthetic signals.
"""

from pathlib import Path
import sys
import numpy as np
import pytest

root_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir / "src"))

from predictive_maintenance.signal.features import (
    calculate_frequency_domain_features,
    calculate_time_domain_features,
    extract_features,
)


def test_rms_on_known_sinusoid():
    # A sinusoid of amplitude A has theoretical RMS = A / sqrt(2)
    fs = 5000.0
    t = np.arange(5000) / fs
    amplitude = 4.0
    freq = 50.0  # Exactly 50 cycles in 1 second
    sig = amplitude * np.sin(2 * np.pi * freq * t)

    features = calculate_time_domain_features(sig)
    expected_rms = amplitude / np.sqrt(2.0)
    assert np.isclose(features["rms"], expected_rms, rtol=1e-3)


def test_peak_to_peak_amplitude():
    fs = 5000.0
    t = np.arange(5000) / fs
    amplitude = 3.5
    sig = amplitude * np.sin(2 * np.pi * 20 * t)

    features = calculate_time_domain_features(sig)
    expected_p2p = 2.0 * amplitude
    assert np.isclose(features["peak_to_peak"], expected_p2p, rtol=1e-3)


def test_constant_signal_division_by_zero_safety():
    # Flat signal where std=0 and mean_abs=constant
    sig = np.full(1024, 5.0)
    features = calculate_time_domain_features(sig)
    # Check that crest_factor, shape_factor, etc do not crash or produce NaN/Inf
    assert not np.isnan(features["crest_factor"])
    assert not np.isinf(features["crest_factor"])
    assert features["kurtosis"] == 0.0


def test_dominant_frequency_known_tone():
    fs = 5000.0
    t = np.arange(5000) / fs
    target_freq = 120.0
    sig = 2.0 * np.sin(2 * np.pi * target_freq * t)

    freq_feats = calculate_frequency_domain_features(sig, sampling_rate_hz=fs)
    assert np.isclose(freq_feats["dominant_frequency"], target_freq, atol=1.0)
    assert freq_feats["spectral_centroid"] > 0.0
    assert freq_feats["spectral_energy"] > 0.0


def test_extract_features_unified():
    sig = np.random.normal(0, 1.0, size=2048)
    all_feats = extract_features(sig, sampling_rate_hz=5000.0)
    assert "rms" in all_feats
    assert "kurtosis" in all_feats
    assert "dominant_frequency" in all_feats
    assert "spectral_entropy" in all_feats

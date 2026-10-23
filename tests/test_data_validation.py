"""
Unit tests for vibration signal validation and quality checks.
"""

from pathlib import Path
import sys
import numpy as np
import pytest

root_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir / "src"))

from predictive_maintenance.signal.validators import validate_vibration_signal


def test_valid_signal():
    # 2048 samples of Gaussian noise with realistic variance
    sig = np.random.normal(0, 1.5, size=2048)
    report = validate_vibration_signal(sig, min_length=1024)
    assert report.is_valid is True
    assert report.status == "VALID"
    assert report.sample_count == 2048


def test_too_short_signal():
    sig = np.random.normal(0, 1.0, size=500)
    report = validate_vibration_signal(sig, min_length=1024)
    assert report.is_valid is False
    assert report.status == "TOO_SHORT"


def test_flatline_sensor_disconnect():
    # Constant or nearly constant signal (frozen ADC / wire unplugged)
    sig = np.zeros(2048)
    report = validate_vibration_signal(sig, min_length=1024)
    assert report.is_valid is False
    assert report.status == "FLATLINE"


def test_nans_and_infs_signal():
    sig = np.random.normal(0, 1.0, size=2048)
    sig[100] = np.nan
    report = validate_vibration_signal(sig, min_length=1024)
    assert report.is_valid is False
    assert report.status == "HAS_NANS"

    sig[100] = 0.0
    sig[200] = np.inf
    report_inf = validate_vibration_signal(sig, min_length=1024)
    assert report_inf.is_valid is False
    assert report_inf.status == "HAS_NANS"


def test_clipping_signal():
    sig = np.random.normal(0, 1.0, size=2048)
    sig[50] = 150.0  # Exceeds max 100g limit
    report = validate_vibration_signal(sig, min_length=1024, max_amplitude_g=100.0)
    assert report.is_valid is False
    assert report.status == "CLIPPED"

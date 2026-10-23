"""
Unit tests for signal preprocessing: detrending, normalization, and window processing.
"""

from pathlib import Path
import sys
import numpy as np
import pytest

root_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir / "src"))

from predictive_maintenance.signal.preprocessing import detrend_signal, normalize_signal, preprocess_window


def test_detrend_removes_dc_bias():
    t = np.linspace(0, 1, 2048)
    # Sinusoid with 5.0 DC offset
    raw = 5.0 + 2.0 * np.sin(2 * np.pi * 50 * t)
    detrended = detrend_signal(raw)
    assert np.isclose(np.mean(detrended), 0.0, atol=1e-12)


def test_normalize_signal():
    raw = np.random.normal(10.0, 3.0, size=2048)
    norm, mean, std = normalize_signal(raw)
    assert np.isclose(mean, 10.0, atol=0.2)
    assert np.isclose(std, 3.0, atol=0.2)
    assert np.isclose(np.mean(norm), 0.0, atol=1e-7)
    assert np.isclose(np.std(norm), 1.0, atol=1e-7)


def test_preprocess_window():
    raw = 12.0 + np.random.normal(0, 2.0, size=2048)
    proc = preprocess_window(raw, detrend=True, normalize=True)
    assert len(proc) == 2048
    assert np.isclose(np.mean(proc), 0.0, atol=1e-7)
    assert np.isclose(np.std(proc), 1.0, atol=1e-7)

"""
Unit tests for operational state transitions, alert confirmation rules, and recovery logic.
"""

from pathlib import Path
import sys
import pytest

root_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir / "src"))

from predictive_maintenance.monitoring.rules import MonitoringStateMachine
from predictive_maintenance.schemas import OperationalStatus


def test_consecutive_windows_required_for_fault():
    # Require 3 consecutive fault windows before state flips to FAULT
    sm = MonitoringStateMachine(
        fault_confirmation_windows=3,
        recovery_confirmation_windows=5,
        fault_confidence_threshold=0.70,
    )
    machine = "GEARBOX_TEST"

    # Window 1: Fault predicted -> State becomes WARNING (provisional)
    s1 = sm.update_state(machine, "BrokenTooth", confidence=0.90, is_anomaly=True, signal_valid=True)
    assert s1 == OperationalStatus.WARNING

    # Window 2: Fault predicted -> State still WARNING
    s2 = sm.update_state(machine, "BrokenTooth", confidence=0.85, is_anomaly=True, signal_valid=True)
    assert s2 == OperationalStatus.WARNING

    # Window 3: 3rd consecutive fault -> State flips to FAULT!
    s3 = sm.update_state(machine, "BrokenTooth", confidence=0.92, is_anomaly=True, signal_valid=True)
    assert s3 == OperationalStatus.FAULT


def test_recovery_requires_multiple_normal_windows():
    sm = MonitoringStateMachine(
        fault_confirmation_windows=2,
        recovery_confirmation_windows=3,
        fault_confidence_threshold=0.70,
    )
    machine = "GEARBOX_TEST"

    # Trip to FAULT
    sm.update_state(machine, "BrokenTooth", 0.9, True, True)
    sm.update_state(machine, "BrokenTooth", 0.9, True, True)
    assert sm.get_status(machine) == OperationalStatus.FAULT

    # Window 1 normal: Still in WARNING/investigation (not immediate normal)
    s1 = sm.update_state(machine, "Healthy", 0.95, False, True)
    assert s1 == OperationalStatus.WARNING

    # Window 2 normal: Still in WARNING
    s2 = sm.update_state(machine, "Healthy", 0.95, False, True)
    assert s2 == OperationalStatus.WARNING

    # Window 3 normal: 3rd consecutive normal -> Recovers to NORMAL!
    s3 = sm.update_state(machine, "Healthy", 0.95, False, True)
    assert s3 == OperationalStatus.NORMAL


def test_sensor_error_on_invalid_signal():
    sm = MonitoringStateMachine()
    s = sm.update_state("GEARBOX_TEST", "Healthy", 0.99, False, signal_valid=False)
    assert s == OperationalStatus.SENSOR_ERROR

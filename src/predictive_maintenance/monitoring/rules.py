"""
Operational status decision rules and maintenance recommendation engine.
Translates model predictions, anomaly scores, and consecutive window history
into verified machine operational states and actionable recommendations.
"""

from typing import Dict, List, Optional
from ..schemas import OperationalStatus


class MonitoringStateMachine:
    """
    State machine enforcing consecutive-window rules to prevent false alarms.
    
    Transitions:
    - NORMAL -> WARNING: 1 anomaly window or uncertain prediction
    - WARNING/NORMAL -> FAULT: N consecutive validated FAULT predictions
    - FAULT/WARNING -> NORMAL: M consecutive validated NORMAL windows
    """

    def __init__(
        self,
        fault_confirmation_windows: int = 3,
        recovery_confirmation_windows: int = 5,
        fault_confidence_threshold: float = 0.70,
    ):
        self.fault_confirmation_windows = fault_confirmation_windows
        self.recovery_confirmation_windows = recovery_confirmation_windows
        self.fault_confidence_threshold = fault_confidence_threshold

        # Internal state per machine
        self._current_status: Dict[str, OperationalStatus] = {}
        self._consecutive_faults: Dict[str, int] = {}
        self._consecutive_normals: Dict[str, int] = {}
        self._in_recovery: Dict[str, bool] = {}

    def update_state(
        self,
        machine_id: str,
        predicted_class: str,
        confidence: Optional[float],
        is_anomaly: bool,
        signal_valid: bool,
    ) -> OperationalStatus:
        """
        Evaluates the next operational status for a machine given latest window inference.
        """
        if not signal_valid:
            self._current_status[machine_id] = OperationalStatus.SENSOR_ERROR
            return OperationalStatus.SENSOR_ERROR

        current = self._current_status.get(machine_id, OperationalStatus.NORMAL)
        conf = confidence if confidence is not None else 0.5
        is_fault_candidate = (predicted_class == "BrokenTooth") and (conf >= self.fault_confidence_threshold)

        if is_fault_candidate:
            self._consecutive_faults[machine_id] = self._consecutive_faults.get(machine_id, 0) + 1
            self._consecutive_normals[machine_id] = 0

            if self._consecutive_faults[machine_id] >= self.fault_confirmation_windows:
                new_status = OperationalStatus.FAULT
                self._in_recovery[machine_id] = True
            else:
                new_status = OperationalStatus.WARNING
        elif predicted_class == "Healthy" and not is_anomaly:
            self._consecutive_normals[machine_id] = self._consecutive_normals.get(machine_id, 0) + 1
            self._consecutive_faults[machine_id] = 0

            if self._in_recovery.get(machine_id, False) or current == OperationalStatus.FAULT:
                # Require M consecutive normals to recover from FAULT
                if self._consecutive_normals[machine_id] >= self.recovery_confirmation_windows:
                    new_status = OperationalStatus.NORMAL
                    self._in_recovery[machine_id] = False
                else:
                    self._in_recovery[machine_id] = True
                    new_status = OperationalStatus.WARNING
            else:
                new_status = OperationalStatus.NORMAL
        else:
            # Anomaly detected or uncertain prediction
            new_status = OperationalStatus.WARNING
            self._consecutive_normals[machine_id] = 0

        self._current_status[machine_id] = new_status
        return new_status

    def get_status(self, machine_id: str) -> OperationalStatus:
        return self._current_status.get(machine_id, OperationalStatus.NORMAL)

    def reset(self, machine_id: Optional[str] = None):
        if machine_id:
            self._current_status.pop(machine_id, None)
            self._consecutive_faults.pop(machine_id, None)
            self._consecutive_normals.pop(machine_id, None)
            self._in_recovery.pop(machine_id, None)
        else:
            self._current_status.clear()
            self._consecutive_faults.clear()
            self._consecutive_normals.clear()
            self._in_recovery.clear()


def generate_maintenance_recommendation(
    operational_status: OperationalStatus,
    predicted_class: str,
    confidence: Optional[float] = None,
    working_parameters: Optional[Dict] = None,
) -> str:
    """
    Generates domain-specific, cautious mechanical inspection and maintenance actions.
    """
    load_info = ""
    if working_parameters and working_parameters.get("load_percent") is not None:
        load_info = f" (Operating under {working_parameters['load_percent']}% load)"

    if operational_status == OperationalStatus.SENSOR_ERROR:
        return (
            "Sensor signal invalid or disconnected. Check accelerometer magnetic/stud mounting, "
            "inspect coaxial wiring and shield grounding, and verify DAQ ADC input scaling."
        )

    if operational_status == OperationalStatus.INSUFFICIENT_DATA:
        return "Insufficient vibration samples collected to form an analysis window. Maintain continuous streaming."

    if operational_status == OperationalStatus.FAULT:
        if predicted_class == "BrokenTooth":
            return (
                f"CRITICAL: Gear tooth damage or crack detected with high confidence{load_info}. "
                "Schedule immediate mechanical inspection: 1) Open gearbox inspection port and visually check "
                "spur gear teeth for chipping, micro-pitting, or root cracks; 2) Check oil sump for metal debris "
                "or particle accumulation; 3) Verify backlash and gear mesh contact pattern."
            )
        return (
            f"FAULT state triggered: {predicted_class}{load_info}. Schedule physical inspection "
            "of rotating components and verify mounting torque."
        )

    if operational_status == OperationalStatus.WARNING:
        if predicted_class == "BrokenTooth":
            return (
                f"WARNING: Developing fault pattern suspected ({predicted_class}){load_info}. "
                "Elevate monitoring frequency. Verify oil lubrication level and schedule an endoscopic visual inspection."
            )
        return (
            f"WARNING: Statistically abnormal vibration detected{load_info}. "
            "Check for coupling misalignment, loose foundation fasteners, or temporary transient load surges. "
            "Collect a secondary confirmation recording."
        )

    # NORMAL
    return (
        f"NORMAL: Gearbox vibration patterns are consistent with healthy operating baseline{load_info}. "
        "Continue routine predictive maintenance monitoring."
    )

"""
Synthetic demonstration signal source.
Generates simulated mechanical vibration waveforms for pipeline testing,
UI validation, and demonstration without physical hardware or downloaded datasets.
Explicitly tagged as SYNTHETIC_DEMO per non-fabrication rules.
"""

import threading
import time
from typing import Any, Dict, Optional
import numpy as np
from .base import BaseSensorSource


class SyntheticDemoSensorSource(BaseSensorSource):
    """
    Simulated vibration generator modeling healthy gear meshing or tooth crack impacts.
    """

    def __init__(
        self,
        mode: str = "Healthy",
        sampling_rate_hz: float = 5000.0,
        shaft_speed_hz: float = 30.0,
        gear_teeth: int = 20,
        chunk_size: int = 256,
        buffer_maxlen: int = 100000,
    ):
        super().__init__(buffer_maxlen=buffer_maxlen)
        self.mode = mode
        self.sampling_rate_hz = sampling_rate_hz
        self.shaft_speed_hz = shaft_speed_hz
        self.gear_mesh_freq = shaft_speed_hz * gear_teeth  # e.g. 600 Hz
        self.chunk_size = chunk_size
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._time_counter = 0.0

    def connect(self) -> bool:
        if self.is_active:
            return True
        self.buffer.clear()
        self._stop_event.clear()
        self.is_active = True
        self._time_counter = 0.0
        self._thread = threading.Thread(target=self._stream_loop, daemon=True)
        self._thread.start()
        return True

    def set_mode(self, mode: str):
        """Switches between 'Healthy' and 'BrokenTooth' simulated dynamics."""
        self.mode = mode

    def _stream_loop(self):
        dt = 1.0 / self.sampling_rate_hz
        sleep_interval = float(self.chunk_size) / float(self.sampling_rate_hz)

        while not self._stop_event.is_set():
            t = self._time_counter + np.arange(self.chunk_size) * dt
            self._time_counter += self.chunk_size * dt

            # 1. Base gear meshing tone (600 Hz)
            meshing = 1.5 * np.sin(2 * np.pi * self.gear_mesh_freq * t)
            # 2. Shaft rotation baseline (30 Hz)
            shaft = 0.5 * np.sin(2 * np.pi * self.shaft_speed_hz * t)
            # 3. Background vibration noise
            noise = np.random.normal(0, 0.4, size=len(t))

            if self.mode == "BrokenTooth":
                # Add sidebands (modulation) around mesh freq + transient impacts at 1x shaft period
                modulation = (1.0 + 0.8 * np.sin(2 * np.pi * self.shaft_speed_hz * t))
                # Periodic sharp impact when broken tooth meshes
                impact_phase = (t * self.shaft_speed_hz) % 1.0
                impact = 4.0 * np.exp(-((impact_phase - 0.5) ** 2) / 0.002)
                chunk = (meshing * modulation) + shaft + impact + noise
            else:
                chunk = meshing + shaft + noise

            self.buffer.extend(chunk.tolist())
            time.sleep(sleep_interval)

        self.is_active = False

    def disconnect(self):
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.0)
        self.is_active = False

    def is_connected(self) -> bool:
        return self.is_active

    def get_metadata(self) -> Dict[str, Any]:
        return {
            "source_type": "SYNTHETIC_DEMO",
            "is_synthetic": True,
            "mode": self.mode,
            "sampling_rate_hz": self.sampling_rate_hz,
            "shaft_speed_hz": self.shaft_speed_hz,
            "buffer_size": self.buffer.size(),
            "warning": "Demonstration signal only. Not real machine sensor measurements.",
        }

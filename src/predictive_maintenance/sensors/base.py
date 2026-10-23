"""
Abstract base class and bounded ring buffer for vibration sensor data acquisition.
Guarantees constant memory consumption regardless of stream duration.
"""

from abc import ABC, abstractmethod
from collections import deque
import threading
from typing import Any, Dict, List, Optional
import numpy as np


class BoundedVibrationBuffer:
    """Thread-safe circular ring buffer for incoming vibration samples."""

    def __init__(self, maxlen: int = 100000):
        self.maxlen = maxlen
        self._buffer = deque(maxlen=maxlen)
        self._lock = threading.Lock()

    def append(self, sample: float):
        with self._lock:
            self._buffer.append(float(sample))

    def extend(self, samples: List[float]):
        with self._lock:
            self._buffer.extend([float(s) for s in samples])

    def get_latest_window(self, window_size: int = 2048) -> Optional[np.ndarray]:
        """Returns the most recent window of samples, or None if buffer has fewer samples."""
        with self._lock:
            if len(self._buffer) < window_size:
                return None
            # Return last window_size elements
            items = list(self._buffer)[-window_size:]
            return np.array(items, dtype=np.float64)

    def size(self) -> int:
        with self._lock:
            return len(self._buffer)

    def clear(self):
        with self._lock:
            self._buffer.clear()


class BaseSensorSource(ABC):
    """Abstract interface for all vibration data acquisition sources."""

    def __init__(self, buffer_maxlen: int = 100000):
        self.buffer = BoundedVibrationBuffer(maxlen=buffer_maxlen)
        self.is_active = False

    @abstractmethod
    def connect(self) -> bool:
        """Establishes connection to the sensor stream."""
        pass

    @abstractmethod
    def disconnect(self):
        """Safely terminates the sensor acquisition stream."""
        pass

    @abstractmethod
    def is_connected(self) -> bool:
        """Returns True if the sensor connection is active."""
        pass

    @abstractmethod
    def get_metadata(self) -> Dict[str, Any]:
        """Returns sensor configuration and sampling metadata."""
        pass

    def get_window(self, window_size: int = 2048) -> Optional[np.ndarray]:
        """Fetches the latest complete analysis window from the bounded buffer."""
        return self.buffer.get_latest_window(window_size)

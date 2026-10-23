"""
File replay sensor source.
Streams historical recording files into the bounded buffer to simulate live telemetry.
"""

from pathlib import Path
import threading
import time
from typing import Any, Dict, Optional
import numpy as np
from .base import BaseSensorSource
from ..data.loaders import load_vibration_file
from ..logging_config import setup_logger

logger = setup_logger(__name__)


class CSVReplaySensorSource(BaseSensorSource):
    """Replays an offline accelerometer file sample-by-sample or chunk-by-chunk."""

    def __init__(
        self,
        file_path: Path,
        sampling_rate_hz: float = 5000.0,
        selected_channel: int = 1,
        chunk_size: int = 256,
        loop: bool = True,
        buffer_maxlen: int = 100000,
    ):
        super().__init__(buffer_maxlen=buffer_maxlen)
        self.file_path = Path(file_path)
        self.sampling_rate_hz = sampling_rate_hz
        self.selected_channel = selected_channel
        self.chunk_size = chunk_size
        self.loop = loop
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._signal_data: np.ndarray = np.array([])
        self._file_metadata: Dict[str, Any] = {}

    def connect(self) -> bool:
        if self.is_active:
            return True

        if not self.file_path.exists():
            logger.error(f"Replay file not found: {self.file_path}")
            return False

        try:
            self._signal_data, self._file_metadata = load_vibration_file(
                self.file_path,
                selected_channel=self.selected_channel
            )
            self.buffer.clear()
            self._stop_event.clear()
            self.is_active = True
            self._thread = threading.Thread(target=self._stream_loop, daemon=True)
            self._thread.start()
            logger.info(f"Replay stream started for file {self.file_path.name}")
            return True
        except Exception as e:
            logger.error(f"Failed to start replay source: {e}")
            self.is_active = False
            return False

    def _stream_loop(self):
        n_samples = len(self._signal_data)
        idx = 0
        sleep_interval = float(self.chunk_size) / float(self.sampling_rate_hz)

        while not self._stop_event.is_set():
            if idx + self.chunk_size <= n_samples:
                chunk = self._signal_data[idx : idx + self.chunk_size]
                self.buffer.extend(chunk.tolist())
                idx += self.chunk_size
            else:
                if self.loop:
                    idx = 0
                else:
                    break
            time.sleep(sleep_interval)

        self.is_active = False

    def disconnect(self):
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.0)
        self.is_active = False
        logger.info("Replay stream stopped.")

    def is_connected(self) -> bool:
        return self.is_active

    def get_metadata(self) -> Dict[str, Any]:
        meta = self._file_metadata.copy()
        meta.update({
            "source_type": "csv_replay",
            "sampling_rate_hz": self.sampling_rate_hz,
            "buffer_size": self.buffer.size(),
            "is_active": self.is_active,
        })
        return meta

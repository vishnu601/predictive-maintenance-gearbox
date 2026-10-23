"""
Hardware serial sensor integration module.
Reads live vibration streaming from a microcontroller (e.g. Arduino, STM32, ESP32)
or DAQ ADC device over a configurable UART/USB serial interface.
"""

from pathlib import Path
import threading
import time
from typing import Any, Dict, Optional
import serial
from .base import BaseSensorSource
from ..logging_config import setup_logger

logger = setup_logger(__name__)


class SerialSensorSource(BaseSensorSource):
    """
    UART/USB Serial data acquisition client.
    Supports framed ASCII/CSV (timestamp,value) or JSON packets.
    """

    def __init__(
        self,
        port: str = "/dev/tty.usbserial-0001",
        baud_rate: int = 115200,
        timeout_sec: float = 1.0,
        sampling_rate_hz: float = 5000.0,
        format_type: str = "csv",
        buffer_maxlen: int = 100000,
    ):
        super().__init__(buffer_maxlen=buffer_maxlen)
        self.port = port
        self.baud_rate = baud_rate
        self.timeout_sec = timeout_sec
        self.sampling_rate_hz = sampling_rate_hz
        self.format_type = format_type

        self._serial_conn: Optional[serial.Serial] = None
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

    def connect(self) -> bool:
        if self.is_active:
            return True

        try:
            self._serial_conn = serial.Serial(
                port=self.port,
                baudrate=self.baud_rate,
                timeout=self.timeout_sec,
            )
            self._serial_conn.reset_input_buffer()
            self._stop_event.clear()
            self.is_active = True
            self._thread = threading.Thread(target=self._read_loop, daemon=True)
            self._thread.start()
            logger.info(f"Connected to physical serial port {self.port} @ {self.baud_rate} baud.")
            return True
        except Exception as e:
            logger.error(f"Failed to open serial port {self.port}: {e}")
            self.is_active = False
            return False

    def _read_loop(self):
        while not self._stop_event.is_set() and self._serial_conn and self._serial_conn.is_open:
            try:
                line = self._serial_conn.readline().decode("utf-8", errors="ignore").strip()
                if not line:
                    continue
                # Parse sample
                sample_val = self._parse_line(line)
                if sample_val is not None:
                    self.buffer.append(sample_val)
            except Exception as e:
                logger.warning(f"Error reading serial packet: {e}")
                time.sleep(0.01)

        self.is_active = False

    def _parse_line(self, line: str) -> Optional[float]:
        """Parses a single line: either 'val' or 'timestamp,val' or JSON."""
        try:
            if "," in line:
                parts = line.split(",")
                return float(parts[-1].strip())
            return float(line.strip())
        except ValueError:
            return None

    def disconnect(self):
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.0)
        if self._serial_conn and self._serial_conn.is_open:
            try:
                self._serial_conn.close()
            except Exception:
                pass
        self.is_active = False
        logger.info(f"Disconnected from serial port {self.port}.")

    def is_connected(self) -> bool:
        return self.is_active and bool(self._serial_conn and self._serial_conn.is_open)

    def get_metadata(self) -> Dict[str, Any]:
        return {
            "source_type": "serial_hardware",
            "port": self.port,
            "baud_rate": self.baud_rate,
            "sampling_rate_hz": self.sampling_rate_hz,
            "format": self.format_type,
            "buffer_size": self.buffer.size(),
            "is_connected": self.is_connected(),
        }

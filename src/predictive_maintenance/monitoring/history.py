"""
SQLite database repository for persisting machine observation history.
Enables historical vibration trend visualization, auditing, and CSV export.
"""

from datetime import datetime, timezone
from pathlib import Path
import sqlite3
from typing import Any, Dict, List, Optional
import pandas as pd
from ..logging_config import setup_logger

logger = setup_logger(__name__)


class HistoryRepository:
    """Manages SQLite storage for vibration observations and machine states."""

    def __init__(self, db_path: Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self):
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS observations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    machine_id TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    source TEXT NOT NULL,
                    rms REAL,
                    peak_to_peak REAL,
                    predicted_class TEXT NOT NULL,
                    operational_status TEXT NOT NULL,
                    confidence REAL,
                    anomaly_score REAL,
                    model_name TEXT,
                    model_version TEXT,
                    load_percent REAL,
                    speed_rpm REAL
                );
            """)
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_obs_machine_ts ON observations(machine_id, timestamp);")
            conn.commit()

    def record_observation(
        self,
        machine_id: str,
        predicted_class: str,
        operational_status: str,
        source: str = "live_sensor",
        rms: Optional[float] = None,
        peak_to_peak: Optional[float] = None,
        confidence: Optional[float] = None,
        anomaly_score: Optional[float] = None,
        model_name: Optional[str] = None,
        model_version: Optional[str] = None,
        load_percent: Optional[float] = None,
        speed_rpm: Optional[float] = None,
        timestamp: Optional[str] = None,
    ) -> int:
        ts = timestamp or datetime.now(timezone.utc).isoformat()
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO observations (
                    machine_id, timestamp, source, rms, peak_to_peak,
                    predicted_class, operational_status, confidence,
                    anomaly_score, model_name, model_version,
                    load_percent, speed_rpm
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                machine_id, ts, source, rms, peak_to_peak,
                predicted_class, operational_status, confidence,
                anomaly_score, model_name, model_version,
                load_percent, speed_rpm
            ))
            conn.commit()
            return cursor.lastrowid

    def get_recent_observations(
        self,
        machine_id: Optional[str] = None,
        limit: int = 100,
    ) -> pd.DataFrame:
        """Retrieves recent observations as a pandas DataFrame."""
        with self._get_connection() as conn:
            query = "SELECT * FROM observations"
            params = []
            if machine_id:
                query += " WHERE machine_id = ?"
                params.append(machine_id)
            query += " ORDER BY id DESC LIMIT ?"
            params.append(limit)
            df = pd.read_sql_query(query, conn, params=params)
            return df

    def export_to_csv(self, output_path: Path, machine_id: Optional[str] = None) -> Path:
        """Exports observations table to CSV."""
        df = self.get_recent_observations(machine_id=machine_id, limit=50000)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        return output_path

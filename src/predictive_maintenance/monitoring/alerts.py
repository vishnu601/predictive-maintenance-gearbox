"""
Alert management and persistence module.
Implements alert deduplication, acknowledgement, and resolution workflows.
"""

from datetime import datetime, timezone
from pathlib import Path
import sqlite3
from typing import Any, Dict, List, Optional
import uuid
import pandas as pd
from ..logging_config import setup_logger

logger = setup_logger(__name__)


class AlertManager:
    """Manages alert creation, deduplication, and lifecycle."""

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
                CREATE TABLE IF NOT EXISTS alerts (
                    alert_id TEXT PRIMARY KEY,
                    machine_id TEXT NOT NULL,
                    severity TEXT NOT NULL,
                    status TEXT NOT NULL,
                    trigger_reason TEXT NOT NULL,
                    recommendation TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    acknowledged_at TEXT,
                    resolved_at TEXT,
                    model_version TEXT
                );
            """)
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_alerts_status ON alerts(machine_id, status);")
            conn.commit()

    def trigger_alert_if_needed(
        self,
        machine_id: str,
        operational_status: str,
        predicted_class: str,
        recommendation: str,
        model_version: Optional[str] = None,
    ) -> Optional[str]:
        """
        Triggers an alert if operational status is WARNING or FAULT,
        deduplicating against existing ACTIVE alerts for the machine.
        """
        if operational_status not in ["WARNING", "FAULT"]:
            # Machine is normal or recovering; auto-resolve active alerts if appropriate
            return None

        severity = "CRITICAL" if operational_status == "FAULT" else "WARNING"

        with self._get_connection() as conn:
            cursor = conn.cursor()
            # Check for existing ACTIVE alert of same severity
            cursor.execute("""
                SELECT alert_id FROM alerts
                WHERE machine_id = ? AND status = 'ACTIVE' AND severity = ?
            """, (machine_id, severity))
            existing = cursor.fetchone()
            if existing:
                # Deduplication: Alert already active
                return existing["alert_id"]

            alert_id = f"ALT-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}-{uuid.uuid4().hex[:6]}"
            created_at = datetime.now(timezone.utc).isoformat()
            trigger_reason = f"{operational_status} condition confirmed: {predicted_class}"

            cursor.execute("""
                INSERT INTO alerts (
                    alert_id, machine_id, severity, status, trigger_reason,
                    recommendation, created_at, model_version
                ) VALUES (?, ?, ?, 'ACTIVE', ?, ?, ?, ?)
            """, (alert_id, machine_id, severity, trigger_reason, recommendation, created_at, model_version))
            conn.commit()
            logger.info(f"Triggered new {severity} alert: {alert_id} for {machine_id}")
            return alert_id

    def acknowledge_alert(self, alert_id: str) -> bool:
        """Marks an active alert as acknowledged."""
        now = datetime.now(timezone.utc).isoformat()
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE alerts SET status = 'ACKNOWLEDGED', acknowledged_at = ?
                WHERE alert_id = ? AND status = 'ACTIVE'
            """, (now, alert_id))
            conn.commit()
            return cursor.rowcount > 0

    def resolve_alert(self, alert_id: str) -> bool:
        """Marks an alert as resolved."""
        now = datetime.now(timezone.utc).isoformat()
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE alerts SET status = 'RESOLVED', resolved_at = ?
                WHERE alert_id = ? AND status != 'RESOLVED'
            """, (now, alert_id))
            conn.commit()
            return cursor.rowcount > 0

    def get_alerts(
        self,
        machine_id: Optional[str] = None,
        status: Optional[str] = None,
        limit: int = 100,
    ) -> pd.DataFrame:
        """Queries alerts history with optional filtering."""
        with self._get_connection() as conn:
            query = "SELECT * FROM alerts WHERE 1=1"
            params = []
            if machine_id:
                query += " AND machine_id = ?"
                params.append(machine_id)
            if status:
                query += " AND status = ?"
                params.append(status)
            query += " ORDER BY created_at DESC LIMIT ?"
            params.append(limit)
            return pd.read_sql_query(query, conn, params=params)

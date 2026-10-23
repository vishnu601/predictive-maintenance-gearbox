"""
Dataset adapter for the US Data.gov / OpenEI Gearbox Fault Diagnosis Dataset.
Scans and normalizes the 20 experimental recordings across loads 0% to 90%.
"""

from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from .loaders import load_vibration_file
from ..logging_config import setup_logger

logger = setup_logger(__name__)


class GearboxDatasetAdapter:
    """
    Adapter for Prestige Institute / US DOE Gearbox Fault Diagnosis Data.
    Processes 4-sensor vibration readings from SpectraQuest Gearbox Diagnostics Simulator.
    """

    def __init__(self, raw_data_dir: Path, selected_channel: int = 1):
        self.raw_data_dir = Path(raw_data_dir)
        self.selected_channel = selected_channel
        self.healthy_dir = self.raw_data_dir / "Healthy Data"
        self.broken_dir = self.raw_data_dir / "BrokenTooth Data"

    def scan_recordings(self) -> List[Dict[str, Any]]:
        """
        Discovers all valid recordings in Healthy and BrokenTooth subdirectories.
        Returns a list of metadata records with paths, labels, and load conditions.
        """
        recordings = []

        # Find healthy files
        if self.healthy_dir.exists():
            for p in sorted(self.healthy_dir.glob("*.txt")):
                rec = self._parse_recording_metadata(p, label="Healthy")
                if rec:
                    recordings.append(rec)

        # Find broken tooth files
        if self.broken_dir.exists():
            for p in sorted(self.broken_dir.glob("*.txt")):
                rec = self._parse_recording_metadata(p, label="BrokenTooth")
                if rec:
                    recordings.append(rec)

        logger.info(f"Discovered {len(recordings)} gearbox recordings ({len([r for r in recordings if r['label'] == 'Healthy'])} Healthy, {len([r for r in recordings if r['label'] == 'BrokenTooth'])} BrokenTooth).")
        return recordings

    def _parse_recording_metadata(self, file_path: Path, label: str) -> Optional[Dict[str, Any]]:
        name = file_path.stem.lower()
        # Pattern: [h|b]30hz<load> e.g. h30hz0, b30hz90
        match = re.search(r"30hz(\d+)", name)
        if match:
            load_val = int(match.group(1))
        else:
            load_val = 0

        recording_id = f"{label.lower()}_load{load_val}pct_run"
        return {
            "recording_id": recording_id,
            "file_name": file_path.name,
            "file_path": str(file_path.resolve()),
            "label": label,
            "load_percent": float(load_val),
            "shaft_speed_hz": 30.0,
            "shaft_speed_rpm": 1800.0,
            "channel_index": self.selected_channel,
        }

    def load_all_recordings(self) -> List[Dict[str, Any]]:
        """
        Loads the signal data for every discovered recording.
        Returns a list of dicts with 'signal', 'metadata', and 'recording_id'.
        """
        meta_list = self.scan_recordings()
        loaded = []
        for meta in meta_list:
            try:
                sig, file_meta = load_vibration_file(
                    meta["file_path"],
                    selected_channel=self.selected_channel
                )
                meta.update(file_meta)
                loaded.append({
                    "recording_id": meta["recording_id"],
                    "signal": sig,
                    "metadata": meta,
                })
            except Exception as e:
                logger.error(f"Error loading recording {meta['file_name']}: {e}")
        return loaded

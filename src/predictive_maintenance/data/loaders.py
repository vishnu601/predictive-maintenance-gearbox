"""
Data ingestion and recording loader module.
Parses accelerometer text and CSV files, extracts operating parameters,
and enforces strict metadata and provenance tracking.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from ..logging_config import setup_logger

logger = setup_logger(__name__)


def load_vibration_file(
    file_path: Union[str, Path],
    selected_channel: int = 1,
    delimiter: Optional[str] = None
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Loads a raw vibration recording file.
    
    Supports:
    - Tab-separated, comma-separated, or whitespace-separated columns
    - Multi-channel accelerometer data (1-indexed channels)
    - Automatic delimiter sniffing
    
    Returns:
        (signal_array, metadata_dict)
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Vibration file not found: {path}")

    # Inspect first few lines to determine delimiter and headers
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        sample_lines = [f.readline() for _ in range(5)]

    # Infer delimiter if not provided
    if delimiter is None:
        first_line = sample_lines[0] if sample_lines else ""
        if "\t" in first_line:
            delimiter = r"\s+"  # Handles multiple tabs/spaces reliably
        elif "," in first_line:
            delimiter = ","
        else:
            delimiter = r"\s+"

    # Load dataframe
    try:
        df = pd.read_csv(path, sep=delimiter, header=None, engine="python")
    except Exception as e:
        logger.error(f"Failed to parse file {path}: {e}")
        raise ValueError(f"Could not parse vibration file {path.name}: {e}")

    # Remove non-numeric rows if headers were present
    df = df.apply(pd.to_numeric, errors="coerce").dropna()
    if df.empty:
        raise ValueError(f"File {path.name} contains no valid numeric vibration samples.")

    n_cols = df.shape[1]
    # Ensure selected channel is valid
    channel_col = selected_channel - 1
    if channel_col >= n_cols or channel_col < 0:
        logger.warning(
            f"Requested channel {selected_channel} out of bounds for {n_cols}-channel file. "
            f"Falling back to channel 1 (col 0)."
        )
        channel_col = 0

    signal = df.iloc[:, channel_col].to_numpy(dtype=np.float64)

    # Metadata extraction from filename (e.g. b30hz0.csv, BrokenTooth_Load30, Healthy_Load0)
    filename = path.stem.lower()
    load_percent = None
    # Parse standard load conventions in OpenEI dataset
    for load in [0, 10, 20, 30, 40, 50, 60, 70, 80, 90]:
        if f"load_{load}" in filename or f"load{load}" in filename or f"{load}hz" in filename or filename.endswith(f"_{load}"):
            load_percent = float(load)
            break

    label = "Unknown"
    if "broken" in filename or "crack" in filename or "fault" in filename:
        label = "BrokenTooth"
    elif "healthy" in filename or "normal" in filename:
        label = "Healthy"

    metadata = {
        "file_name": path.name,
        "file_path": str(path.resolve()),
        "total_samples": len(signal),
        "channel_index": selected_channel,
        "total_channels": n_cols,
        "inferred_load_percent": load_percent,
        "inferred_label": label,
    }

    return signal, metadata

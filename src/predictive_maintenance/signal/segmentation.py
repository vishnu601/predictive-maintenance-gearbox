"""
Signal segmentation and windowing module.
Slices continuous accelerometer signals into analysis windows while preserving metadata and provenance.
"""

from typing import Any, Dict, Generator, List, Optional
import numpy as np


def generate_windows(
    signal: np.ndarray,
    window_size: int = 2048,
    overlap_percent: float = 50.0,
    metadata: Optional[Dict[str, Any]] = None,
) -> Generator[Dict[str, Any], None, None]:
    """
    Generates sliding windows from a continuous 1D vibration signal.

    Args:
        signal: 1D numpy array of vibration readings.
        window_size: Number of samples per window.
        overlap_percent: Overlap between consecutive windows (0.0 to 95.0%).
        metadata: Optional dictionary with recording_id, load_percent, label, etc.

    Yields:
        Dictionary containing:
        - window_idx: Window sequential index within recording
        - data: 1D numpy array of shape (window_size,)
        - start_sample: Start index in original signal
        - end_sample: End index in original signal
        - metadata: Associated provenance metadata
    """
    signal = np.asarray(signal, dtype=np.float64).flatten()
    n_samples = len(signal)
    if n_samples < window_size:
        return

    step_size = max(1, int(window_size * (1.0 - (overlap_percent / 100.0))))
    base_meta = metadata.copy() if metadata else {}

    window_idx = 0
    start = 0
    while start + window_size <= n_samples:
        window_data = signal[start : start + window_size]
        yield {
            "window_idx": window_idx,
            "data": window_data,
            "start_sample": start,
            "end_sample": start + window_size,
            "metadata": base_meta,
        }
        start += step_size
        window_idx += 1


def segment_dataset(
    recordings: List[Dict[str, Any]],
    window_size: int = 2048,
    overlap_percent: float = 50.0,
) -> List[Dict[str, Any]]:
    """
    Segments a collection of recordings into structured windows.
    Each recording is a dict with 'signal' (np.ndarray) and 'metadata' (dict).
    """
    all_windows = []
    for rec in recordings:
        signal = rec["signal"]
        meta = rec.get("metadata", {})
        for win in generate_windows(signal, window_size, overlap_percent, meta):
            all_windows.append(win)
    return all_windows

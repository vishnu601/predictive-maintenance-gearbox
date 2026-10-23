"""
Data preparation script.
Segments raw vibration recordings, calculates features, performs group-aware splitting,
and persists processed datasets for training and evaluation.
"""

import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd

# Add project root to sys.path
root_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir / "src"))

from predictive_maintenance.config import settings
from predictive_maintenance.data.dataset_adapters import GearboxDatasetAdapter
from predictive_maintenance.data.splitting import group_train_val_test_split, verify_no_group_leakage
from predictive_maintenance.logging_config import setup_logger
from predictive_maintenance.signal.features import extract_features
from predictive_maintenance.signal.preprocessing import preprocess_window
from predictive_maintenance.signal.segmentation import generate_windows

logger = setup_logger("prepare_data")


def prepare_dataset(
    raw_dir: Path,
    output_dir: Path,
    channel: int = 1,
    window_size: int = 2048,
    overlap_percent: float = 50.0,
    sampling_rate_hz: float = 5000.0,
):
    output_dir.mkdir(parents=True, exist_ok=True)
    adapter = GearboxDatasetAdapter(raw_dir, selected_channel=channel)
    recordings = adapter.load_all_recordings()

    if not recordings:
        raise ValueError(f"No valid recordings found in {raw_dir}. Run scripts/download_dataset.py first.")

    logger.info(f"Loaded {len(recordings)} recordings. Starting windowing and feature extraction...")

    feature_rows = []
    windows_list = []
    groups = []
    labels = []

    for rec in recordings:
        sig = rec["signal"]
        meta = rec["metadata"]
        rec_id = rec["recording_id"]
        label = meta["label"]
        load_pct = meta.get("load_percent", 0.0)

        for win in generate_windows(sig, window_size=window_size, overlap_percent=overlap_percent, metadata=meta):
            raw_w = win["data"]
            # Extract features
            feats = extract_features(raw_w, sampling_rate_hz=sampling_rate_hz)
            feats["recording_id"] = rec_id
            feats["label"] = label
            feats["load_percent"] = load_pct
            feats["window_idx"] = win["window_idx"]

            feature_rows.append(feats)
            # Store preprocessed window for CNN
            proc_w = preprocess_window(raw_w, detrend=True, normalize=True)
            windows_list.append(proc_w)
            groups.append(rec_id)
            labels.append(label)

    df_features = pd.DataFrame(feature_rows)
    arr_windows = np.array(windows_list, dtype=np.float32)

    logger.info(f"Extracted {len(df_features)} total windows across {len(np.unique(groups))} unique recording groups.")
    logger.info(f"Class distribution: {dict(df_features['label'].value_counts())}")

    # Group-Aware Splitting
    train_idx, val_idx, test_idx = group_train_val_test_split(
        groups=groups,
        test_size=0.20,
        val_size=0.15,
        random_state=settings.system.get("seed", 42),
    )

    train_df = df_features.iloc[train_idx].reset_index(drop=True)
    val_df = df_features.iloc[val_idx].reset_index(drop=True)
    test_df = df_features.iloc[test_idx].reset_index(drop=True)

    train_win = arr_windows[train_idx]
    val_win = arr_windows[val_idx]
    test_win = arr_windows[test_idx]

    # Verify no leakage
    verify_no_group_leakage(
        list(train_df["recording_id"]),
        list(val_df["recording_id"]),
        list(test_df["recording_id"])
    )

    # Save to disk
    train_df.to_csv(output_dir / "train_features.csv", index=False)
    val_df.to_csv(output_dir / "val_features.csv", index=False)
    test_df.to_csv(output_dir / "test_features.csv", index=False)

    np.save(output_dir / "train_windows.npy", train_win)
    np.save(output_dir / "val_windows.npy", val_win)
    np.save(output_dir / "test_windows.npy", test_win)

    splits_summary = {
        "total_windows": len(df_features),
        "train_windows": len(train_df),
        "val_windows": len(val_df),
        "test_windows": len(test_df),
        "train_groups": sorted(list(train_df["recording_id"].unique())),
        "val_groups": sorted(list(val_df["recording_id"].unique())),
        "test_groups": sorted(list(test_df["recording_id"].unique())),
        "window_size": window_size,
        "sampling_rate_hz": sampling_rate_hz,
        "channel": channel,
    }

    with open(output_dir / "splits_summary.json", "w", encoding="utf-8") as f:
        json.dump(splits_summary, f, indent=2)

    logger.info(f"Dataset preparation complete! Processed files saved to {output_dir}")
    logger.info(f"Train groups: {splits_summary['train_groups']}")
    logger.info(f"Val groups: {splits_summary['val_groups']}")
    logger.info(f"Test groups: {splits_summary['test_groups']}")


if __name__ == "__main__":
    raw_path = root_dir / "data" / "raw" / "gearbox_extracted"
    out_path = root_dir / "data" / "processed"
    prepare_dataset(raw_path, out_path)

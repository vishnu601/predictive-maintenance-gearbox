"""
Unit tests for data leakage prevention and group-aware dataset splitting.
"""

from pathlib import Path
import sys
import numpy as np
import pytest

root_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir / "src"))

from predictive_maintenance.data.splitting import group_train_val_test_split, verify_no_group_leakage


def test_no_leakage_across_recording_groups():
    # 10 recording runs with 50 windows each
    groups = []
    for g_id in range(10):
        groups.extend([f"recording_run_{g_id}"] * 50)

    train_idx, val_idx, test_idx = group_train_val_test_split(
        groups=groups,
        test_size=0.20,
        val_size=0.20,
        random_state=42,
    )

    train_groups = set([groups[i] for i in train_idx])
    val_groups = set([groups[i] for i in val_idx])
    test_groups = set([groups[i] for i in test_idx])

    # 1. Zero overlap
    assert len(train_groups.intersection(test_groups)) == 0
    assert len(train_groups.intersection(val_groups)) == 0
    assert len(val_groups.intersection(test_groups)) == 0

    # 2. Complete partitioning
    all_assigned = train_groups.union(val_groups).union(test_groups)
    assert len(all_assigned) == 10


def test_verify_no_group_leakage_raises_on_overlap():
    train_g = ["run_01", "run_02", "run_03"]
    val_g = ["run_04"]
    test_g = ["run_03", "run_05"]  # 'run_03' is leaked!

    with pytest.raises(ValueError, match="DATA LEAKAGE DETECTED"):
        verify_no_group_leakage(train_g, val_g, test_g)

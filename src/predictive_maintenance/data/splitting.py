"""
Group-aware dataset splitting to strictly prevent data leakage.
Ensures that all windows derived from a single recording run reside in either
train, validation, or test sets—never shared across splits.
"""

from typing import Dict, List, Optional, Set, Tuple
import numpy as np
from sklearn.model_selection import GroupShuffleSplit


def verify_no_group_leakage(
    train_groups: List[str],
    val_groups: Optional[List[str]],
    test_groups: List[str]
) -> bool:
    """
    Validates that group sets (recording IDs) are completely disjoint.
    Raises ValueError if any group ID appears in more than one partition.
    """
    set_train = set(train_groups)
    set_test = set(test_groups)

    train_test_overlap = set_train.intersection(set_test)
    if train_test_overlap:
        raise ValueError(
            f"DATA LEAKAGE DETECTED! Group IDs present in both train and test splits: {train_test_overlap}"
        )

    if val_groups is not None:
        set_val = set(val_groups)
        train_val_overlap = set_train.intersection(set_val)
        val_test_overlap = set_val.intersection(set_test)
        if train_val_overlap:
            raise ValueError(
                f"DATA LEAKAGE DETECTED! Group IDs present in both train and val splits: {train_val_overlap}"
            )
        if val_test_overlap:
            raise ValueError(
                f"DATA LEAKAGE DETECTED! Group IDs present in both val and test splits: {val_test_overlap}"
            )

    return True


def group_train_val_test_split(
    groups: List[str],
    test_size: float = 0.20,
    val_size: float = 0.15,
    random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Splits sample indices into Train, Validation, and Test sets using GroupShuffleSplit.
    
    Args:
        groups: List or array of recording group identifiers corresponding to each sample.
        test_size: Fraction of groups reserved for held-out testing.
        val_size: Fraction of remaining training groups reserved for validation.
        random_state: Seed for reproducibility.

    Returns:
        (train_indices, val_indices, test_indices) as numpy integer arrays.
    """
    unique_groups = np.unique(groups)
    n_unique = len(unique_groups)

    if n_unique < 3:
        raise ValueError(
            f"At least 3 distinct recording groups are required to split into Train, Val, and Test without leakage. "
            f"Found only {n_unique} unique groups: {list(unique_groups)}."
        )

    # First split: Hold out test groups
    gss_test = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    all_indices = np.arange(len(groups))
    train_val_idx, test_idx = next(gss_test.split(all_indices, groups=groups))

    # Second split: Train vs Validation from the remaining data
    train_val_groups = [groups[i] for i in train_val_idx]
    # Adjusted val size relative to the remaining train_val subset
    adjusted_val_size = val_size / (1.0 - test_size)
    gss_val = GroupShuffleSplit(n_splits=1, test_size=adjusted_val_size, random_state=random_state)
    train_sub_idx, val_sub_idx = next(gss_val.split(train_val_idx, groups=train_val_groups))

    train_idx = train_val_idx[train_sub_idx]
    val_idx = train_val_idx[val_sub_idx]

    # Verification check
    train_g = [groups[i] for i in train_idx]
    val_g = [groups[i] for i in val_idx]
    test_g = [groups[i] for i in test_idx]

    verify_no_group_leakage(train_g, val_g, test_g)

    return train_idx, val_idx, test_idx

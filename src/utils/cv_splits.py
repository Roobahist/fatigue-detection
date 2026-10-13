"""Data-independent integrity checks for grouped cross-validation splits."""

from __future__ import annotations

from collections.abc import Callable, Hashable, Sequence
from typing import TypeVar

Item = TypeVar("Item", bound=Hashable)


def validate_group_kfold_input(
    data_ids: Sequence[Item],
    n_splits: int,
    group_of: Callable[[Item], Hashable],
) -> None:
    """Reject inputs that cannot produce nonempty, participant-disjoint folds."""
    if n_splits < 2:
        raise ValueError("Cross-validation requires at least two folds")
    if not data_ids:
        raise ValueError("Cross-validation requires at least one sample")
    if len(set(data_ids)) != len(data_ids):
        raise ValueError("Cross-validation sample IDs must be unique")

    n_groups = len({group_of(sample) for sample in data_ids})
    if n_splits > n_groups:
        raise ValueError(
            f"Cannot create {n_splits} folds from {n_groups} distinct groups"
        )


def validate_group_partition(
    data_ids: Sequence[Item],
    train_ids: Sequence[Item],
    heldout_ids: Sequence[Item],
    group_of: Callable[[Item], Hashable],
) -> None:
    """Check complete sample coverage and group isolation after a split.

    The grouping callback is injected so this check also works for datasets
    whose participant identifier is not encoded in a filename.
    """
    if not train_ids or not heldout_ids:
        raise ValueError("Both training and held-out partitions must be nonempty")

    original = set(data_ids)
    training = set(train_ids)
    heldout = set(heldout_ids)

    if len(original) != len(data_ids):
        raise ValueError("Cross-validation sample IDs must be unique")
    if len(training) != len(train_ids) or len(heldout) != len(heldout_ids):
        raise ValueError("A partition contains duplicate sample IDs")
    if training & heldout:
        raise ValueError("Training and held-out partitions share samples")
    if training | heldout != original:
        raise ValueError("Partitions must cover the input samples exactly")

    training_groups = {group_of(sample) for sample in train_ids}
    heldout_groups = {group_of(sample) for sample in heldout_ids}
    overlapping_groups = training_groups & heldout_groups
    if overlapping_groups:
        raise ValueError(
            "Training and held-out partitions share participant groups: "
            f"{len(overlapping_groups)} overlapping group(s)"
        )

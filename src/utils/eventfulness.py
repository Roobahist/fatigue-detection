from typing import List, Tuple, Optional, Union

import numpy as np
import pandas as pd

from Zh.src.utils.memmap_handler import MemmapHandler


def rank_eventfulness(
    eventfulness_scores_path: str,
    stride: int,
    window_size: int,
    aggregate_by: str = "mean",
) -> List[Tuple[int, int, float]]:
    """
    Ranks segments of eventfulness scores.

    Args:
        eventfulness_scores_path (str): Path to the eventfulness scores .dat file.
        stride (int): Stride for segmenting the scores.
        window_size (int): Window size for each segment.
        aggregate_by (str): Method for aggregating eventfulness scores within a segment.

    Returns:
        List[Tuple[int, float]]: List of tuples containing segment start index and its average eventfulness score, sorted by score in descending order.
    """
    mh = MemmapHandler(path=eventfulness_scores_path, mode="r")
    num_frames = mh.shape[0]
    segments = []

    if aggregate_by == "mean":
        aggregate_func = np.mean
    elif aggregate_by == "max":
        aggregate_func = np.max
    else:
        raise ValueError(
            f"Unsupported aggregation method: {aggregate_by}. Supported methods: mean, max"
        )

    for start in range(0, num_frames - window_size + 1, stride):
        end = start + window_size
        segment_scores = mh.read(slice=(slice(start, end), slice(0, 1)))
        avg_score = aggregate_func(segment_scores)
        segments.append((start, end, avg_score))

    mh.close()
    segments.sort(key=lambda x: x[2], reverse=True)
    return segments


def pick_eventful_segments(
    eventfulness_scores_path: str,
    window_size: int,
    stride: Optional[Union[int, None]] = None,
    pick_mean_top: int = 0,
    pick_max_top: int = 0,
    pick_random: int = 0,
    rng: Optional[np.random.Generator] = None,
):
    if rng is None:
        rng = np.random.default_rng()

    stride = stride or window_size
    num_frames = MemmapHandler(eventfulness_scores_path).shape[0]
    num_windows = (num_frames - window_size) // stride

    if not (isinstance(pick_mean_top, int) or isinstance(pick_mean_top, np.integer)) or pick_mean_top < 0:
        raise ValueError(
            f"pick_mean_top must be a non-negative integer. Please provide a valid value, {pick_mean_top} of type {type(pick_mean_top)} given."
        )
    if not (isinstance(pick_max_top, int) or isinstance(pick_max_top, np.integer)) or pick_max_top < 0:
        raise ValueError(
            f"pick_max_top must be a non-negative integer. Please provide a valid value, {pick_max_top} of type {type(pick_max_top)} given."
        )
    if not (isinstance(pick_random, int) or isinstance(pick_random, np.integer)) or pick_random < 0:
        raise ValueError(
            f"pick_random must be a non-negative integer. Please provide a valid value, {pick_random} of type {type(pick_random)} given."
        )
    if pick_mean_top + pick_max_top + pick_random > num_windows:
        raise ValueError(
            "Total number of picks exceeds the number of windows. Please reduce the number of picks."
        )

    eventfulness_mean = pd.DataFrame(
        rank_eventfulness(
            eventfulness_scores_path=eventfulness_scores_path,
            stride=stride,
            window_size=window_size,
            aggregate_by="mean",
        ),
        columns=["start_frame", "end_frame", "mean_eventfulness"],
    )

    eventfulness_max = pd.DataFrame(
        rank_eventfulness(
            eventfulness_scores_path=eventfulness_scores_path,
            stride=stride,
            window_size=window_size,
            aggregate_by="max",
        ),
        columns=["start_frame", "end_frame", "max_eventfulness"],
    )

    eventfulness = (
        eventfulness_mean.merge(eventfulness_max, on=["start_frame", "end_frame"])
        .sort_values("start_frame")
        .reset_index(drop=True)
    )

    if len(eventfulness) < pick_mean_top + pick_max_top + pick_random:
        print(
            f"Warning: Not enough windows ({len(eventfulness)}) to satisfy the pick requirements (mean: {pick_mean_top}, max: {pick_max_top}, random: {pick_random})."
        )
        pick_mean_top = min(pick_mean_top, len(eventfulness))
        pick_max_top = min(pick_max_top, len(eventfulness) - pick_mean_top)
        pick_random = min(pick_random, len(eventfulness) - pick_mean_top - pick_max_top)

    picked_indices = []

    picked_indices += (
        eventfulness.drop(index=picked_indices)
        .sort_values("mean_eventfulness", ascending=False)
        .iloc[:pick_mean_top]
        .index.to_list()
    )
    picked_indices += (
        eventfulness.drop(index=picked_indices)
        .sort_values("max_eventfulness", ascending=False)
        .iloc[:pick_max_top]
        .index.to_list()
    )
    if pick_random > 0:
        remaining = eventfulness.drop(index=picked_indices)

        if len(remaining) == 0:
            raise RuntimeError("No remaining windows to sample from.")

        weights = remaining["mean_eventfulness"].to_numpy(dtype=np.float64)
        weights = weights / weights.sum()

        sampled_pos = rng.choice(
            len(remaining),
            size=pick_random,
            replace=False,
            p=weights,
        )

        picked_indices += remaining.index.to_numpy()[sampled_pos].tolist()

    return eventfulness, picked_indices

from typing import List, Tuple
from Zh.src.utils.memmap_handler import MemmapHandler
import numpy as np

def rank_eventfulness(evenfulness_scores_path : str, stride : int, window_size : int) -> List[Tuple[int, int, float]]:
    """
    Ranks segments of eventfulness scores.

    Args:
        evenfulness_scores_path (str): Path to the eventfulness scores .dat file.
        stride (int): Stride for segmenting the scores.
        window_size (int): Window size for each segment.

    Returns:
        List[Tuple[int, float]]: List of tuples containing segment start index and its average eventfulness score, sorted by score in descending order.
    """
    mh = MemmapHandler(path=evenfulness_scores_path, mode="r")
    num_frames = mh.shape[0]
    segments = []

    for start in range(0, num_frames - window_size + 1, stride):
        end = start + window_size
        segment_scores = mh.read(slice=(slice(start, end), slice(0, 1)))
        avg_score = np.mean(segment_scores)
        segments.append((start, end, avg_score))

    mh.close()
    segments.sort(key=lambda x: x[2], reverse=True)
    return segments

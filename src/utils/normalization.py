from dataclasses import dataclass
from typing import Iterable, Tuple

import numpy as np


@dataclass
class FileStats:
    n: int
    mean: np.ndarray  # shape (d,)
    M2: np.ndarray  # shape (d,)


def compute_file_stats(iterator_over_batches) -> FileStats:
    """
    iterator_over_batches yields arrays of shape (B, d) for one file.
    Uses Welford's algorithm in a streaming way.
    """
    n = 0
    mean = None
    M2 = None

    for X in iterator_over_batches:  # X: (B, d)
        X = np.asarray(X)
        if mean is None:
            # lazy init to get dimension
            d = X.shape[1]
            mean = np.zeros(d, dtype=np.float64)
            M2 = np.zeros(d, dtype=np.float64)

        for x in X:  # x: (d,)
            n += 1
            delta = x - mean
            mean += delta / n
            delta2 = x - mean
            M2 += delta * delta2

    return FileStats(n=n, mean=mean, M2=M2)


def merge_two_stats(a: FileStats, b: FileStats) -> FileStats:
    if a.n == 0:
        return b
    if b.n == 0:
        return a

    n_a, mean_a, M2_a = a.n, a.mean, a.M2
    n_b, mean_b, M2_b = b.n, b.mean, b.M2

    n = n_a + n_b
    delta = mean_b - mean_a  # shape (d,)
    mean = mean_a + delta * (n_b / n)
    M2 = M2_a + M2_b + delta * delta * (n_a * n_b / n)

    return FileStats(n=n, mean=mean, M2=M2)


def compute_file_stats_from_batches(
    batches: Iterable[np.ndarray],
) -> FileStats:
    n_total = 0
    mean_total = None
    M2_total = None

    for batch in batches:
        batch = np.asarray(batch, dtype=np.float64)
        if batch.ndim == 1:
            batch = batch.reshape(1, -1)

        if batch.size == 0:
            continue

        n_b = batch.shape[0]
        mean_b = batch.mean(axis=0)  # (d,)
        diff = batch - mean_b  # (B, d)
        M2_b = (diff * diff).sum(axis=0)  # (d,)

        fs_b = FileStats(n=n_b, mean=mean_b, M2=M2_b)

        if mean_total is None:
            n_total = fs_b.n
            mean_total = fs_b.mean
            M2_total = fs_b.M2
        else:
            merged = merge_two_stats(FileStats(n_total, mean_total, M2_total), fs_b)
            n_total, mean_total, M2_total = merged.n, merged.mean, merged.M2

    if mean_total is None:
        return FileStats(
            n=0, mean=np.array([], dtype=np.float64), M2=np.array([], dtype=np.float64)
        )

    return FileStats(n=n_total, mean=mean_total, M2=M2_total)


def aggregate_stats_for_files(file_stats: Iterable[FileStats]) -> FileStats:
    """
    Combine stats for all training files in a split.
    """
    agg = FileStats(n=0, mean=None, M2=None)
    for fs in file_stats:
        if agg.mean is None:
            # initialize agg with fs, but copy to avoid aliasing
            agg = FileStats(
                n=fs.n,
                mean=fs.mean.astype(np.float64).copy(),
                M2=fs.M2.astype(np.float64).copy(),
            )
        else:
            agg = merge_two_stats(agg, fs)
    return agg


def stats_to_mean_std(
    stats: FileStats, eps: float = 1e-8
) -> Tuple[np.ndarray, np.ndarray]:
    if stats.n < 2:
        raise ValueError("Need at least 2 samples to compute variance.")
    mean = stats.mean
    var = stats.M2 / (stats.n - 1)
    std = np.sqrt(var + eps)
    return mean, std

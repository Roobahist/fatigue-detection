"""Portable dataset path resolution and label parsing for research experiments."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Mapping, Sequence

# Legacy HPC fallback is retained for existing experiment configurations.
DEFAULT_DATA_ROOT = Path("/work/long_lab/Mojtaba/data")

_DATA_LAYOUTS: dict[str, dict[str, str | None]] = {
    "UTA": {
        "embeddings_dir": "UTA/3.1-embedding/dinov2-base/cls/ROI0",
        "features_dir": "UTA/3-augmented_roi/masked_aligned_cropped0.05/features",
        "eventfulness_scores_dir": "UTA/3-augmented_roi/masked_aligned_cropped0.05/eventfulness",
        "normalization_info_path": "UTA/3.2.1-subjectwise_normalization_info_aggregated/normalization_info.csv",
        "kss_score_path": None,
    },
    "DROZY": {
        "embeddings_dir": "DROZY/data/videos/3.1-embedding/dinov2-base/cls/ROI0",
        "features_dir": "DROZY/data/videos/3-augmented_roi/masked_aligned_cropped0.05/features",
        "eventfulness_scores_dir": "DROZY/data/videos/3-augmented_roi/masked_aligned_cropped0.05/eventfulness",
        "normalization_info_path": "DROZY/data/videos/3.2.1-subjectwise_normalization_info_aggregated/normalization_info.csv",
        "kss_score_path": "DROZY/data/original/KSS.txt",
    },
}


def resolve_data_paths(
    dataset_name: str,
    data_root: str | Path,
    overrides: Mapping[str, str] | None = None,
) -> dict[str, str | None]:
    """Construct dataset paths from a caller-supplied root and overrides."""
    try:
        layout = _DATA_LAYOUTS[dataset_name]
    except KeyError as exc:
        raise ValueError(f"Unsupported dataset: {dataset_name!r}") from exc
    root = Path(data_root).expanduser()
    resolved = {
        name: str(root / relative) if relative is not None else None
        for name, relative in layout.items()
    }
    for name, path in (overrides or {}).items():
        if name not in layout:
            raise ValueError(f"Unknown dataset path override: {name!r}")
        if not str(path).strip():
            raise ValueError(f"Dataset path override {name!r} cannot be empty")
        resolved[name] = str(Path(path).expanduser())
    return resolved


@lru_cache(maxsize=16)
def load_kss_scores(path: str) -> tuple[tuple[int, ...], ...]:
    """Read the DROZY KSS matrix once per file path and worker process."""
    with open(path, encoding="utf-8") as handle:
        rows = [
            tuple(int(value) for value in line.split())
            for line in handle
            if line.strip()
        ]
    if not rows or any(not row for row in rows):
        raise ValueError(f"KSS score table is empty or malformed: {path}")
    return tuple(rows)


def extract_target(
    video_id: str,
    dataset_name: str,
    *,
    kss_scores: Sequence[Sequence[int]] | None = None,
) -> float:
    """Extract original UTA or DROZY video labels without applying label mapping."""
    basename = Path(video_id).name
    if dataset_name == "UTA":
        try:
            return float(basename.split("_")[2])
        except (IndexError, ValueError) as exc:
            raise ValueError(f"Invalid UTA video ID: {basename!r}") from exc

    if dataset_name != "DROZY":
        raise ValueError(f"Unsupported dataset: {dataset_name!r}")
    if kss_scores is None:
        raise ValueError("DROZY label extraction requires a KSS score matrix")
    try:
        subject_text, session_text = basename.split("_")[0].split("-")
        subject, session = int(subject_text), int(session_text)
        if subject <= 0 or session <= 0:
            raise ValueError("Subject and session must be positive")
        return float(kss_scores[subject - 1][session - 1])
    except (IndexError, ValueError) as exc:
        raise ValueError(f"Invalid DROZY ID or missing KSS score: {basename!r}") from exc

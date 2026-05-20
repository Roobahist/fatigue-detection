#!/usr/bin/env python3
"""
Compare row alignment across:
  1) source video (total frames)
  2) embeddings memmap rows
  3) features memmap rows

Assumptions based on your naming:
  - video:    <id>__masked_aligned_cropped0.05__ROI0__base.mp4
  - features: <id>__features.dat
  - embed:    <id>__masked_aligned_cropped0.05__ROI0__base__embedding_*.dat

Outputs:
  - prints a readable summary
  - optionally writes a CSV report

Usage example:
  python -m Zh.src.preprocess.check_alignment --videos_dir /work/long_lab/Mojtaba/data/DROZY/data/videos/3-augmented_roi/masked_aligned_cropped0.05 --features_dir /work/long_lab/Mojtaba/data/DROZY/data/videos/3-augmented_roi/masked_aligned_cropped0.05/features --embeddings_dir /work/long_lab/Mojtaba/data/DROZY/data/videos/3.1-embedding --embedding_glob "*__embedding_dinov2-base_strategy_cls.dat" --out_csv /work/long_lab/Mojtaba/data/DROZY/data/videos/alignment_report.csv
  python -m Zh.src.preprocess.check_alignment --videos_dir /work/long_lab/Mojtaba/data/UTA/3-augmented_roi/masked_aligned_cropped0.05 --features_dir /work/long_lab/Mojtaba/data/UTA/3-augmented_roi/masked_aligned_cropped0.05/features --embeddings_dir /work/long_lab/Mojtaba/data/UTA/3.1-embedding --embedding_glob "*__embedding_dinov2-base_strategy_cls.dat" --out_csv /work/long_lab/Mojtaba/data/UTA/alignment_report.csv
"""

from __future__ import annotations

import argparse
import csv
import glob
import os
from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence, Tuple

from Zh.src.utils.memmap_handler import MemmapHandler
from Zh.src.utils.video import VideoReader


@dataclass(frozen=True)
class Paths:
    video_path: str
    features_path: str
    embedding_path: str


@dataclass(frozen=True)
class Counts:
    video_frames: Optional[int]
    features_rows: Optional[int]
    embedding_rows: Optional[int]


@dataclass(frozen=True)
class Row:
    key: str
    video_path: str
    features_path: str
    embedding_path: str
    video_frames: Optional[int]
    features_rows: Optional[int]
    embedding_rows: Optional[int]
    ok_all_equal: bool
    mismatch_video_vs_features: bool
    mismatch_video_vs_embedding: bool
    mismatch_features_vs_embedding: bool
    missing_video: bool
    missing_features: bool
    missing_embedding: bool
    error: str


def default_logger(msg: str) -> None:
    print(msg)


def list_videos(videos_dir: str, video_glob: str) -> List[str]:
    # Recursive search
    pattern = os.path.join(videos_dir, "**", video_glob)
    paths = sorted(glob.glob(pattern, recursive=True))
    return paths


def video_key_from_path(video_path: str) -> str:
    """
    From:
      1_01_0_15fps__masked_aligned_cropped0.05__ROI0__base.mp4
    Return:
      1_01_0_15fps
    """
    base = os.path.basename(video_path)
    stem, _ext = os.path.splitext(base)
    return stem.split("__")[0]


def build_features_path(features_dir: str, key: str) -> str:
    return os.path.join(features_dir, f"{key}__features.dat")


def find_embedding_path(
    embeddings_dir: str,
    key: str,
    embedding_glob: str,
) -> Optional[str]:
    """
    We search for embeddings using a glob pattern, because embedding file names may include model/strategy.
    Example pattern:
      "*__embedding_dinov2-base_strategy_cls.dat"
    We constrain by key and "__embedding" presence.
    """
    # Recursive search
    pattern = os.path.join(embeddings_dir, "**", f"{key}__*{embedding_glob.lstrip('*')}")
    matches = sorted(glob.glob(pattern, recursive=True))
    if len(matches) == 0:
        return None
    if len(matches) == 1:
        return matches[0]
    # If multiple matches, pick the shortest path (often most specific), but record is still useful.
    return min(matches, key=len)


def safe_video_frames(video_path: str) -> Tuple[Optional[int], str]:
    try:
        if not os.path.exists(video_path):
            return None, "missing_video"
        vr = VideoReader(video_path)
        return int(vr.total_frames), ""
    except Exception as e:
        return None, f"video_error: {type(e).__name__}: {e}"


def safe_memmap_rows(memmap_path: Optional[str]) -> Tuple[Optional[int], str, bool]:
    """
    Returns: (rows, error_str, missing)
    """
    if memmap_path is None:
        return None, "missing_path", True
    if not os.path.exists(memmap_path):
        return None, "missing_file", True
    try:
        mh = MemmapHandler(memmap_path, mode="r")
        shape = mh.shape
        if shape is None or len(shape) == 0:
            return None, "bad_shape", False
        return int(shape[0]), "", False
    except Exception as e:
        return None, f"memmap_error: {type(e).__name__}: {e}", False


def compare_counts(video_frames: Optional[int], features_rows: Optional[int], embedding_rows: Optional[int]) -> Tuple[bool, bool, bool, bool]:
    """
    Returns:
      ok_all_equal
      mismatch_video_vs_features
      mismatch_video_vs_embedding
      mismatch_features_vs_embedding
    Missing values do not count as mismatch; they simply prevent equality checks.
    """
    def neq(a: Optional[int], b: Optional[int]) -> bool:
        if a is None or b is None:
            return False
        return a != b

    mvf = neq(video_frames, features_rows)
    mve = neq(video_frames, embedding_rows)
    mfe = neq(features_rows, embedding_rows)

    ok = True
    if (video_frames is not None) and (features_rows is not None) and (embedding_rows is not None):
        ok = (video_frames == features_rows == embedding_rows)
    else:
        ok = False

    return ok, mvf, mve, mfe


def make_paths_for_video(
    video_path: str,
    features_dir: str,
    embeddings_dir: str,
    embedding_glob: str,
) -> Paths:
    key = video_key_from_path(video_path)
    features_path = build_features_path(features_dir, key)
    embedding_path = find_embedding_path(embeddings_dir, key, embedding_glob) or ""
    return Paths(video_path=video_path, features_path=features_path, embedding_path=embedding_path)


def analyze_one(
    video_path: str,
    features_dir: str,
    embeddings_dir: str,
    embedding_glob: str,
) -> Row:
    key = video_key_from_path(video_path)

    features_path = build_features_path(features_dir, key)
    embedding_path_opt = find_embedding_path(embeddings_dir, key, embedding_glob)
    embedding_path = embedding_path_opt if embedding_path_opt is not None else ""

    vf, vf_err = safe_video_frames(video_path)
    fr, fr_err, fr_missing = safe_memmap_rows(features_path)
    er, er_err, er_missing = safe_memmap_rows(embedding_path_opt)

    error_parts: List[str] = []
    if vf_err:
        error_parts.append(vf_err)
    if fr_err and not fr_missing:
        error_parts.append(f"features_{fr_err}")
    if er_err and not er_missing:
        error_parts.append(f"embedding_{er_err}")
    if embedding_path_opt is None:
        error_parts.append("missing_embedding_match")

    ok, mvf, mve, mfe = compare_counts(vf, fr, er)

    return Row(
        key=key,
        video_path=video_path,
        features_path=features_path,
        embedding_path=embedding_path if embedding_path else "",
        video_frames=vf,
        features_rows=fr,
        embedding_rows=er,
        ok_all_equal=ok,
        mismatch_video_vs_features=mvf,
        mismatch_video_vs_embedding=mve,
        mismatch_features_vs_embedding=mfe,
        missing_video=not os.path.exists(video_path),
        missing_features=not os.path.exists(features_path),
        missing_embedding=(embedding_path_opt is None) or (not os.path.exists(embedding_path_opt)),
        error="; ".join(error_parts),
    )


def write_csv(rows: Sequence[Row], out_csv: str) -> None:
    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    fieldnames = [
        "key",
        "video_path",
        "features_path",
        "embedding_path",
        "video_frames",
        "features_rows",
        "embedding_rows",
        "ok_all_equal",
        "mismatch_video_vs_features",
        "mismatch_video_vs_embedding",
        "mismatch_features_vs_embedding",
        "missing_video",
        "missing_features",
        "missing_embedding",
        "error",
    ]
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: getattr(r, k) for k in fieldnames})


def print_summary(rows: Sequence[Row], logger: Callable[[str], None]) -> None:
    total = len(rows)
    ok = sum(1 for r in rows if r.ok_all_equal)
    missing_any = sum(1 for r in rows if (r.missing_video or r.missing_features or r.missing_embedding))
    mismatched_any = sum(1 for r in rows if (r.mismatch_video_vs_features or r.mismatch_video_vs_embedding or r.mismatch_features_vs_embedding))

    logger("")
    logger("Alignment summary")
    logger(f"  total samples: {total}")
    logger(f"  all equal (video == features == embeddings): {ok}")
    logger(f"  missing any file: {missing_any}")
    logger(f"  any mismatch among available counts: {mismatched_any}")
    logger("")

    # Print a compact list of problematic ones
    bad = [r for r in rows if (not r.ok_all_equal) or r.error or r.missing_video or r.missing_features or r.missing_embedding]
    if not bad:
        logger("No issues detected.")
        return

    logger("Problems (first 50)")
    for r in bad[:50]:
        logger(
            "  "
            + f"key={r.key} "
            + f"video={r.video_frames} "
            + f"features={r.features_rows} "
            + f"embed={r.embedding_rows} "
            + f"missing(v,f,e)=({r.missing_video},{r.missing_features},{r.missing_embedding}) "
            + (f"err={r.error}" if r.error else "")
        )
    if len(bad) > 50:
        logger(f"  ... and {len(bad) - 50} more")


def run_alignment_check(
    *,
    videos_dir: str,
    features_dir: str,
    embeddings_dir: str,
    video_glob: str,
    embedding_glob: str,
    out_csv: Optional[str],
    logger: Callable[[str], None],
) -> int:
    videos = list_videos(videos_dir, video_glob)
    if not videos:
        logger(f"No videos found in {videos_dir} with pattern {video_glob}")
        return 2

    rows: List[Row] = []
    for i, vp in enumerate(videos):
        r = analyze_one(
            video_path=vp,
            features_dir=features_dir,
            embeddings_dir=embeddings_dir,
            embedding_glob=embedding_glob,
        )
        rows.append(r)

        # periodic progress
        if (i + 1) % 200 == 0:
            logger(f"checked {i+1}/{len(videos)}")

    print_summary(rows, logger)

    if out_csv:
        write_csv(rows, out_csv)
        logger(f"Wrote CSV report: {out_csv}")

    # Exit code: 0 if everything aligned, else 1
    all_ok = all(r.ok_all_equal for r in rows)
    return 0 if all_ok else 1


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Check alignment between video frames, embeddings rows, and features rows.")
    p.add_argument("--videos_dir", required=True)
    p.add_argument("--features_dir", required=True)
    p.add_argument("--embeddings_dir", required=True)
    p.add_argument("--video_glob", default="*.mp4")
    p.add_argument(
        "--embedding_glob",
        default="*__embedding_dinov2-base_strategy_cls.dat",
        help="Glob suffix for embedding files. It is joined with key prefix.",
    )
    p.add_argument("--out_csv", default="")
    return p


def main() -> None:
    args = build_arg_parser().parse_args()

    out_csv = args.out_csv.strip() or None
    code = run_alignment_check(
        videos_dir=args.videos_dir,
        features_dir=args.features_dir,
        embeddings_dir=args.embeddings_dir,
        video_glob=args.video_glob,
        embedding_glob=args.embedding_glob,
        out_csv=out_csv,
        logger=default_logger,
    )
    raise SystemExit(code)


if __name__ == "__main__":
    main()
import logging
import math
import os
from datetime import datetime
from typing import List, Optional, Union

import numpy as np
from tqdm import tqdm

from Zh.src.process.embedding_models.embedders import *
from Zh.src.utils.common import update_file_name
from Zh.src.utils.file_folder import pick_selected_file_paths
from Zh.src.utils.memmap_handler import MemmapHandler
from Zh.src.utils.serialized_parallel import process
from Zh.src.utils.video import VideoReader

embedder = None


def _add_unfinished_suffix(path: str) -> str:
    root, ext = os.path.splitext(path)
    return f"{root}_unfinished{ext}"


def _resolve_log_dir(output_folder: str) -> str:
    output_dir = os.environ.get("OUTPUT_DIR", "").strip()
    if output_dir:
        return os.path.join(output_dir, "logs")
    return os.path.join(output_folder, "logs")


def _setup_logger(log_dir: str, name: str) -> logging.Logger:
    os.makedirs(log_dir, exist_ok=True)

    logger = logging.getLogger(f"{name}_{os.getpid()}")
    logger.setLevel(logging.INFO)
    logger.propagate = False

    if logger.handlers:
        return logger

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(log_dir, f"{name}_pid{os.getpid()}_{ts}.log")
    fmt = logging.Formatter("[%(asctime)s][%(levelname)s] %(message)s")

    fh = logging.FileHandler(log_path)
    fh.setLevel(logging.INFO)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setLevel(logging.INFO)
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    logger.info(f"log_path={log_path}")
    return logger


def _init_valid_column_to_unwritten(
    mh: MemmapHandler, valid_col: int, chunk_rows: int = 200_000
) -> None:
    n = int(mh.shape[0])
    start = 0
    while start < n:
        end = min(start + chunk_rows, n)
        block = np.full((end - start, 1), -1.0, dtype=np.float32)
        mh.write(
            data=block,
            start=start,
            slice=(slice(start, end), slice(valid_col, valid_col + 1)),
        )
        start = end


def _find_resume_row_by_valid(
    mh: MemmapHandler, valid_col: int, chunk_rows: int = 200_000
) -> int:
    """
    Returns first row index where valid == -1.0 (unwritten).
    If all rows are written (valid in {0,1}), returns mh.shape[0].
    """
    n = int(mh.shape[0])
    if n == 0:
        return 0
    start = 0
    while start < n:
        end = min(start + chunk_rows, n)
        arr = mh.read(slice=(slice(start, end), slice(valid_col, valid_col + 1)))
        if arr is None:
            return start
        v = np.asarray(arr).reshape(-1)
        idx = np.where(v == -1.0)[0]
        if idx.size > 0:
            return start + int(idx[0])
        start = end
    return n


def _validate_before_finalize(
    unfinished_path: str,
    expected_rows: int,
    expected_cols: int,
    valid_col: int,
    chunk_rows: int = 200_000,
) -> tuple[bool, str]:
    try:
        mh = MemmapHandler(path=unfinished_path, mode="r")
    except Exception as e:
        return False, f"open_failed {type(e).__name__}: {e}"

    if int(mh.shape[0]) != int(expected_rows):
        return (
            False,
            f"row_count_mismatch shape_rows={mh.shape[0]} expected_rows={expected_rows}",
        )

    if int(mh.shape[1]) != int(expected_cols):
        return (
            False,
            f"col_count_mismatch shape_cols={mh.shape[1]} expected_cols={expected_cols}",
        )

    n = int(expected_rows)
    start = 0
    while start < n:
        end = min(start + chunk_rows, n)
        arr = mh.read(slice=(slice(start, end), slice(valid_col, valid_col + 1)))
        if arr is None:
            return False, f"valid_read_failed range={start}:{end}"

        v = np.asarray(arr).reshape(-1)

        bad_val = np.where((v != -1.0) & (v != 0.0) & (v != 1.0))[0]
        if bad_val.size > 0:
            i = start + int(bad_val[0])
            return (
                False,
                f"valid_out_of_domain row={i} value={float(v[int(bad_val[0])])}",
            )

        unwritten = np.where(v == -1.0)[0]
        if unwritten.size > 0:
            i = start + int(unwritten[0])
            return False, f"unwritten_rows_remaining first_unwritten_row={i}"

        start = end

    return True, ""


def _touch_success_files(logger: logging.Logger) -> None:
    output_dir = os.environ.get("OUTPUT_DIR", "").strip()
    if not output_dir:
        return

    os.makedirs(output_dir, exist_ok=True)

    default_success = os.path.join(output_dir, "EXPERIMENT_COMPLETE")
    try:
        with open(default_success, "a"):
            os.utime(default_success, None)
        logger.info(f"success_file_touched path={default_success}")
    except Exception as e:
        logger.error(
            f"success_file_touch_failed path={default_success} err={type(e).__name__}: {e}"
        )

    env_success = os.environ.get("SUCCESS_FILE", "").strip()
    if env_success:
        env_path = os.path.join(output_dir, env_success)
        try:
            with open(env_path, "a"):
                os.utime(env_path, None)
            logger.info(f"success_file_touched path={env_path}")
        except Exception as e:
            logger.error(
                f"success_file_touch_failed path={env_path} err={type(e).__name__}: {e}"
            )


def get_embedder(embedding_model_name: str):
    global embedder
    if embedder is None:
        embedder = make_embedder(embedding_model_name)
    return embedder


def process_single_video(
    input_folder: str,
    output_folder: str,
    file_path: str,
    start_second: Optional[int] = 0,
    end_second: Optional[int] = None,
    embedding_model_name: Optional[str] = "Facenet",
    strategies: Union[str, List[str]] = "cls",
    batch_size: Optional[int] = 32,
    use_tqdm: Optional[bool] = False,
    file_index: Optional[int] = 0,
):
    logger = _setup_logger(
        _resolve_log_dir(output_folder=output_folder), name="embeddings"
    )

    logger.info(f"file_start file_index={file_index} file_path={file_path}")

    embedder = get_embedder(embedding_model_name)

    vr = VideoReader(file_path)
    info = vr.get_info()
    total_frames = info["total_frames"]
    batch_size = min(batch_size, total_frames) if total_frames > 0 else batch_size
    video_fps = info["fps"]

    start_frame = (
        int(start_second * video_fps) if isinstance(start_second, (int, float)) else 0
    )
    end_frame = (
        int(end_second * video_fps)
        if isinstance(end_second, (int, float))
        else total_frames
    )

    expected_rows = int(end_frame - start_frame)
    if expected_rows <= 0:
        logger.warning(
            f"file_skip_empty file_index={file_index} start_frame={start_frame} end_frame={end_frame}"
        )
        vr.release()
        return

    valid_strategies = [s for s in strategies if s in embedder.valid_strategies]

    writers = {}
    writer_paths = {}

    ret, temp_frame = vr.read_frame(start_frame)
    if not ret:
        logger.error(
            f"file_fail_read_start_frame file_index={file_index} start_frame={start_frame} file_path={file_path}"
        )
        vr.release()
        return

    for strategy in valid_strategies:
        base_dim = embedder.embed(frames=temp_frame, strategy=strategy).shape[0]
        dim_with_valid = int(base_dim) + 1
        shape = (expected_rows, dim_with_valid)
        valid_col = dim_with_valid - 1

        output_file_path = update_file_name(
            file_path=file_path,
            input_folder=input_folder,
            output_folder=output_folder,
            suffix=f"__embedding_{embedding_model_name.split('/')[-1]}_strategy_{strategy}",
            prefix=f"{embedding_model_name.split('/')[-1]}/{strategy}/",
            extension=".dat",
        )

        final_path = output_file_path
        unfinished_path = _add_unfinished_suffix(final_path)
        os.makedirs(os.path.dirname(final_path), exist_ok=True)

        if os.path.exists(final_path):
            logger.info(
                f"file_skip_finalized file_index={file_index} strategy={strategy} path={final_path}"
            )
            continue

        if os.path.exists(unfinished_path):
            writer = MemmapHandler(path=unfinished_path, mode="r+")
            if (
                int(writer.shape[0]) != expected_rows
                or int(writer.shape[1]) != dim_with_valid
            ):
                logger.warning(
                    f"file_warn_unfinished_shape_mismatch file_index={file_index} strategy={strategy} "
                    f"shape={writer.shape} expected={(expected_rows, dim_with_valid)} path={unfinished_path}"
                )
            writers[strategy] = writer
            writer_paths[strategy] = (
                unfinished_path,
                final_path,
                valid_col,
                expected_rows,
                dim_with_valid,
            )
        else:
            writer = MemmapHandler(
                path=unfinished_path, mode="w+", shape=shape, dtype="float32"
            )
            _init_valid_column_to_unwritten(writer, valid_col=valid_col)
            writers[strategy] = writer
            writer_paths[strategy] = (
                unfinished_path,
                final_path,
                valid_col,
                expected_rows,
                dim_with_valid,
            )
            logger.info(
                f"file_create_unfinished file_index={file_index} strategy={strategy} path={unfinished_path} shape={shape}"
            )

    if not writers:
        vr.release()
        logger.info(f"file_end_no_work file_index={file_index} file_path={file_path}")
        return

    any_strategy = next(iter(writers.keys()))
    any_writer = writers[any_strategy]
    valid_col_any = int(any_writer.shape[1]) - 1
    resume_row = _find_resume_row_by_valid(any_writer, valid_col=valid_col_any)
    start_iter = start_frame + int(resume_row)

    if resume_row > 0:
        logger.info(
            f"file_resume file_index={file_index} resume_row={resume_row} start_iter_frame={start_iter}"
        )

    read_failures = 0

    for frame_index in tqdm(
        range(start_iter, end_frame, batch_size),
        desc=f"Processing {file_index}: {file_path}",
        disable=not use_tqdm,
        unit="batch",
        total=(
            math.ceil((end_frame - start_iter) / batch_size)
            if end_frame > start_iter
            else 0
        ),
        miniters=5,
    ):
        ret, frames = vr.read_frames(
            frame_indices=range(frame_index, min(frame_index + batch_size, end_frame)),
            jump=False,
        )
        local_idx = frame_index - start_frame
        n = min(batch_size, end_frame - frame_index)

        if not ret:
            read_failures += 1
            for strategy, writer in writers.items():
                dim = int(writer.shape[1])
                out = np.zeros((n, dim), dtype=np.float32)
                out[:, -1] = 0.0
                writer.write(data=out, start=local_idx)
            continue

        for strategy, writer in writers.items():
            emb = embedder.embed_batch(frames=frames, strategy=strategy)
            emb = np.asarray(emb, dtype=np.float32)
            n_local = emb.shape[0]
            dim = int(writer.shape[1])

            out = np.zeros((n_local, dim), dtype=np.float32)
            out[:, : emb.shape[1]] = emb
            out[:, -1] = 1.0
            writer.write(data=out, start=local_idx)

    for _, writer in writers.items():
        writer.close()

    finalized_ok = True

    for strategy, (
        unfinished_path,
        final_path,
        valid_col,
        exp_rows,
        exp_cols,
    ) in writer_paths.items():
        ok, reason = _validate_before_finalize(
            unfinished_path=unfinished_path,
            expected_rows=exp_rows,
            expected_cols=exp_cols,
            valid_col=valid_col,
        )
        if ok:
            if os.path.exists(unfinished_path) and not os.path.exists(final_path):
                os.replace(unfinished_path, final_path)
            logger.info(f"file_finalize strategy={strategy} path={final_path}")
        else:
            finalized_ok = False
            logger.error(
                f"file_not_finalized strategy={strategy} path={unfinished_path} reason={reason}"
            )

    vr.release()

    if read_failures > 0:
        logger.warning(
            f"file_issues file_index={file_index} read_failures={read_failures}"
        )

    logger.info(
        f"file_end file_index={file_index} finalized_ok={finalized_ok} file_path={file_path}"
    )


def process_multiple_videos(
    input_folder: str,
    output_folder: str,
    relative_selected_paths: Optional[List[str]] = None,
    file_indices_spec: Optional[str] = "",
    start_second: Optional[int] = 0,
    end_second: Optional[int] = None,
    embedding_model_name: Optional[str] = "Facenet",
    strategies: Union[str, List[str]] = "cls",
    batch_size: Optional[int] = 32,
    use_tqdm: Optional[bool] = False,
    parallel: Optional[bool] = False,
    max_workers: Optional[int] = 10,
    maxtasksperchild: Optional[int] = 1,
):
    logger = _setup_logger(
        _resolve_log_dir(output_folder=output_folder), name="embeddings"
    )

    if relative_selected_paths is None:
        relative_selected_paths = []

    file_index_to_path = pick_selected_file_paths(
        input_folder=input_folder,
        depth=4,
        EXT_LIST=[".mp4"],
        relative_selected_paths=relative_selected_paths,
        file_indices_spec=file_indices_spec,
        sort_and_dedupe=True,
    )

    logger.info(f"batch_start n_files={len(file_index_to_path)}")

    args_list = [
        {
            "input_folder": input_folder,
            "output_folder": output_folder,
            "file_path": file_path,
            "start_second": start_second,
            "end_second": end_second,
            "embedding_model_name": embedding_model_name,
            "strategies": strategies,
            "batch_size": batch_size,
            "use_tqdm": use_tqdm,
            "file_index": file_index,
        }
        for file_index, file_path in file_index_to_path.items()
    ]

    process(
        worker_function=process_single_video,
        worker_params_list=args_list,
        handler_function=None,
        parallel=parallel,
        max_workers=max_workers,
        maxtasksperchild=maxtasksperchild,
        use_tqdm=use_tqdm,
        auto_slice=False,
    )

    # Completion check: for every selected file and every valid strategy, final file exists and unfinished does not.
    emb = get_embedder(embedding_model_name)
    valid_strategies = [s for s in strategies if s in emb.valid_strategies]

    incomplete = []
    for _, file_path in file_index_to_path.items():
        for strategy in valid_strategies:
            final_path = update_file_name(
                file_path=file_path,
                input_folder=input_folder,
                output_folder=output_folder,
                suffix=f"__embedding_{embedding_model_name.split('/')[-1]}_strategy_{strategy}",
                prefix=f"{embedding_model_name.split('/')[-1]}/{strategy}/",
                extension=".dat",
            )
            unfinished_path = _add_unfinished_suffix(final_path)

            if os.path.exists(unfinished_path) or not os.path.exists(final_path):
                incomplete.append((file_path, strategy))

    if len(incomplete) == 0:
        logger.info("batch_complete all_outputs_finalized=true")
        _touch_success_files(logger)
    else:
        logger.warning(
            f"batch_incomplete missing_or_unfinished_count={len(incomplete)} sample={incomplete[:5]}"
        )

    logger.info("batch_end")


if __name__ == "__main__":
    import argparse
    import logging

    logging.basicConfig(
        level=logging.INFO, format="[%(asctime)s][%(levelname)s] %(message)s"
    )

    parser = argparse.ArgumentParser(description="Find Video Embeddings")

    parser.add_argument(
        "input_folder",
        type=str,
        help="Path to the input folder containing video files.",
    )
    parser.add_argument(
        "output_folder",
        type=str,
        help="Path to the output folder to save processed videos.",
    )
    parser.add_argument(
        "relative_selected_paths",
        nargs="*",
        help="List of relative file path to process. If empty, all files in the input folder will be processed.",
    )
    parser.add_argument(
        "--file_indices_spec",
        type=str,
        default="",
        help="Comma-separated indices/ranges.",
    )
    parser.add_argument(
        "--start_second", type=int, default=0, help="Start processing from this second."
    )
    parser.add_argument(
        "--end_second", type=int, default=None, help="End processing at this second."
    )
    parser.add_argument(
        "--embedding_model_name",
        type=str,
        default="Facenet",
        help="Embedding model to use.",
    )
    parser.add_argument(
        "--strategies", type=str, nargs="*", default=["cls"], help="List of strategies."
    )
    parser.add_argument(
        "--batch_size", type=int, default=32, help="Batch size for processing frames."
    )
    parser.add_argument(
        "--parallel", action="store_true", help="Process videos in parallel."
    )
    parser.add_argument(
        "--max_workers", type=int, default=None, help="Maximum number of workers."
    )
    parser.add_argument(
        "--maxtasksperchild",
        type=int,
        default=None,
        help="Maximum tasks per child process.",
    )
    parser.add_argument(
        "--no_tqdm",
        action="store_false",
        dest="use_tqdm",
        default=True,
        help="Disable tqdm progress bars.",
    )

    args = parser.parse_args()

    process_multiple_videos(
        input_folder=args.input_folder,
        output_folder=args.output_folder,
        relative_selected_paths=args.relative_selected_paths,
        file_indices_spec=args.file_indices_spec,
        start_second=args.start_second,
        end_second=args.end_second,
        embedding_model_name=args.embedding_model_name,
        strategies=args.strategies,
        batch_size=args.batch_size,
        use_tqdm=args.use_tqdm,
        parallel=args.parallel,
        max_workers=args.max_workers,
        maxtasksperchild=args.maxtasksperchild,
    )

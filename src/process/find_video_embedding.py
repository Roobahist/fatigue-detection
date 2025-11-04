import logging
import math
import os
from datetime import datetime
from typing import List, Optional, Tuple, Union

from tqdm import tqdm

from Zh.src.process.embedding_models.embedders import *
from Zh.src.utils.common import update_file_name
from Zh.src.utils.file_folder import copy_directory_structure, pick_selected_file_paths
from Zh.src.utils.memmap_handler import MemmapHandler
from Zh.src.utils.serialized_parallel import process
from Zh.src.utils.video import VideoReader

embedder = None


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
    """
    Process a single video and write embeddings to HDF5 using multiple strategies.

    Args:
        input_folder (str): Path to the input folder containing video files.
        output_folder (str): Path to the output folder to save processed videos.
        file_path (str): Relative path of the video file to process.
        start_second (int, optional): Start processing from this second. Defaults to 0.
        end_second (int, optional): End processing at this second. Defaults to None.
        embedding_model_name (str, optional): Embedding model to use. Defaults to "Facenet".
        strategies (Union[str, List[str]]): List of strategies to use for embedding. Defaults to "cls".
        batch_size (int, optional): Number of frames to process in parallel. Defaults to 32.
        use_tqdm (bool, optional): Whether to use tqdm progress bars. Defaults to False.
        file_index (int, optional): Index of the file being processed. Defaults to 0
    """
    logging.info(f"Processing file {file_index}: {file_path}")

    embedder = get_embedder(embedding_model_name)

    vr = VideoReader(file_path)
    info = vr.get_info()
    total_frames = info["total_frames"]
    batch_size = min(batch_size, total_frames)
    video_fps = info["fps"]

    start_frame = int(start_second * video_fps)
    end_frame = int(end_second * video_fps) if end_second > 0 else total_frames

    valid_strategies = [s for s in strategies if s in embedder.valid_strategies]

    # calculating the embedding for the first frame to find the
    # dimension of the embeddings and instantiate the writers
    writers = {}
    ret, temp_frame = vr.read_frame(0)
    if ret:
        for strategy in valid_strategies:
            shape = (
                end_frame - start_frame,
                embedder.embed(frame=temp_frame, strategy=strategy).shape[0],
            )
            output_file_path = update_file_name(
                file_path=file_path,
                input_folder=input_folder,
                output_folder=output_folder,
                suffix=f"__embedding_{embedding_model_name.split('/')[-1]}_strategy_{strategy}",
                prefix=f"{embedding_model_name.split('/')[-1]}/{strategy}/",
                extension=".dat",
            )
            writer = MemmapHandler(
                path=output_file_path, mode="w+", shape=shape, dtype="float32"
            )
            writers[strategy] = writer
    else:
        logging.error(f"Failed to read frame at index 0, skipping {file_path}")
        return

    for frame_index in tqdm(
        range(start_frame, end_frame, batch_size),
        desc=f"Processing {file_index}: {file_path}",
        disable=not use_tqdm,
        unit="frame",
        total=math.ceil((end_frame - start_frame) / batch_size),
        miniters=5,
    ):
        ret, frames = vr.read_frames(
            frame_indices=range(frame_index, min(frame_index + batch_size, end_frame)),
            jump=False,
        )
        if not ret:
            print(f"{file_path} Failed to read frame at index {frame_index}, skipping.")
            continue

        for strategy, writer in writers.items():
            local_idx = frame_index - start_frame
            writer.write(
                data=embedder.embed_batch(frames=frames, strategy=strategy),
                start=local_idx,
            )

    for _, writer in writers.items():
        writer.close()


def process_multiple_videos(
    input_folder: str,
    output_folder: str,
    relative_selected_paths: Optional[List[str]] = [],
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
    """
    video_files : relative path after the input folder should be provided,
                       otherwise all the files in all subdirectories would be
                       processed.
    """

    selected_file_paths = pick_selected_file_paths(
        input_folder=input_folder,
        depth=4,
        EXT_LIST=[".mp4"],
        relative_selected_paths=relative_selected_paths,
    )

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
        for file_index, file_path in enumerate(selected_file_paths)
    ]

    process(
        worker_function=process_single_video,
        worker_params_list=args_list,
        handler_function=None,
        parallel=parallel,
        max_workers=max_workers,
        maxtasksperchild=maxtasksperchild,
        use_tqdm=use_tqdm,
        auto_slice=True,
    )


if __name__ == "__main__":
    import argparse
    import logging

    logging.basicConfig(
        level=logging.INFO,  # or DEBUG for more verbosity
        format="[%(asctime)s][%(levelname)s] %(message)s",
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
        "--start_second",
        type=int,
        default=0,
        help="Start processing from this second.",
    )
    parser.add_argument(
        "--end_second",
        type=int,
        default=None,
        help="End processing at this second.",
    )
    parser.add_argument(
        "--embedding_model_name",
        type=str,
        default="Facenet",
        help="Embedding model to use (default: Facenet)",
    )
    parser.add_argument(
        "--strategies",
        type=str,
        nargs="*",
        default=["cls"],
        help="List of strategies to use for embedding (default: ['cls'])",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for processing frames (default: 32)",
    )
    parser.add_argument(
        "--parallel",
        action="store_true",
        help="Process videos in parallel.",
    )
    parser.add_argument(
        "--max_workers",
        type=int,
        default=None,
        help="Maximum number of workers for parallel processing.",
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

    logging.info("Starting ...")

    process_multiple_videos(
        input_folder=args.input_folder,
        output_folder=args.output_folder,
        relative_selected_paths=args.relative_selected_paths,
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

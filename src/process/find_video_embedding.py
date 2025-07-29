import logging
import multiprocessing
import os
import sys
import traceback
from datetime import datetime
from typing import Callable, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import tensorflow as tf
import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoImageProcessor, AutoModel, pipeline

from Zh.src.process.frame_processor import FrameProcessor
from Zh.src.utils.csvwriter import CSVWriter
from Zh.src.utils.file_folder import copy_directory_structure, get_file_paths
from Zh.src.utils.hdf5writer import HDF5Writer
from Zh.src.utils.memory import cleanup
from Zh.src.utils.serialized_parallel import process
from Zh.src.utils.setup_device import setup_tensorflow, setup_torch
from Zh.src.utils.video import VideoReader, get_video_info, is_video_valid

logging.basicConfig(
    level=logging.INFO,  # or DEBUG for more verbosity
    format="[%(asctime)s][%(levelname)s] %(message)s",
)

setup_tensorflow()
device = setup_torch()

VIDEO_EXTS = [".mp4", ".mov", ".avi"]


# Overwrite placeholder metadata consistency checker
def check_metadata_consistency(existing_metadata, current_metadata):
    """
    Ensure all existing entries in the metadata are consistent with the current one.
    Raises ValueError if inconsistency is found.
    """
    for entry in existing_metadata:
        if (
            entry["video_path"] != current_metadata["video_path"]
            or entry["file_size"] != current_metadata["file_size"]
            or entry["model_name"] != current_metadata["model_name"]
            or entry["start_frame"] != current_metadata["start_frame"]
            or entry["end_frame"] != current_metadata["end_frame"]
            or entry.get("frame_stride") != current_metadata.get("frame_stride")
        ):
            raise ValueError(
                f"Inconsistent metadata detected:\nExisting: {entry}\nCurrent: {current_metadata}"
            )


HDF5Writer.check_metadata_consistency = staticmethod(check_metadata_consistency)

def process_single_video(
    video_path: str,
    model_name: str,
    model_bundle: Tuple,
    strategy_list: List[str],
    output_file: str,
    start_second: int = 0,
    end_second: int = -1,
    frame_stride: int = 1,
    **kwargs,
) -> None:
    """
    Process a single video and write embeddings to HDF5 using multiple strategies.

    Args:
        video_path (str): Path to the video file.
        model_name (str): Name of the model used for embedding.
        model_bundle (Tuple): The processor and model.
        strategy_list (List[str]): List of embedding strategies to compute.
        output_file (str): Path to the HDF5 output file.
        start_second (int): Starting second of the video to process.
        end_second (int): Ending second of the video to process.
        frame_stride (int): Frame interval to sample the video.
    """
    VR = VideoReader(video_path)
    info = VR.get_info()
    total_frames = info["total_frames"]
    video_fps = info["fps"]

    start_frame = int(start_second * video_fps)
    end_frame = int(end_second * video_fps) if end_second > 0 else total_frames
    expected_num_frames = (end_frame - start_frame) // frame_stride

    file_size = os.path.getsize(video_path)
    metadata = {
        "video_path": video_path,
        "file_size": str(file_size),
        "model_name": model_name,
        "start_frame": str(start_frame),
        "end_frame": str(end_frame),
        "frame_stride": str(frame_stride),
        "timestamp": datetime.now().isoformat(),
    }

    writer = HDF5Writer(output_file)
    existing_metadata = writer.read_all_metadata()
    writer.check_metadata_consistency(existing_metadata, metadata)

    strategy_done = {
        strategy: writer.get_dataset_length(strategy) >= expected_num_frames
        for strategy in strategy_list
    }

    for frame_number in tqdm(
        range(start_frame, end_frame, frame_stride), desc=os.path.basename(video_path)
    ):
        ret, frame = VR.jump_to_frame(frame_number)
        if not ret:
            break

        active_strategies = [s for s, done in strategy_done.items() if not done]
        if active_strategies:
            results = get_embedding(model_name, model_bundle, frame, active_strategies)
            writer.batch_write(results)

    lengths = [writer.get_dataset_length(s) for s in strategy_list]
    if len(set(lengths)) > 1:
        raise ValueError(
            f"Inconsistent dataset lengths in {output_file}: {dict(zip(strategy_list, lengths))}"
        )

    writer.add_metadata(metadata)
    writer.close()


def get_embedding_model(model_name: str = "google/vit-base-patch16-224-in21k") -> Tuple:
    """
    Load the specified embedding model and image processor.

    Args:
        model_name (str): The name of the model to load.

    Returns:
        Tuple: (processor, model) for use in embedding extraction.
    """
    # todo : make it compatible with Facenet and other models
    # FP = FrameProcessor()
    # frame_embedding = [
    #     round(x, 3)
    #     for x in FP.represent(
    #         img_path=frame,
    #         model_name=embedding_model,
    #         enforce_detection=False,
    #         detector_backend="skip",
    #         normalization=embedding_normalization,
    #     )[0]["embedding"]
    # ]

    if "/" in model_name:
        processor = AutoImageProcessor.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)
        model.eval()  # Set to eval mode to avoid dropout
        return processor, model
    else:
        raise ValueError(f"Unsupported model: {model_name}")


def get_embedding(
    model_name: str,
    model_bundle: Tuple,
    frame: np.ndarray,
    strategies: Union[str, List[str]] = "cls",
) -> Dict[str, np.ndarray]:
    """
    Get one or more embeddings for a single frame using specified strategies.

    Args:
        model_name (str): Model name for validation.
        model_bundle (Tuple): (processor, model) from get_embedding_model().
        frame (np.ndarray): The input frame in HWC RGB format.
        strategies (Union[str, List[str]]): Strategy or list of strategies.
            Options: "cls", "cls+mean", "pooled", "tokens", "tokens+cls"

    Returns:
        Dict[str, np.ndarray]: A dict mapping strategy names to their embeddings.
    """
    # todo : make it compatible with Facenet and other models

    if "/" not in model_name:
        raise ValueError(f"Unsupported model: {model_name}")

    if isinstance(strategies, str):
        strategies = [strategies]

    processor, model = model_bundle
    image = Image.fromarray(frame)
    inputs = processor(images=image, return_tensors="pt")

    result = {}

    with torch.no_grad():
        outputs = model(**inputs)
        last_hidden_state = outputs.last_hidden_state  # [1, num_tokens, hidden_dim]
        pooler_output = outputs.pooler_output  # [1, hidden_dim] or None

        cls_token = last_hidden_state[:, 0, :]  # [1, hidden_dim]
        patch_tokens = last_hidden_state[:, 1:, :]  # [1, 196, hidden_dim]
        mean_pooled = patch_tokens.mean(dim=1)  # [1, hidden_dim]
        cls_plus_mean = torch.cat([cls_token, mean_pooled], dim=1)  # [1, 2*hidden_dim]

        for strategy in strategies:
            if strategy == "cls":
                result["cls"] = cls_token.squeeze(0).cpu().numpy().astype(np.float32)

            elif strategy == "cls+mean":
                result["cls+mean"] = (
                    cls_plus_mean.squeeze(0).cpu().numpy().astype(np.float32)
                )

            elif strategy == "pooled":
                if pooler_output is not None:
                    result["pooled"] = (
                        pooler_output.squeeze(0).cpu().numpy().astype(np.float32)
                    )
                else:
                    raise ValueError(
                        "This model does not support pooled output (pooler_output is None)."
                    )

            elif strategy == "tokens":
                flat_tokens = (
                    patch_tokens.squeeze(0).flatten().cpu().numpy().astype(np.float32)
                )  # shape: [196 * hidden_dim]
                result["tokens"] = flat_tokens

            elif strategy == "tokens+cls":
                full_tokens = (
                    last_hidden_state.squeeze(0)
                    .flatten()
                    .cpu()
                    .numpy()
                    .astype(np.float32)
                )  # shape: [197 * hidden_dim]
                result["tokens+cls"] = full_tokens

            else:
                raise ValueError(f"Unsupported strategy: {strategy}")

    return result


def process_multiple_videos(
    input_folder_path: str,
    output_folder_path: str,
    video_file_names: Optional[List[str]] = [],
    start_second: Optional[int] = 0,
    end_second: Optional[int] = None,
    embedding_model_name: Optional[str] = "Facenet",
    strategies: Union[str, List[str]] = "cls",
    use_tqdm: Optional[bool] = False,
    parallel: Optional[bool] = False,
    max_workers: Optional[int] = 10,
    maxtasksperchild: Optional[int] = 1,
):
    """
    video_file_names : relative path after the input folder should be provided,
                       otherwise all the files in all subdirectories would be
                       processed.
    """
    copy_directory_structure(input_folder_path, output_folder_path)
    all_video_file_names = get_file_paths(
        input_path=input_folder_path, depth=5, EXT_LIST=VIDEO_EXTS
    )

    if video_file_names != []:
        video_file_paths = [
            video_file_name
            for video_file_name in all_video_file_names
            if os.path.basename(video_file_name) in video_file_names
        ]
        logging.info(
            f"Video File Paths Captured Using Provided List: {len(video_file_paths)}"
        )
    else:
        video_file_paths = all_video_file_names
        logging.info(
            f"Video File Paths Captured Automatically: {len(video_file_paths)}"
        )
    model_bundle = get_embedding_model(embedding_model_name)

    args_list = [
        {
            "input_folder_path": input_folder_path,
            "output_folder_path": output_folder_path,
            "video_file_path": video_file_paths[i],
            "start_second": start_second,
            "end_second": end_second,
            "model_bundle": model_bundle,
            "embedding_model_name": embedding_model_name,
            "strategies": strategies,
            "use_tqdm": use_tqdm,
            "file_index": i,
        }
        for i in range(len(video_file_paths))
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

    parser = argparse.ArgumentParser(description="Process video files")
    parser.add_argument(
        "input_folder_path",
        type=str,
        default="/work/long_lab/Mojtaba/data/UTA/1.4-augmentated_masked",
        help="Path to the input folder",
    )
    parser.add_argument(
        "output_folder_path",
        type=str,
        default="/work/long_lab/Mojtaba/data/UTA/1.4.1-augmentated_masked_embedded",
        help="Path to the output folder",
    )
    parser.add_argument(
        "--video_file_names",
        type=str,
        nargs="*",
        help="List of video file names to process (optional, if not provided all videos will be processed)",
    )

    parser.add_argument(
        "--start_second", type=int, default=0, help="Start second for processing"
    )
    parser.add_argument(
        "--end_second",
        type=int,
        default=None,
        help="End second for processing (default: None, process till the end)",
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
        "--disable_tqdm", action="store_false", help="Disable tqdm for progress bar"
    )
    parser.add_argument(
        "--disable_parallel", action="store_false", help="Disable parallel processing"
    )
    parser.add_argument("--max_workers", type=int, help="Maximum number of workers")
    parser.add_argument(
        "--maxtasksperchild",
        type=int,
        default=None,
        help="Maximum tasks per child process (default: None)",
    )

    args = parser.parse_args()

    logging.info("Starting ...")
    process_multiple_videos(
        input_folder_path=args.input_folder_path,
        output_folder_path=args.output_folder_path,
        video_file_names=args.video_file_names,
        start_second=args.start_second,
        end_second=args.end_second,
        embedding_model_name=args.embedding_model_name,
        strategies=args.strategies,
        use_tqdm=args.disable_tqdm,
        parallel=args.disable_parallel,
        max_workers=args.max_workers,
        maxtasksperchild=args.maxtasksperchild,
    )

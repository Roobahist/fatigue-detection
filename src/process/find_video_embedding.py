import logging
import multiprocessing
import os
import sys
import traceback
from typing import List, Optional

import cv2
import numpy as np
import tensorflow as tf
import torch
from tqdm import tqdm
from transformers import pipeline
from PIL import Image

from Zh.src.utils.setup_device import setup_tensorflow, setup_torch
from Zh.src.process.frame_processor import FrameProcessor
from Zh.src.utils.csvwriter import CSVWriter
from Zh.src.utils.file_folder import copy_directory_structure, get_file_paths
from Zh.src.utils.memory import cleanup
from Zh.src.utils.serialized_parallel import process
from Zh.src.utils.video import get_video_info

logging.basicConfig(
    level=logging.INFO,  # or DEBUG for more verbosity
    format="[%(asctime)s][%(levelname)s] %(message)s",
)

setup_tensorflow()
device = setup_torch()

VIDEO_EXTS = [".mp4", ".mov", ".avi"]

def process_single_video(
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



def process_multiple_videos(
    input_folder_path: str,
    output_folder_path: str,
    video_file_names: Optional[List[str]] = [],
    start_second: Optional[int] = 0,
    end_second: Optional[int] = None,
    embedding_model: Optional[str] = "Facenet",
    embedding_normalization: Optional[str] = "Facenet",
    parallel: Optional[bool] = False,
    max_workers: Optional[int] = 10,
    maxtasksperchild: Optional[int] = 1,
    use_tqdm: Optional[bool] = False,
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
        video_file_names = [
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

    model_path = "/work/long_lab/Mojtaba/src/process/vit-base-patch16-224-in21k"
    pipe = pipeline(task="image-feature-extraction", model=model_path, device=0)

    args_list = [
        {
            "video_file_path": video_file_paths[i],
            "input_folder_path": input_folder_path,
            "output_folder_path": output_folder_path,
            "start_second": start_second,
            "end_second": end_second,
            "embedding_model": embedding_model,
            "embedding_normalization": embedding_normalization,
            "pipe":pipe,
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

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
from transformers import AutoVideoProcessor, AutoModel

from Zh.src.process.frame_processor import FrameProcessor
from Zh.src.utils.csvwriter import CSVWriter
from Zh.src.utils.file_folder import copy_directory_structure, get_file_paths
from Zh.src.utils.memory import cleanup
from Zh.src.utils.serialized_parallel import process
from Zh.src.utils.video import get_video_info, VideoReader

logging.basicConfig(
    level=logging.INFO,  # or DEBUG for more verbosity
    format="[%(asctime)s][%(levelname)s] %(message)s",
)

VIDEO_EXTS = [".mp4", ".mov", ".avi"]

# Get available devices
gpus = tf.config.list_physical_devices('GPU')

# Determine device strategy
if len(gpus) > 0:
    print(f"Using GPU: {gpus[0]}")
    # Optional: Configure memory growth
    try:
        tf.config.experimental.set_memory_growth(gpus[0], True)
    except RuntimeError as e:
        print(f"Memory growth setting error: {e}")
else:
    print("No GPU found. Using CPU instead.")

if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
else:
    device = torch.device("cpu")
    print("No GPU found. Using CPU instead.")


def process_single_video(
    video_file_path: str,
    input_folder_path: str,
    output_folder_path: str,
    model_path: str,
    downsample_factor: int = 3,
    model=None,
    processor=None,
    chunk_size: int = 64,
    start_second: Optional[int] = 0,
    end_second: Optional[int] = None,
    use_tqdm: Optional[bool] = False,
    file_index: Optional[int] = -1,
):
    try:
        
        logging.info(
            f"--- Processing Video {file_index} - {video_file_path} Ended Successfully ---"
        )

        cap = cv2.VideoCapture(video_file_path)
        if not cap.isOpened():
            logging.error(f"Failed to open video {file_index} - {video_file_path}")
            return

        VR = VideoReader(cap_object = cap)
        video_info = VR.get_info()
        total_frames = video_info["total_frames"]

        downsampled_indices = np.arange(0, total_frames, downsample_factor)
        num_downsampled = len(downsampled_indices)

        for start in range(0, num_downsampled, chunk_size):
            end = start + chunk_size
            if end > num_downsampled:
                break  # drop last incomplete chunk
            chunk = downsampled_indices[start:end]



            

        
    except Exception as e:
        logging.error(
            f"Error Processing Video {video_file_path}: %s", traceback.format_exc()
        )
        try:
            VR.release()
        except:
            pass
        return


def process_multiple_videos(
    input_folder_path: str,
    output_folder_path: str,
    video_file_names: Optional[List[str]] = [],
    model_path = None,
    downsample_factor=3, 
    chunk_size=64,
    start_second: Optional[int] = 0,
    end_second: Optional[int] = None,
    parallel: Optional[bool] = False,
    max_workers: Optional[int] = 10,
    maxtasksperchild: Optional[int] = None,
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

    model = AutoModel.from_pretrained(model_path)
    processor = AutoVideoProcessor.from_pretrained(model_path)

    args_list = [
        {
            "video_file_path"   : video_file_paths[i],
            "input_folder_path" : input_folder_path,
            "output_folder_path": output_folder_path,
            "model_path"        : model_path,
            "downsample_factor" : downsample_factor,
            "model"             : model,
            "processor"         : processor,
            "chunk_size"        : chunk_size,
            "start_second"      : start_second,
            "end_second"        : end_second,
            "use_tqdm"          : use_tqdm,
            "file_index"        : i,
        }
        for i in range(len(video_file_paths))
    ]

    process(
        worker_function=process_single_video,
        worker_params_list=args_list,
        parallel=parallel,
        max_workers=max_workers,
        maxtasksperchild=maxtasksperchild,
        use_tqdm=use_tqdm,
    )


if __name__ == "__main__":

    input_folder_path = sys.argv[1] or "/work/long_lab/Mojtaba/data/UTA/1.4-augmentated_masked"
    output_folder_path = sys.argv[2] or "/work/long_lab/Mojtaba/data/UTA/1.4.2-augmentated_masked_embedded_vjepa"
    max_workers = int(sys.argv[3]) if len(sys.argv) > 3 else 30

    logging.info("Starting ...")

    process_multiple_videos(
        input_folder_path=input_folder_path,
        output_folder_path=output_folder_path,
        video_file_names=[],
        model_path="facebook/vjepa2-vitl-fpc64-256",
        downsample_factor=3, 
        chunk_size=64,
        parallel=True,
        start_second=0,
        end_second=None,
        max_workers=max_workers,
        maxtasksperchild = None,
        use_tqdm=True,
    )
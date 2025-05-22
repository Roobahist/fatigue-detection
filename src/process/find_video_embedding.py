import logging
import multiprocessing
import os
import sys
import traceback
from typing import List, Optional

import cv2
from tqdm import tqdm

from Zh.src.process.frame_processor import FrameProcessor
from Zh.src.utils.csvwriter import CSVWriter
from Zh.src.utils.memory import cleanup
from Zh.src.utils.video import get_video_info
from Zh.src.utils.serialized_parallel import process
from Zh.src.utils.file_folder import copy_directory_structure, get_file_paths

logging.basicConfig(
    level=logging.INFO,  # or DEBUG for more verbosity
    format="[%(asctime)s][%(levelname)s] %(message)s",
)


VIDEO_EXTS = [".mp4", ".mov", ".avi"]


def process_single_video(
    video_file_path: str,
    input_folder_path: str,
    output_folder_path: str,
    start_second: Optional[int] = 0,
    end_second: Optional[int] = None,
    embedding_model: Optional[str] = "Facenet",
    embedding_normalization: Optional[str] = "Facenet",
    use_tqdm: Optional[bool] = False,
    file_index: Optional[int] = -1,
):
    try:
        logging.info(f"--- Processing Video {file_index} - {video_file_path} ---")
        FP = FrameProcessor()

        CSVW = CSVWriter(
            file_path=os.path.splitext(
                video_file_path.replace(input_folder_path, output_folder_path)
            )[0]
            + ".csv"
        )

        cap = cv2.VideoCapture(video_file_path)
        if not cap.isOpened():
            logging.error(f"Failed to open video {file_index} - {video_file_path}")

        fps, _, _, total_frames, _ = get_video_info(cap=cap)
        start_frame = max(int(start_second * fps), CSVW.get_current_line_number(), 0)
        end_frame = int(end_second * fps) if end_second is not None else total_frames

        logging.info(
            f"Video {file_index} - Start Frame: {start_frame}, End Frame: {end_frame}"
        )
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        with tqdm(
            total=end_frame - start_frame,
            desc=f"Processing {os.path.basename(video_file_path)}",
            unit="frame",
            disable=not use_tqdm,
        ) as progress_bar:
            while cap.isOpened() and cap.get(cv2.CAP_PROP_POS_FRAMES) < end_frame:
                current_frame_idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
                if current_frame_idx % 1000 == 0:
                    logging.info(
                        f"Frame {current_frame_idx} : {os.path.basename(video_file_path)}"
                    )
                ret, frame = cap.read()
                if not ret:
                    logging.error(
                        f"Failed to read frame {current_frame_idx} : {os.path.basename(video_file_path)}"
                    )
                    progress_bar.update(1)
                    continue

                frame_embedding = [
                    round(x, 3)
                    for x in FP.represent(
                        img_path=frame,
                        model_name=embedding_model,
                        enforce_detection=False,
                        detector_backend="skip",
                        normalization=embedding_normalization,
                    )[0]["embedding"]
                ]
                CSVW.write(frame_embedding)
                progress_bar.update(1)

        logging.info(
            f"--- Processing Video {file_index} - {video_file_path} Ended Successfully ---"
        )

    except Exception as e:
        logging.error(
            f"Error Processing Video {video_file_path}: %s", traceback.format_exc()
        )
        return False
    finally:
        cleanup(
            var_names=[
                "FP",
                "CSVW",
                "cap",
                "fps",
                "total_frames",
                "progress_bar",
                "ret",
                "frame",
                "frame_embedding",
            ],
            local_vars=locals(),
            global_vars=globals(),
        )


def process_parallel_or_serial(
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
    all_video_file_names = get_file_paths(input_path=input_folder_path, depth=5, EXT_LIST=VIDEO_EXTS)

    if video_file_names != []:
        video_file_names = [video_file_name for video_file_name in all_video_file_names 
                            if os.path.basename(video_file_name) in video_file_names]
        logging.info(
            f"Video File Paths Captured Using Provided List: {len(video_file_paths)}"
        )
    else:
        video_file_paths = all_video_file_names
        logging.info(
            f"Video File Paths Captured Automatically: {len(video_file_paths)}"
        )

    args_list = [
        {
            "video_file_path": video_file_paths[i],
            "input_folder_path": input_folder_path,
            "output_folder_path": output_folder_path,
            "start_second": start_second,
            "end_second": end_second,
            "embedding_model": embedding_model,
            "embedding_normalization": embedding_normalization,
            "use_tqdm": use_tqdm,
            "file_index": i,
        }
        for i in range(len(video_file_paths))
    ]

    process(
        worker_function = process_single_video,
        worker_params_list = args_list,
        handler_function = None,
        parallel = parallel,
        max_workers = max_workers,
        maxtasksperchild = maxtasksperchild,
        use_tqdm = use_tqdm,
    )


if __name__ == "__main__":

    input_folder_path, output_folder_path, max_workers = (
        sys.argv[1],
        sys.argv[2],
        int(sys.argv[3]),
    )
    # input_folder_path = "/work/long_lab/Mojtaba/augmented"
    # output_folder_path = "/work/long_lab/Mojtaba/data/UTA/1.3.2-augmentated_embedded"
    # output_folder_path = "/home/mojtaba.kananisarche/Zh/temp/out/embedding/Facenet"
    # output_folder_path = "/work/long_lab/Mojtaba/data/UTA/1.3.1-augmentated_embedded"
    # input_folder_path, output_folder_path, max_workers = (
    # "/work/long_lab/Mojtaba/augmented/brightness_alpha0.8_beta-10",
    # "/home/mojtaba.kananisarche/Zh/temp/out4",
    # 2,
    # )
    # video_file_names=[
    #     "1_06_5_30fps_brightness_alpha0.8_beta-10.mp4"
    # ],
    logging.info("Starting ...")
    process_parallel_or_serial(
        input_folder_path=input_folder_path,
        output_folder_path=output_folder_path,
        video_file_names=[],
        parallel=True,
        start_second=0,
        end_second=None,
        max_workers=max_workers,
        embedding_model="Facenet",
        embedding_normalization="Facenet",
        use_tqdm=False,
    )
    # python -m Zh.src.process.find_video_embedding /work/long_lab/Mojtaba/augmented/cropped /work/long_lab/Mojtaba/data/UTA/1.3.2-augmentated_embedded/cropped 44

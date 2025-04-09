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

logging.basicConfig(
    level=logging.INFO,  # or DEBUG for more verbosity
    format="[%(levelname)s] %(message)s",
)


VIDEO_EXTS = [".mp4", ".mov", ".avi"]


def find_folders_to_create(input_folder_path, output_folder_path):
    try:
        to_create_folder_paths = [output_folder_path]
        folder_names_inside_input_folder_path = [
            folder_name
            for folder_name in os.listdir(input_folder_path)
            if os.path.isdir(os.path.join(input_folder_path, folder_name))
        ]
        for folder_name in folder_names_inside_input_folder_path:
            to_create_folder_paths.append(os.path.join(output_folder_path, folder_name))
        return to_create_folder_paths
    except Exception as e:
        logging.error(f"Error Finding Folders to Create: %s", traceback.format_exc())
        return False


def create_folders(to_create_folder_paths):
    try:
        for to_create_folder_path in to_create_folder_paths:
            os.makedirs(to_create_folder_path, exist_ok=True)
        return True
    except Exception as e:
        logging.error(f"Error Creating Folders: %s", traceback.format_exc())
        return False


def process_single_video_wrapper(kwargs):
    return process_single_video(**kwargs)


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
        file_path = (
            os.path.splitext(
                video_file_path.replace(input_folder_path, output_folder_path)
            )[0]
            + ".csv"
        )
        # logging.info(f'fp, {file_path}')

        CSVW = CSVWriter(
            file_path=os.path.splitext(
                video_file_path.replace(input_folder_path, output_folder_path)
            )[0]
            + ".csv"
        )

        cap = cv2.VideoCapture(video_file_path)
        if not cap.isOpened():
            logging.error(f"Failed to open video {file_index} - {video_file_path}")

        fps, _, _, total_frames = get_video_info(cap=cap)
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
    parallel: Optional[bool] = False,
    start_second: Optional[int] = 0,
    end_second: Optional[int] = None,
    max_workers: Optional[int] = 10,
    embedding_model: Optional[str] = "Facenet",
    embedding_normalization: Optional[str] = "Facenet",
    use_tqdm: Optional[bool] = False,
):
    """
    video_file_names : relative path after the input folder should be provided,
                       otherwise all the files in all subdirectories would be
                       processed.
    """
    to_create_folder_paths = find_folders_to_create(
        input_folder_path=input_folder_path, output_folder_path=output_folder_path
    )
    if not isinstance(to_create_folder_paths, list):
        return False
    if not create_folders(to_create_folder_paths):
        return False

    logging.info(
        "Folders Created :\n" + "\n".join(f"- {p}" for p in to_create_folder_paths)
    )

    if video_file_names != []:
        video_file_paths = [
            os.path.join(input_folder_path, video_file_name)
            for video_file_name in video_file_names
            if os.path.splitext(video_file_name)[1].lower() in VIDEO_EXTS
        ]
        logging.info(
            f"Video File Paths Captured Using Provided List: {len(video_file_paths)}"
        )
    else:
        video_file_paths = [
            os.path.join(root, file)
            for root, _, files in os.walk(input_folder_path)
            for file in files
            if os.path.splitext(file)[1].lower() in VIDEO_EXTS
        ]
        logging.info(
            f"Video File Paths Captured Automatically: {len(video_file_paths)}"
        )

    if parallel:
        logging.info("Processing in Parallel Mode ...")
        with multiprocessing.Pool(processes=max_workers, maxtasksperchild=1) as pool:
            pool.map(
                process_single_video_wrapper,
                [
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
                ],
            )

    else:
        logging.info("Processing in Serialized Mode ...")
        for i in range(len(video_file_paths)):
            process_single_video(
                **{
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

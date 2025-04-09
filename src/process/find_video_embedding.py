import logging
import multiprocessing
import os
import traceback
from typing import List, Optional

from Zh.src.process.frame_processor import FrameProcessor
from Zh.src.utils.csvwriter import CSVWriter
from Zh.src.utils.video import get_video_info

VIDEO_EXTS = [".mp4", ".mov", ".avi"]


def find_folders_to_create(input_folder_path, output_folder_path):
    to_create_folder_paths = []
    folder_names_inside_input_folder_path = [
        folder_name
        for folder_name in os.listdir(input_folder_path)
        if os.path.isdir(os.path.join(input_folder_path, folder_name))
    ]
    for folder_name in folder_names_inside_input_folder_path:
        to_create_folder_paths.append(os.path.join(output_folder_path, folder_name))
    return to_create_folder_paths


def create_folders(to_create_folder_paths):
    try:
        for to_create_folder_path in to_create_folder_paths:
            os.makedirs(to_create_folder_path, exist_ok=True)
        return True
    except Exception as e:
        logging.error(f"Error Creating Folders: %s", traceback.format_exc())
        return False


def process_single_video(
    video_file_path: str,
    input_folder_path: str,
    output_folder_path: str,
    start_second: Optional[int] = 0,
    end_second: Optional[int] = None,
    embedding_model: Optional[str] = "Facenet",
    embedding_normalization: Optional[str] = "Facenet",
):
    FP = FrameProcessor()
    CSVW = CSVWriter(
        file_path=os.path.splitext(
            video_file_path.replace(input_folder_path, output_folder_path)
        )[0]
        + ".csv"
    )
    

    pass


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
):
    """
    video_file_names : relative path after the input folder should be provided,
                       otherwise all the files in all subdirectories would be
                       processed.
    """

    if video_file_names != []:
        video_file_paths = [
            os.path.join(input_folder_path, video_file_name)
            for video_file_name in video_file_names
            if os.path.splitext(video_file_name)[1].lower() in VIDEO_EXTS
        ]
    else:
        video_file_paths = [
            os.path.join(root, file)
            for root, _, files in os.walk(input_folder_path)
            for file in files
            if os.path.splitext(file)[1].lower() in VIDEO_EXTS
        ]

    if parallel:
        with multiprocessing.Pool(processes=max_workers, maxtasksperchild=1) as pool:
            pool.starmap(
                process_single_video,
                [
                    {
                        "video_file_path": video_file_path,
                        "input_folder_path": input_folder_path,
                        "output_folder_path": output_folder_path,
                        "start_second": start_second,
                        "end_second": end_second,
                        "embedding_model": embedding_model,
                        "embedding_normalization": embedding_normalization,
                    }
                    for video_file_path in video_file_paths
                ],
            )

    else:
        for video_file_path in video_file_paths:
            process_single_video(
                **{
                    "video_file_path": video_file_path,
                    "input_folder_path": input_folder_path,
                    "output_folder_path": output_folder_path,
                    "start_second": start_second,
                    "end_second": end_second,
                    "embedding_model": embedding_model,
                    "embedding_normalization": embedding_normalization,
                }
            )


if __name__ == "__main__":
    input_folder_path = "/home/mojtaba.kananisarche/Zh/augmented"
    output_folder_path = "/work/long_lab/Mojtaba/data/UTA/1.3.1-augmentated_embedded"

    to_create_folder_paths = find_folders_to_create(
        input_folder_path=input_folder_path, output_folder_path=output_folder_path
    )
    if not create_folders(to_create_folder_paths):
        exit(1)

    process_parallel_or_serial(
        input_folder_path=input_folder_path,
        output_folder_path=output_folder_path,
    )

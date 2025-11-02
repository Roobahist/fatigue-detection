import os
from typing import List, Optional

import numpy as np

from Zh.src.utils.common import read_obj
from Zh.src.utils.file_folder import (
    copy_directory_structure,
    move_file,
    pick_selected_file_paths,
)
from Zh.src.utils.memmap_handler import MemmapHandler
from Zh.src.utils.serialized_parallel import process


def process_single_video(
    input_folder: str,
    output_folder: str,
    file_path: str,
    start_second: Optional[int] = 0,
    end_second: Optional[int] = None,
    video_fps: Optional[int] = 15,
    weights: Optional[dict] = {},
    file_index: Optional[int] = 0,
):
    logging.info(f"Processing file {file_index}: {file_path}")

    mh_input = MemmapHandler(
        path=file_path,
        mode="r",
    )
    mh_output = MemmapHandler(
        path=os.path.join(output_folder, os.path.relpath(file_path, input_folder)).replace(".dat", "__eventfulness.dat"),
        shape=(mh_input.shape[0], 1),
        dtype="float32",
        mode="w+",
    )

    start_frame = int(start_second * video_fps)
    end_frame = int(end_second * video_fps) if end_second is not None else mh_input.shape[0]

    for k in weights.keys():
        if k not in mh_input._column_names:
            logging.error(
                f"Key {k} not found in input memmap columns of file {file_index}: {file_path}."
            )
            raise ValueError(f"Key {k} not found in input memmap columns.")

    eventfulness = np.zeros((end_frame - start_frame, 1))
    for k, v in weights.items():
        col_idx = mh_input._column_names.index(k)
        eventfulness += (
            v * mh_input.read(slice=(slice(start_frame, end_frame), slice(col_idx, col_idx + 1)))
        )

    mh_output.write(eventfulness)
    mh_output.close()
    mh_input.close()

    logging.info(f"Finished processing file {file_index}: {file_path}")


def process_multiple_videos(
    input_folder: str,
    output_folder: str,
    path_to_weights: str,
    relative_selected_paths: Optional[List[str]] = [],
    start_second: Optional[int] = 0,
    end_second: Optional[int] = None,
    video_fps: Optional[int] = 15,
    parallel: Optional[bool] = False,
    max_workers: Optional[int] = 10,
    maxtasksperchild: Optional[int] = 1,
    use_tqdm: Optional[bool] = True,
):

    copy_directory_structure(input_folder, output_folder)

    weights = read_obj(path_to_weights, file_type="json")
    move_file(path_to_weights, os.path.join(output_folder, "weights.json"), mode="cut")

    selected_file_paths = pick_selected_file_paths(
        input_folder=input_folder,
        depth=1,
        EXT_LIST=[".dat"],
        relative_selected_paths=relative_selected_paths,
    )

    
    args_list = [
        {
            "input_folder": input_folder,
            "output_folder": output_folder,
            "file_path": file_path,
            "start_second": start_second,
            "end_second": end_second,
            "video_fps": video_fps,
            "weights": weights,
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

    parser = argparse.ArgumentParser(description="Process video files")
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
        "path_to_weights",
        type=str,
        help="path to weights for finding weighted sum of blendshapes as a representation of eventfulness.",
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
        "--video_fps",
        type=int,
        default=15,
        help="fps of the source video that resulted in the .dat files",
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
        video_fps=args.video_fps,
        path_to_weights=args.path_to_weights,
        parallel=args.parallel,
        max_workers=args.max_workers,
        maxtasksperchild=args.maxtasksperchild,
        use_tqdm=args.use_tqdm,
    )

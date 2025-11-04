import os
from typing import List, Optional

from Zh.src.utils.eventfulness import rank_eventfulness
from Zh.src.utils.file_folder import copy_directory_structure, pick_selected_file_paths
from Zh.src.utils.serialized_parallel import process
from Zh.src.utils.video import VideoReader, video_to_gif


def process_single_video(
    input_folder: str,
    eventfulness_scores_folder: str,
    output_folder: str,
    file_path: str,
    stride: int = 30,
    window_size: int = 90,
    video_fps: int = 15,
    file_index: int = 0,
):
    logging.info(f"Processing file {file_index}: {file_path}")

    eventfulness_scores_path = os.path.join(
        eventfulness_scores_folder,
        os.path.relpath(file_path, input_folder).split("__")[0]
        + "__features__eventfulness.dat",
    )

    ranked_segments = rank_eventfulness(
        evenfulness_scores_path=eventfulness_scores_path,
        stride=stride,
        window_size=window_size,
    )

    most_eventful_start_frame, _, _ = ranked_segments[0]
    least_eventful_start_frame, _, _ = ranked_segments[-1]

    vr = VideoReader(file_path)

    most_eventful_output_path = os.path.join(
        output_folder,
        os.path.relpath(file_path, input_folder).replace(".mp4", "__most_eventful.gif"),
    )
    least_eventful_output_path = os.path.join(
        output_folder,
        os.path.relpath(file_path, input_folder).replace(
            ".mp4", "__least_eventful.gif"
        ),
    )

    video_to_gif(
        input_path=file_path,
        output_path=most_eventful_output_path,
        start_time=most_eventful_start_frame / video_fps,
        duration=window_size / video_fps,
        fps=video_fps,
    )

    video_to_gif(
        input_path=file_path,
        output_path=least_eventful_output_path,
        start_time=least_eventful_start_frame / video_fps,
        duration=window_size / video_fps,
        fps=video_fps,
    )

    vr.release()

    logging.info(f"Finished processing file {file_index}: {file_path}")


def process_multiple_videos(
    input_folder: str,
    eventfulness_scores_folder: str,
    output_folder: str,
    relative_selected_paths: List[str] = None,
    file_indices_spec: Optional[str] = "",
    stride: int = 30,
    window_size: int = 90,
    video_fps: int = 15,
    parallel: bool = False,
    max_workers: int = None,
    maxtasksperchild: int = None,
    use_tqdm: bool = True,
):
    """
    Process multiple videos to find the most and least eventful segments.

    Args:
        input_folder (str): Path to the input folder containing video files.
        eventfulness_scores_folder (str): Path to the folder containing eventfulness scores .dat files.
        output_folder (str): Path to the output folder to save processed videos.
        relative_selected_paths (List[str], optional): List of relative file paths to process. If not provided, all files in the input folder are processed.
        stride (int): Stride for segmenting the eventfulness scores.
        window_size (int): Window size for each segment.
        video_fps (int): Frames per second of the source video that resulted in the .dat files. Default is 15.
    """

    copy_directory_structure(input_folder, output_folder)
    
    file_index_to_path = pick_selected_file_paths(
        input_folder=input_folder,
        depth=1,
        EXT_LIST=[".mp4"],
        relative_selected_paths=relative_selected_paths,
        file_indices_spec=file_indices_spec,
        sort_and_dedupe=True,
    )

    args_list = [
        {
            "input_folder": input_folder,
            "eventfulness_scores_folder": eventfulness_scores_folder,
            "output_folder": output_folder,
            "file_path": file_path,
            "stride": stride,
            "window_size": window_size,
            "video_fps": video_fps,
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
        auto_slice=True,
    )


if __name__ == "__main__":
    import argparse
    import logging

    logging.basicConfig(
        level=logging.INFO,  # or DEBUG for more verbosity
        format="[%(asctime)s][%(levelname)s] %(message)s",
    )

    parser = argparse.ArgumentParser(description="Find Most and Least Eventful Video Segments")
    
    parser.add_argument(
        "input_folder",
        type=str,
        help="Path to the input folder containing video files.",
    )
    parser.add_argument(
        "eventfulness_scores_folder",
        type=str,
        help="Path to the folder containing eventfulness scores .dat files.",
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
        help="Comma-separated indices/ranges, e.g. '0-9,20,35-'",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=30,
        help="Stride for segmenting the eventfulness scores.",
    )
    parser.add_argument(
        "--window_size",
        type=int,
        default=90,
        help="Window size for each segment.",
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
        eventfulness_scores_folder=args.eventfulness_scores_folder,
        output_folder=args.output_folder,
        relative_selected_paths=args.relative_selected_paths,
        file_indices_spec=args.file_indices_spec,
        stride=args.stride,
        window_size=args.window_size,
        video_fps=args.video_fps,
        parallel=args.parallel,
        max_workers=args.max_workers,
        maxtasksperchild=args.maxtasksperchild,
        use_tqdm=args.use_tqdm,
    )

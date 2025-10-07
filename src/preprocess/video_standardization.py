import logging
import os
from pathlib import Path

from Zh.src.utils.file_folder import copy_directory_structure, get_file_paths
from Zh.src.utils.serialized_parallel import process
from Zh.src.utils.video import convert_video


def standardize_videos(
    input_directory_path: str,
    output_directory_path: str,
    video_file_names: list[str] = [],
    target_fps: int = 15,
    target_ext: str = ".mp4",
    no_audio: bool = True,
    quality_level: str = "fast",
    parallel: bool = True,
    use_gpu: bool = False,
) -> None:
    """
    Standardizes the format and FPS of all videos in a folder.

    Args:
        input_directory_path (str): Path to the input folder containing video files.
        output_directory_path (str): Path to the output folder to save processed videos.
        target_fps (int): Desired frames per second (default: 30).
        target_ext (str): Desired video format (default: "mp4").
    """
    logging.info("input_directory_path", input_directory_path)
    logging.info("output_directory_path", output_directory_path)
    logging.info("video_file_names", video_file_names)
    logging.info("target_fps", target_fps)
    logging.info("target_ext", target_ext)
    logging.info("no_audio", no_audio)
    logging.info("quality_level", quality_level)
    logging.info("parallel", parallel)
    logging.info("use_gpu", use_gpu)

    copy_directory_structure(input_directory_path, output_directory_path)
    input_video_paths = sorted(
        get_file_paths(
            input_path=input_directory_path,
            depth=5,
            EXT_LIST=[".mp4", ".avi", ".mov", ".mkv"],
        )
    )
    args_list = [
        {
            "input_path": input_path,
            "output_directory_path": output_directory_path,
            "output_file_name": Path(input_path).name,
            "target_ext": target_ext,
            "target_fps": target_fps,
            "no_audio": no_audio,
            "quality_level": quality_level,
            "use_gpu": use_gpu,
        }
        for input_path in input_video_paths
        if Path(input_path).name in video_file_names 
        or not video_file_names
    ]

    # The process function now handles SLURM slicing automatically
    process(
        worker_function=convert_video,
        worker_params_list=args_list,
        parallel=parallel,
        use_tqdm=True,
        auto_slice=True,
    )


if __name__ == "__main__":
    import argparse
    logging.info("Process ID:",os.getpid())

    parser = argparse.ArgumentParser(description="Change FPS of videos in a folder.")
    parser.add_argument(
        "input_directory_path",
        type=str,
        help="Path to the input folder containing video files.",
    )
    parser.add_argument(
        "output_directory_path",
        type=str,
        help="Path to the output folder to save processed videos.",
    )
    parser.add_argument(
        "video_file_names",
        type=str,
        nargs="*",
        help="Names of the video files to process.",
    )
    parser.add_argument(
        "--target_fps",
        type=int,
        default=15,
        help="Desired frames per second (default: 15).",
    )
    parser.add_argument(
        "--target_ext",
        type=str,
        default=".mp4",
        help="Desired video format (default: 'mp4').",
    )
    parser.add_argument(
        "--no_audio", action="store_true", help="Remove audio from the video."
    )
    parser.add_argument(
        "--quality_level",
        type=str,
        default="fast",
        help="Quality preset - 'fast', 'medium', 'high', 'best' (default: 'fast').",
    )
    parser.add_argument(
        "--parallel", action="store_true", help="Use parallel processing."
    )
    parser.add_argument("--use_gpu", action="store_true", help="Use GPU acceleration.")

    args = parser.parse_args()

    standardize_videos(
        input_directory_path=args.input_directory_path,
        output_directory_path=args.output_directory_path,
        video_file_names=args.video_file_names,
        target_fps=args.target_fps,
        target_ext=args.target_ext,
        no_audio=args.no_audio,
        quality_level=args.quality_level,
        parallel=args.parallel,
        use_gpu=args.use_gpu,
    )

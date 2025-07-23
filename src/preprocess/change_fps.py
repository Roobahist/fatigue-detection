import os

from Zh.src.utils.file_folder import copy_directory_structure, get_file_paths
from Zh.src.utils.video import change_fps
from Zh.src.utils.serialized_parallel import process, get_task_slice

def change_fps_folder(input_folder: str, 
                      output_folder: str, 
                      target_fps: int = 30) -> None:
    """
    Changes the FPS of all videos in a folder using FFmpeg.

    Args:
        input_folder (str): Path to the input folder containing video files.
        output_folder (str): Path to the output folder to save processed videos.
        target_fps (int): Desired frames per second (default: 30).
    """
    copy_directory_structure(input_folder, output_folder)
    input_video_paths = sorted(get_file_paths(
        input_path=input_folder,
        depth=5,
        EXT_LIST=[".mp4", ".avi", ".mov", ".mkv"]
    ))
    start, end, task_id, total_tasks = get_task_slice(len(input_video_paths))
    print(f"[SLURM] Task {task_id}/{total_tasks} → processing files {start}:{end}")

    args_list = [
    {
        "input_path": input_path,
        "output_path": os.path.join(
            output_folder, os.path.relpath(input_path, input_folder)).replace('30fps', '10fps'),
        "target_fps": target_fps
    } for input_path in input_video_paths
    ][start:end]

    process(
        worker_function=change_fps,
        worker_params_list=args_list,
        parallel=False,
        use_tqdm=True)
    

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Change FPS of videos in a folder.")
    parser.add_argument("input_folder", type=str, help="Path to the input folder containing video files.")
    parser.add_argument("output_folder", type=str, help="Path to the output folder to save processed videos.")
    parser.add_argument("--target_fps", type=int, default=30, help="Desired frames per second (default: 30).")

    args = parser.parse_args()
    
    change_fps_folder(args.input_folder, args.output_folder, args.target_fps)

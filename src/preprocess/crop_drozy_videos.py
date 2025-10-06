# since subjects are not centered in the drozy videos and also the videos are greyscale
# the performance of mediapipe in detecting the face for cropping and aligning is not
# adequate and for that reason we need to first crop the videos based on the automatically
# generated landmarks from the dataset itself and then provide those videos to the mediapipe.

from Zh.src.utils.file_folder import (
    copy_directory_structure,
    create_folders,
    get_file_paths,
)
from Zh.src.utils.serialized_parallel import process
from Zh.src.utils.video import *


def process_single_video(
    landmark_folder: str,
    input_folder: str,
    output_folder: str,
    video_file: str,
    start_second: int = 0,
    end_second: Optional[int] = None,
    expansion_factor: float = 3.0,
    output_size: Union[Tuple[int, int], str] = "auto",
):
    landmark_file_path = os.path.join(landmark_folder, video_file.replace(".mp4", "-s2.txt"))
    if not os.path.exists(landmark_file_path):
        raise ValueError(f"Landmark file {landmark_file_path} does not exist.")
    
    input_file_path = os.path.join(input_folder, video_file)
    if not os.path.exists(input_file_path):
        raise ValueError(f"Input file {input_file_path} does not exist.")
    
    output_file_path = os.path.join(output_folder, video_file)
    if os.path.exists(output_file_path):
        if is_video_valid(output_file_path):
            logging.info(f"Skipping already processed video: {input_file_path}")
            return
        else:
            logging.info(f"Removing corrupted video: {input_file_path}")
            os.remove(input_file_path)
            logging.info(f"Removing corrupted video: {output_file_path} and reprocessing")

    with open(landmark_file_path, "r") as f:
        frames_landmarks = [[[int(float(line.split(' ')[2*coordinates])), int(float(line.split(' ')[2*coordinates+1]))] 
                            for coordinates in range(int(len(line.split(' '))/2))]
                            for line in f.readlines()]
        
    vr = VideoReader(video_path = input_file_path)
    input_video_info = vr.get_info()
    fps = input_video_info['fps']
    fourcc = input_video_info['fourcc']
    total_frames = input_video_info['total_frames']

    if len(frames_landmarks) != input_video_info['total_frames']:
        raise ValueError(f"Number of frames in landmarks file {landmark_file_path} does not match number of frames in video {input_file_path}")

    center_face_landmark_idx = 29

    if output_size == "auto":
        temp_frame_idx = 0
        temp_frame = vr.read_frame(frame_index=temp_frame_idx, jump=False)[1]
        if temp_frame is None:
            raise ValueError(f"Could not read frame from video {input_file_path}")
        ys = [y for _, y in frames_landmarks[0]]
        side = int(expansion_factor*(max(ys)-min(ys)))
        output_size = (side, side)

    if not isinstance(output_size, tuple):
        raise ValueError(f"Output size must be a tuple, got {output_size} instead.")
    if len(output_size) != 2:
        raise ValueError(f"Output size must be a tuple of length 2, got {output_size} instead.")

    vw = VideoWriter(output_path=output_file_path, 
                     fps = fps,
                     frame_size = output_size,
                     fourcc = fourcc)
    
    start_frame = int(start_second * fps)
    end_frame = (
        int(end_second * fps) if end_second is not None else total_frames
    )

    for frame_idx in range(start_frame, end_frame):
        frame = vr.read_frame(frame_index=frame_idx, jump=True)[1]
        if frame is None:
            raise ValueError(f"Could not read frame from video {input_file_path}")
        frame_landmarks = frames_landmarks[frame_idx]
        face_center = frame_landmarks[center_face_landmark_idx]
        xs, ys = map(list, zip(*frame_landmarks))

        face_w = max(xs) - min(xs)
        face_h = max(ys) - min(ys)
        face_size = max(face_w, face_h)

        half_exp = int((expansion_factor / 2) * face_size)

        x = max(0, face_center[0] - half_exp)
        y = max(0, face_center[1] - half_exp)
        w = min(input_video_info['width'] - x, half_exp * 2)
        h = min(input_video_info['height'] - y, half_exp * 2)

        cropped_frame = frame[y:y+h, x:x+w]
        vw.write_frame(cropped_frame)

    pass


def process_multiple_videos(
    landmark_folder: str,
    input_folder: str,
    output_folder: str,
    video_files: List[str],
    start_second: int = 0,
    end_second: Optional[int] = None,
    expansion_factor: float = 3.0,
    output_size: Union[Tuple[int, int], str] = "auto",
    parallel: bool = True,
    max_workers: Union[int, None] = None,
    maxtasksperchild: Union[int, None] = None,
    use_tqdm: bool = True,
):
    if not os.path.exists(landmark_folder):
        raise ValueError(f"Landmark folder {landmark_folder} does not exist.")
    if not os.path.exists(input_folder):
        raise ValueError(f"Input folder {input_folder} does not exist.")

    copy_directory_structure(input_folder, output_folder)

    video_files = video_files or get_file_paths(
        input_path=input_folder,
        depth=5,
        EXT_LIST=["mov", "mp4", "mkv", "avi"],
        return_full_path=False,
    )

    args_list = [
        {
            "landmark_folder": landmark_folder,
            "input_folder": input_folder,
            "output_folder": output_folder,
            "video_file": video_file,
            "start_second": start_second,
            "end_second": end_second,
            "expansion_factor": expansion_factor,
            "output_size": output_size,
        }
        for video_file in video_files
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

    pass


if __name__ == "__main__":
    import argparse
    import ast

    parser = argparse.ArgumentParser(description="Crop Drozy Videos")
    parser.add_argument("landmark_folder", type=str, help="Landmark folder")
    parser.add_argument("input_folder", type=str, help="Input folder")
    parser.add_argument("output_folder", type=str, help="Output folder")
    parser.add_argument(
        "video_files",
        nargs="*",
        help="List of video file names to process.",
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
        "--expansion_factor",
        type=float,
        default=3.0,
        help="expansion factor with respect to the height of the face for cropping.",
    )
    parser.add_argument(
        "--output_size",
        type=str,
        default="auto",
        help="Output size for the cropped video.",
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
    
    process_multiple_videos(
        landmark_folder=args.landmark_folder,
        input_folder=args.input_folder,
        output_folder=args.output_folder,
        video_files=args.video_files,
        start_second=args.start_second,
        end_second=args.end_second,
        expansion_factor=args.expansion_factor,
        output_size=ast.literal_eval(args.output_size),
        parallel=args.parallel,
        max_workers=args.max_workers,
        maxtasksperchild=args.maxtasksperchild,
        use_tqdm=args.use_tqdm,
    )

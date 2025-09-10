import logging
import os
import subprocess
from typing import Dict, List, Optional, Tuple, Union

import cv2
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm


def is_video_valid(video_path: str) -> bool:
    if not os.path.exists(video_path) or os.path.getsize(video_path) == 0:
        return False

    command = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v",
        "-show_entries",
        "stream=codec_type,nb_read_frames,nb_frames",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        video_path,
    ]

    try:
        result = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=10,
            check=True,
        )
        output = result.stdout.decode("utf-8").strip().split("\n")
        if "video" in output:
            frame_counts = [int(s) for s in output if s.isdigit()]
            return any(f > 0 for f in frame_counts)
        return False
    except Exception:
        return False

def draw_frame(frame: np.ndarray, title: str = "Video Frame"):
    """
    Displays a given frame using matplotlib, suitable for Jupyter Notebooks.

    Args:
        frame (np.ndarray): The frame image (NumPy array) to display.
        title (str): The title for the plot. Defaults to "Video Frame".
    """
    if frame is not None:
        # Convert BGR (OpenCV default) to RGB (Matplotlib default)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        plt.figure(figsize=(6, 6))  # Optional: Adjust figure size
        plt.imshow(frame_rgb)
        plt.title(title)
        plt.axis("off")  # Hide axes
        plt.show()
    else:
        print("Warning: Cannot draw a None frame.")

def change_fps(
    input_path: str,
    output_path: str,
    target_fps: int = 30,
) -> bool:
    """
    Changes the FPS of a single video using FFmpeg.

    Args:
        input_path (str): Path to the input video file.
        output_path (str): Path to save the processed video file.
        target_fps (int): Desired frames per second (default: 30).

    Returns:
        bool: True if processing succeeded, False otherwise.
    """
    if os.path.exists(output_path):
        if is_video_valid(output_path):
            return True

    command = [
        "ffmpeg",
        "-y",  # Overwrite output if it exists
        "-i",
        input_path,
        "-r",
        str(target_fps),
        output_path,
    ]

    try:
        result = subprocess.run(
            command,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        logging.info(f"Successfully changed FPS for: {os.path.basename(input_path)}")
        return True
    except subprocess.CalledProcessError as e:
        logging.error(
            f"Error changing FPS for {os.path.basename(input_path)}: {e.stderr.decode().strip()}"
        )
        return False

def save_video(
    frames: List[np.ndarray],
    output_path: str,
    fps: float,
    use_tqdm: bool = True,
    fourcc: str = "mp4v",
) -> None:
    """
    Saves processed frames as a video.

    Args:
        frames: List of processed frames (NumPy arrays).
        output_path: Path to save the output video.
        fps: Frames per second of the video.
    """
    if not frames:
        logging.warning(f"No frames to save for {output_path}. Skipping video save.")
        return

    first_frame_height, first_frame_width = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*fourcc)
    out = cv2.VideoWriter(
        output_path, fourcc, fps, (first_frame_width, first_frame_height)
    )

    with tqdm(
        total=end_frame - start_frame,
        desc=f"Saving {os.path.basename(output_path)}",
        unit="frame",
        disable=not use_tqdm,
    ) as progress_bar:
        for frame in frames:
            resized_frame = cv2.resize(
                frame,
                (first_frame_width, first_frame_height),
                interpolation=cv2.INTER_LINEAR,
            )
            out.write(resized_frame)
            progress_bar.update(1)
        out.release()
    logging.info(f"Augmented video saved at: {output_path}")


class VideoReaderInitializationError(Exception):
    pass


class VideoReader:
    def __init__(
        self,
        video_path: Optional[str] = None,
        cap_object: Optional[cv2.VideoCapture] = None,
        validate: bool = True,
    ):
        self.video_path = video_path

        if cap_object is not None and isinstance(cap_object, cv2.VideoCapture):
            if not cap_object.isOpened():
                raise VideoReaderInitializationError(
                    "cv2.VideoCapture object is not opened."
                )
            self.cap = cap_object
        elif video_path is not None:
            if validate and not is_video_valid(video_path):
                raise VideoReaderInitializationError(
                    f"Invalid video file: {video_path}"
                )
            self.cap = cv2.VideoCapture(video_path)
            if not self.cap.isOpened():
                raise VideoReaderInitializationError(
                    f"cv2 could not open video: {video_path}"
                )
        else:
            raise VideoReaderInitializationError(
                "Must provide either video_path or cap_object."
            )

        # metadata
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fourcc = int(self.cap.get(cv2.CAP_PROP_FOURCC))
        self.current_frame_index = -1
    
    def get_frame(
        self, 
        frame_index: Union[int, str],
        jump: bool = True,
    ) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Retrieves the frame at the specified index.

        Args:
            frame_index (Union[int, str]): The 0-based index of the frame to read,
                                        or "next"/"prev"/"current" for relative access.
            jump (bool): If True, updates the internal current_frame_index and moves the pointer.
                        If False, resets pointer after read. Defaults to True.

        Returns:
            Tuple[bool, Optional[np.ndarray]]: Success flag and the frame (or None).
        """

        # Resolve frame_index to int
        # Not currenltly in use, but could be useful
        if frame_index == "init":
            frame_index = -1
            self.current_frame_index = -1
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            return True, None

        if frame_index == "next":
            frame_index = self.current_frame_index + 1
        elif frame_index == "prev":
            frame_index = self.current_frame_index - 1
        elif frame_index == "current":
            frame_index = self.current_frame_index
        if not isinstance(frame_index, int):
            print("Error: frame_index must be an int or one of 'next', 'prev', 'current'.")
            return False, None

        # Validate frame_index
        if frame_index < 0 or frame_index >= self.total_frames and self.cap.isOpened() and self.cap:
            print(f"Error: frame_index {frame_index} is out of bounds 0 - {self.total_frames-1} or video not opened.")
            return False, None

        # Save position if peeking
        original_index = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES)) if not jump else None

        try:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ret, frame = self.cap.read()

            if not ret:
                return False, None

            if jump:
                self.current_frame_index = frame_index
            else:
                # Reset to previous position if not jumping
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, original_index)

            return True, frame

        except Exception as e:
            print(f"Error reading frame at index {frame_index}: {e}")
            return False, None

    def get_info(self) -> Dict[str, Union[int, float]]:
        """
        Returns a dictionary containing information about the video.

        Returns:
            Dict[str, Union[int, float]]: A dictionary with video properties like 'total_frames', 'fps',
                  'width', 'height', and 'current_frame_index'.
        """
        return {
            "total_frames": self.total_frames,
            "fps": self.fps,
            "width": self.width,
            "height": self.height,
            "fourcc": self.fourcc,
            "current_frame_index": self.current_frame_index,
        }

    def release(self):
        """
        Releases the video capture object. This should be called when you are
        finished with the video to free up resources.
        """
        if self.cap and self.cap.isOpened():
            self.cap.release()
            print("Video capture released.")

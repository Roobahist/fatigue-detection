import logging
import os
import subprocess
import traceback
from pathlib import Path
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


def change_fps(
    input_path: str,
    output_path: str,
    target_fps: float = 30.0,
    no_audio: bool = True,
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
    if not isinstance(input_path, str) or not isinstance(output_path, str):
        raise TypeError("Input and output paths must be strings")
    if not isinstance(target_fps, int):
        raise TypeError("Target FPS must be a float")
    if target_fps <= 0:
        raise ValueError("Target FPS must be greater than 0")

    if os.path.exists(output_path):
        if is_video_valid(output_path):
            logging.info(f"FPS already changed for: {os.path.basename(input_path)}")
            return True

    command = [
        "ffmpeg",
        "-y",
        "-i",
        input_path,
        "-r",
        str(target_fps)] + \
        (["-an"] if no_audio else []) + \
        [output_path]

    try:
        _ = subprocess.run(
            command,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=60,
        )
        logging.info(f"Successfully changed FPS for: {os.path.basename(input_path)}")
        return True
    except Exception as e:
        logging.error(
            f"Error changing FPS for: {os.path.basename(input_path)}: {traceback.format_exc()}"
        )
        return False


def change_format(
    input_path: str,
    output_path: str,
    target_ext: str = ".mp4",
    no_audio: bool = True,
) -> bool:
    """
    Converts a video to a different format/container using FFmpeg.

    Args:
        input_path (str): Path to the input video file.
        output_path (str): Path to save the converted video file.
        target_ext (str): Target extension (default: ".mp4").

    Returns:
        bool: True if processing succeeded, False otherwise.
    """
    if not isinstance(input_path, str) or not isinstance(output_path, str):
        raise TypeError("Input and output paths must be strings")
    if not isinstance(target_ext, str):
        raise TypeError("Target extension must be a string")
    if not target_ext.startswith("."):
        raise ValueError("Target extension must start with '.'")

    # Ensure output path uses the correct extension
    output_file = Path(output_path).with_suffix(target_ext)

    if os.path.exists(output_file):
        if is_video_valid(output_path):
            logging.info(f"Format already converted for: {os.path.basename(input_path)}")
            return True

    command = [
        "ffmpeg",
        "-y",
        "-i",
        input_path] + \
        (["-an"] if no_audio else []) + \
        [str(output_file)]

    try:
        _ = subprocess.run(
            command,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=60,
        )
        logging.info(
            f"Successfully converted format for: {os.path.basename(input_path)}"
        )
        return True
    except Exception:
        logging.error(
            f"Error converting format for: {os.path.basename(input_path)}: {traceback.format_exc()}"
        )
        return False


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
            print(
                "Error: frame_index must be an int or one of 'next', 'prev', 'current'."
            )
            return False, None

        # Validate frame_index
        if (
            frame_index < 0
            or frame_index >= self.total_frames
            and self.cap.isOpened()
            and self.cap
        ):
            print(
                f"Error: frame_index {frame_index} is out of bounds 0 - {self.total_frames-1} or video not opened."
            )
            return False, None

        # Save position if peeking
        original_index = (
            int(self.cap.get(cv2.CAP_PROP_POS_FRAMES)) if not jump else None
        )

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


class VideoWriter:
    def __init__(
        self,
        output_path: str,
        fps: float,
        frame_size: Tuple[int, int],
        fourcc: str = "mp4v",
    ):
        """
        Initializes the VideoWriter object.
        Args:
            output_path (str): Path to save the output video.
            fps (float): Frames per second of the video.
            frame_size (Tuple[int, int]): Size of the video frames as (width, height).
            fourcc (str): FourCC code for the codec (default: "mp4v").
        Raises:
            ValueError: If frame_size is not a tuple of two integers.
        """
        self.output_path = output_path
        self.fps = fps
        self.frame_size = frame_size  # (width, height)
        self.fourcc = cv2.VideoWriter_fourcc(*fourcc)
        self.writer = cv2.VideoWriter(output_path, self.fourcc, fps, frame_size)
        if not self.writer.isOpened():
            raise IOError(f"Could not open video writer for {output_path}")

    def write_frame(self, frame: np.ndarray) -> bool:
        """
        Writes a single frame to the video file.

        Args:
            frame (np.ndarray): The frame image (NumPy array) to write.

        Returns:
            bool: True if the frame was written successfully, False otherwise.
        """
        if frame.shape[1] != self.frame_size[0] or frame.shape[0] != self.frame_size[1]:
            frame = cv2.resize(
                frame,
                self.frame_size,
                interpolation=cv2.INTER_LINEAR,
            )

        try:
            self.writer.write(frame)
            return True
        except Exception as e:
            print(f"Error writing frame: {e}")
            return False

    def write_frames(self, frames: List[np.ndarray], use_tqdm: bool = True):
        """
        Writes multiple frames to the video file.

        Args:
            frames (List[np.ndarray]): List of frame images (NumPy arrays) to write.
            use_tqdm (bool): Whether to display a progress bar. Defaults to True.
        """
        for frame in tqdm(
            frames,
            desc=f"Writing to {os.path.basename(self.output_path)}",
            unit="frame",
            disable=not use_tqdm,
        ):
            if not self.write_frame(frame):
                print("Stopping write due to error.")
                break

    def release(self):
        """
        Releases the video writer object. This should be called when you are
        finished writing the video to free up resources.
        """
        if self.writer:
            self.writer.release()
            print(f"Video writer released and file saved at: {self.output_path}")

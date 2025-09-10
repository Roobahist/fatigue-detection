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
    self,
    frames: List[np.ndarray],
    output_path: str,
    fps: float,
    start_frame: int,
    end_frame: int,
    fourcc: str = "mp4v",
) -> None:
    """
    Saves processed frames as a video.

    Args:
        frames: List of processed frames (NumPy arrays).
        output_path: Path to save the output video.
        fps: Frames per second of the video.
        start_frame: Start frame of the processing.
        end_frame: End frame of the processing.
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
        disable=not self.use_tqdm,
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
        self.current_frame_index = 0

    def _read_frame_at_index(
        self, frame_index: int
    ) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Helper method to read a specific frame by its index.

        Args:
            frame_index (int): The 0-based index of the frame to read.

        Returns:
            Tuple[bool, Optional[np.ndarray]]: A tuple containing:
                - bool: True if the frame was successfully read, False otherwise.
                - np.ndarray | None: The frame image as a NumPy array if successful,
                                     None otherwise.
        """
        if (
            not self.cap
            or not self.cap.isOpened()
            or frame_index < 0
            or frame_index >= self.total_frames
        ):
            return False, None

        # Set the position to the desired frame
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ret, frame = self.cap.read()
        if ret:
            self.current_frame_index = frame_index
        return ret, frame

    def get_current_frame(self) -> Optional[np.ndarray]:
        """
        Retrieves the frame at the current position.

        Returns:
            Optional[np.ndarray]: The current frame image as a NumPy array, or None if
                               no frame is currently loaded or video is not open.
        """
        if self.current_frame_index == -1 or not self.cap or not self.cap.isOpened():
            return False, None
        # Re-read the current frame to ensure it's up-to-date
        ret, frame = self._read_frame_at_index(self.current_frame_index)
        return ret, frame

    def get_next_frame(self) -> Optional[np.ndarray]:
        """
        Reads and returns the next frame in the video sequence.

        Returns:
            Optional[np.ndarray]: The next frame image as a NumPy array, or None if
                               the end of the video is reached or video is not open.
        """
        if not self.cap or not self.cap.isOpened():
            return False, None

        next_index = self.current_frame_index + 1
        if next_index < self.total_frames:
            self.current_frame_index = self.current_frame_index + 1
            ret, frame = self.get_current_frame()
            if ret:
                return ret, frame
        print("Info: Reached end of video.")
        return False, None

    def get_previous_frame(self) -> Optional[np.ndarray]:
        """
        Reads and returns the previous frame in the video sequence.

        Returns:
            Optional[np.ndarray]: The previous frame image as a NumPy array, or None if
                               the beginning of the video is reached or video is not open.
        """
        if not self.cap or not self.cap.isOpened():
            return False, None

        prev_index = self.current_frame_index - 1
        if prev_index >= 0:
            self.current_frame_index = self.current_frame_index - 1
            ret, frame = self.get_current_frame()
            if ret:
                return ret, frame
        print("Info: Reached beginning of video.")
        return False, None

    def jump_to_frame(self, frame_index: int) -> Optional[np.ndarray]:
        """
        Jumps to a specific frame by its index and returns it.

        Args:
            frame_index (int): The 0-based index of the frame to jump to.

        Returns:
            Optional[np.ndarray]: The frame image as a NumPy array if successful,
                               None if the index is out of bounds or video is not open.
        """
        if not self.cap or not self.cap.isOpened():
            return False, None

        if 0 <= frame_index < self.total_frames:
            self.current_frame_index = frame_index
            ret, frame = self._read_frame_at_index(self.current_frame_index)
            if ret:
                return True, frame
        print(
            f"Error: Frame index {frame_index} is out of bounds (0 to {self.total_frames - 1})."
        )
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

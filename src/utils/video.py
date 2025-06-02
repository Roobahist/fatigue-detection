import logging
import os
from typing import Dict, List, Optional, Tuple, Union

import cv2
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm


def get_video_info(
    cap: Optional[cv2.VideoCapture] = None, file_path: Optional[str] = None
) -> Tuple[int, int, int, int, str]:
    """
    Retrieves video information from a cv2.VideoCapture object.

    Args:
        cap: The cv2.VideoCapture object.
        file_path: The path to the video file.

        Raises:
        ValueError: If both `cap` and `file_path` are provided, or if neither is provided.

    Returns:
        A tuple containing (fps, width, height, total_frames, fourcc) as integers.
    """

    if (cap is None and not file_path) or (cap is not None and file_path):
        raise ValueError(
            "Provide either `cap` or `file_path`, but not both or neither."
        )
    if cap is None:
        cap = cv2.VideoCapture(file_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))

    if file_path is not None:
        cap.release()

    return fps, width, height, total_frames, fourcc


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


def save_video(
    self,
    frames: List[np.ndarray],
    output_path: str,
    fps: float,
    start_frame: int,
    end_frame: int,
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
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
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


class VideoReader:
    """
    A class for reading video files, providing methods to navigate through frames
    and retrieve video information.
    """

    def __init__(
        self,
        video_path: Optional[str] = None,
        cap_object: Optional[cv2.VideoCapture] = None,
    ):
        """
        Initializes the VideoReader with either a path to a video file or an
        existing cv2.VideoCapture object.

        Args:
            video_path (str, optional): The file path to the video. Required if
                                        cap_object is not provided.
            cap_object (cv2.VideoCapture, optional): An existing cv2.VideoCapture
                                                     object. If provided, video_path
                                                     will be ignored.
        """
        self.video_path = video_path
        self.cap = None

        if cap_object is not None and isinstance(cap_object, cv2.VideoCapture):
            self.cap = cap_object
        elif video_path is not None:
            self.cap = cv2.VideoCapture(video_path)
        else:
            print("Error: Either video_path or cap_object must be provided.")
            self.total_frames = 0
            self.fps = 0
            self.width = 0
            self.height = 0
            self.current_frame_index = -1  # Indicate no frame is loaded
            return

        if (
            not self.cap or not self.cap.isOpened()
        ):  # Check self.cap before calling isOpened()
            if video_path:
                print(f"Error: Could not open video file at {video_path}")
            else:
                print("Error: Provided cv2.VideoCapture object is not opened or valid.")
            self.total_frames = 0
            self.fps = 0
            self.width = 0
            self.height = 0
            self.current_frame_index = -1  # Indicate no frame is loaded
            return

        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.current_frame_index = -1  # Start before the first frame


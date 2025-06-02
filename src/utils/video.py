import logging
import os
from typing import Dict, List, Optional, Tuple, Union

import cv2
import matplotlib.pyplot as plt
import numpy as np
import logging
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

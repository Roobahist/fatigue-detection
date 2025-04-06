import cv2
from typing import Tuple

def get_video_info(cap: cv2.VideoCapture) -> Tuple[int, int, int, int]:
    """
    Retrieves video information from a cv2.VideoCapture object.

    Args:
        cap: The cv2.VideoCapture object.

    Returns:
        A tuple containing (fps, width, height, total_frames) as integers.
    """
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    return fps, width, height, total_frames
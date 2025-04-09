import cv2
from typing import Tuple, Optional

def get_video_info(cap: Optional[cv2.VideoCapture] = None,
                   file_path: Optional[str] = None) -> Tuple[int, int, int, int]:
    """
    Retrieves video information from a cv2.VideoCapture object.

    Args:
        cap: The cv2.VideoCapture object.
        file_path: The path to the video file.

        Raises:
        ValueError: If both `cap` and `file_path` are provided, or if neither is provided.

    Returns:
        A tuple containing (fps, width, height, total_frames) as integers.
    """

    if (cap is None and not file_path) or (cap is not None and file_path):
        raise ValueError("Provide either `cap` or `file_path`, but not both or neither.")
    if cap is None:
        cap = cv2.VideoCapture(file_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if file_path is not None: cap.release()

    return fps, width, height, total_frames
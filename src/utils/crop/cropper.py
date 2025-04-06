import cv2
from typing import Tuple, Optional
import numpy as np

from Zh.src.utils.crop.indices import FACE_OVAL


def get_face_bbox(detection_result, margin: float = 0.1) -> Tuple[float, float, float, float]:
    """
    Calculates the bounding box of a face based on facial landmarks.

    Args:
        detection_result: The face detection result containing facial landmarks.
        margin: The margin to add around the bounding box as a fraction of the face's width/height.

    Returns:
        A tuple (x_min, x_max, y_min, y_max) representing the bounding box coordinates (normalized).
    """
    face_oval_landmarks_x, face_oval_landmarks_y = zip(
        *(
            (detection_result.face_landmarks[0][i].x, detection_result.face_landmarks[0][i].y)
            for i in FACE_OVAL
        )
    )

    x_min = min(face_oval_landmarks_x)
    x_max = max(face_oval_landmarks_x)
    y_min = min(face_oval_landmarks_y)
    y_max = max(face_oval_landmarks_y)

    face_height = y_max - y_min  # Corrected calculation
    face_width = x_max - x_min # Corrected calculation

    x_min = max(x_min - margin * face_width, 0)
    x_max = min(x_max + margin * face_width, 1)
    y_min = max(y_min - margin * face_height, 0)
    y_max = min(y_max + margin * face_height, 1)

    return x_min, x_max, y_min, y_max


def crop_face_from_frame(
    frame: np.ndarray,
    detection_result,
    width: int,
    height: int,
    margin: float = 0.1,
) -> Optional[np.ndarray]:
    """
    Crops a square face region from a frame based on face detection results.

    Args:
        frame: The input frame as a NumPy array.
        detection_result: The face detection result containing facial landmarks.
        width: The width of the input frame.
        height: The height of the input frame.
        margin: The margin to add around the bounding box as a fraction of the face's width/height.

    Returns:
        The cropped square face region as a NumPy array, or None if cropping fails.
    """
    try:
        x_min, x_max, y_min, y_max = get_face_bbox(detection_result, margin)
        x_min = int(x_min * width)
        x_max = int(x_max * width)
        y_min = int(y_min * height)
        y_max = int(y_max * height)

        face = frame[y_min:y_max, x_min:x_max]
        new_width = x_max - x_min
        new_height = y_max - y_min
        new_size = max(new_width, new_height)

        square_face = cv2.resize(
            face, (new_size, new_size), interpolation=cv2.INTER_LINEAR
        )

        return square_face

    except Exception as e:
        print(f"Error cropping face: {e}")
        return None
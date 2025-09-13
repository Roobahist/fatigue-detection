import cv2
from typing import Tuple, Optional, Union
import mediapipe as mp
import numpy as np
import logging
import traceback
from Zh.src.utils.face_indices import *

def mask_frame(frame: np.ndarray, detection_result, width: int, height: int) -> np.ndarray:
    """
    Masks the face region using both FACE_OVAL and HORIZONTAL_CENTER landmarks to handle face rotation.

    Args:
        frame: The input frame as a NumPy array.
        detection_result: The face detection result containing facial landmarks.
        width: The width of the input frame.
        height: The height of the input frame.

    Returns:
        The masked frame as a NumPy array.
    """
    # Get all landmarks needed for the mask
    all_landmarks = list(set(FACE_OVAL + HORIZONTAL_CENTER))  # Remove duplicates
    
    # Convert landmarks to pixel coordinates
    points = np.array([
        (
            int(detection_result.face_landmarks[0][i].x * width),
            int(detection_result.face_landmarks[0][i].y * height)
        )
        for i in all_landmarks
    ], dtype=np.int32)
    
    # Find the convex hull of all points to create a complete face mask
    hull = cv2.convexHull(points)
    
    # Create a mask of zeros (black)
    mask_img = np.zeros(frame.shape[:2], dtype=np.uint8)
    
    # Fill the convex hull with white (255)
    cv2.fillConvexPoly(mask_img, hull, 255)
    
    # Apply the mask to the frame
    masked_frame = cv2.bitwise_and(frame, frame, mask=mask_img)

    return masked_frame

def align_frame(frame: np.ndarray, detection_result, width: int, height: int) -> Tuple[np.ndarray, Optional[object]]:
    """
    Aligns the face in the frame based on eye positions.

    Args:
        frame: The input frame as a NumPy array.
        detection_result: The face detection result containing facial landmarks.
        width: The width of the input frame.
        height: The height of the input frame.

    Returns:
        The aligned frame as a NumPy array and detection_result with updated landmarks.
        """
    try:
        # --- validate inputs ---
        if detection_result is None or not getattr(detection_result, "face_landmarks", None):
            return frame
        lms = detection_result.face_landmarks[0]

        # --- collect eye landmarks (pixel coords) ---
        left_eye = np.array([(int(lms[i].x * width), int(lms[i].y * height)) for i in LEFT_EYE], dtype=np.int32)
        right_eye = np.array([(int(lms[i].x * width), int(lms[i].y * height)) for i in RIGHT_EYE], dtype=np.int32)

        # --- centers ---
        left_eye_center  = left_eye.mean(axis=0)   # float64
        right_eye_center = right_eye.mean(axis=0)  # float64

        # --- enforce left->right ordering in image coordinates ---
        # If needed, swap to guarantee dx > 0
        if right_eye_center[0] < left_eye_center[0]:
            left_eye_center, right_eye_center = right_eye_center, left_eye_center

        # --- angle (small tilt) ---
        dy = float(right_eye_center[1] - left_eye_center[1])
        dx = float(right_eye_center[0] - left_eye_center[0])
        angle = np.degrees(np.arctan2(dy, dx))  # small angle around 0 after enforcing dx>0

        # --- eyes center (use plain Python floats) ---
        cx = float((left_eye_center[0] + right_eye_center[0]) / 2.0)
        cy = float((left_eye_center[1] + right_eye_center[1]) / 2.0)

        # --- rotation ---
        M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
        rotated_frame = cv2.warpAffine(frame, M, (int(width), int(height)), flags=cv2.INTER_CUBIC)
        
        # --- Update all landmarks for the first face using the same rotation matrix ---
        # todo: other objects in the detection_result object should also be updated
        for landmark in lms:
            # Convert normalized to pixel coordinates
            x_pixel = landmark.x * width
            y_pixel = landmark.y * height
            
            # Apply rotation matrix (add homogeneous coordinate for transformation)
            rotated_point = M @ np.array([x_pixel, y_pixel, 1])
            
            # Convert back to normalized coordinates and update
            landmark.x = rotated_point[0] / width
            landmark.y = rotated_point[1] / height

        return rotated_frame, detection_result

    except Exception as e:
        logging.error(f"Error in align_face: {e}")
        traceback.print_exc()
        return frame, detection_result

def _get_face_bbox(detection_result, margin: float = 0.05) -> Tuple[float, float, float, float]:
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
            for i in list(set(FACE_OVAL + HORIZONTAL_CENTER))
        )
    )

    x_min = min(face_oval_landmarks_x)
    x_max = max(face_oval_landmarks_x)
    y_min = min(face_oval_landmarks_y)
    y_max = max(face_oval_landmarks_y)

    face_height = y_max - y_min 
    face_width = x_max - x_min

    x_min = max(x_min - margin * face_width, 0)
    x_max = min(x_max + margin * face_width, 1)
    y_min = max(y_min - margin * face_width, 0)
    y_max = min(y_max + margin * face_width, 1)

    return x_min, x_max, y_min, y_max

def crop_frame(frame: np.ndarray, detection_result, width: int, height: int, margin: float = 0.1) -> np.ndarray:
    x_min, x_max, y_min, y_max = _get_face_bbox(detection_result, margin)
    frame = frame[int(y_min * height):int(y_max * height), 
                  int(x_min * width):int(x_max * width)]
    new_width = int((x_max - x_min) * width)
    new_height = int((y_max - y_min) * height)

    return frame, new_width, new_height

def process_face_in_frame(
    frame: np.ndarray,
    detection_result,
    mask: bool = True,
    align: bool = True,
    crop: bool = True,
    margin: float = 0.05,
    output_size: Union[Tuple[int, int], str] = (224, 224), #could be set to "original" or "original_square"
) -> Optional[np.ndarray]:
    """
    Crops a square face region from a frame based on face detection results.

    Args:
        frame: The input frame as a NumPy array.
        detection_result: The face detection result containing facial landmarks.
        margin: The margin to add around the bounding box as a fraction of the face's width/height.
        mask: Whether to apply a mask to the face region.
        align: Whether to align the face based on eye positions.
        output_size: The desired output size of the cropped face (width, height).

    Returns:
        The cropped square face region as a NumPy array, or None if cropping fails.
    """
    if detection_result == mp.tasks.vision.FaceLandmarkerResult(face_landmarks=[],
                                                                face_blendshapes=[],
                                                                facial_transformation_matrixes=[])\
    or detection_result is None:
        return None

    height, width = frame.shape[:2]

    if mask:
        frame = mask_frame(frame, detection_result, width, height)

    if align:
        frame, detection_result = align_frame(frame, detection_result, width, height)

    if crop:
        frame, width, height = crop_frame(frame, detection_result, width, height, margin)

    if output_size == "original_square":
        output_size = (max(width, height), max(width, height))
    elif output_size == "original":
        output_size = (width, height)

    square_face = cv2.resize(
        frame, output_size, interpolation=cv2.INTER_LINEAR
    )

    return square_face
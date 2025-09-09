import cv2
from typing import Tuple, Optional
import mediapipe as mp
import numpy as np
import logging
import traceback

#reference_image = https://github.com/google-ai-edge/mediapipe/blob/e0eef9791ebb84825197b49e09132d3643564ee2/mediapipe/modules/face_geometry/data/canonical_face_model_uv_visualization.png
#reference_for_indices = https://gist.github.com/Asadullah-Dal17/fd71c31bac74ee84e6a31af50fa62961

# face bounder indices 
FACE_OVAL=[ 10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103,67, 109]

# lips indices for Landmarks
LIPS=[ 61, 146, 91, 181, 84, 17, 314, 405, 321, 375,291, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95,185, 40, 39, 37,0 ,267 ,269 ,270 ,409, 415, 310, 311, 312, 13, 82, 81, 42, 183, 78 ]
LOWER_LIPS =[61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95]
UPPER_LIPS=[ 185, 40, 39, 37,0 ,267 ,269 ,270 ,409, 415, 310, 311, 312, 13, 82, 81, 42, 183, 78] 

# Left eyes indices 
LEFT_EYE =[ 362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385,384, 398 ]
LEFT_EYEBROW =[ 336, 296, 334, 293, 300, 276, 283, 282, 295, 285 ]

# right eyes indices
RIGHT_EYE=[ 33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161 , 246 ]  
RIGHT_EYEBROW=[ 70, 63, 105, 66, 107, 55, 65, 52, 53, 46 ]

HORIZONTAL_CENTER = [1, 6, 197, 0, 17, 2, 152, 199, 15, 168, 94]

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

def align_face(frame: np.ndarray, detection_result, width: int, height: int) -> np.ndarray: 
    """
    Aligns the face in the frame based on eye positions.

    Args:
        frame: The input frame as a NumPy array.
        detection_result: The face detection result containing facial landmarks.
        width: The width of the input frame.
        height: The height of the input frame.
        x_min: The minimum x-coordinate of the face bounding box.
        x_max: The maximum x-coordinate of the face bounding box.
        y_min: The minimum y-coordinate of the face bounding box.
        y_max: The maximum y-coordinate of the face bounding box.

    Returns:
        The aligned frame as a NumPy array.
    """
    try:
        left_eye = np.array([
            (
                int(detection_result.face_landmarks[0][i].x * width),
                int(detection_result.face_landmarks[0][i].y * height)
            )
            for i in LEFT_EYE
        ])
        right_eye = np.array([
            (
                int(detection_result.face_landmarks[0][i].x * width),
                int(detection_result.face_landmarks[0][i].y * height)
            )
            for i in RIGHT_EYE
        ])

        left_eye_center = left_eye.mean(axis=0).astype("int")
        right_eye_center = right_eye.mean(axis=0).astype("int")

        # Calculate the angle between the eye centers
        dy = right_eye_center[1] - left_eye_center[1]
        dx = right_eye_center[0] - left_eye_center[0]
        angle = np.degrees(np.arctan2(dy, dx))  # Corrected to use arctan2

        # Calculate the center of the eyes
        eyes_center = ((left_eye_center[0] + right_eye_center[0]) // 2,
                       (left_eye_center[1] + right_eye_center[1]) // 2)

        # Get the rotation matrix
        M = cv2.getRotationMatrix2D(eyes_center, angle, 1)

        # Rotate the entire frame
        rotated_frame = cv2.warpAffine(frame, M, (width, height), flags=cv2.INTER_CUBIC)

        return rotated_frame
    except Exception as e:
        logging.error(f"Error in align_face: {e}")
        traceback.print_exc()
        return frame  # Return original frame if alignment fails

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
    mask: bool = False,
    align: bool = False,
    output_size: Tuple[int, int] = (224, 224),
) -> Optional[np.ndarray]:
    """
    Crops a square face region from a frame based on face detection results.

    Args:
        frame: The input frame as a NumPy array.
        detection_result: The face detection result containing facial landmarks.
        width: The width of the input frame.
        height: The height of the input frame.
        margin: The margin to add around the bounding box as a fraction of the face's width/height.
        align: Whether to align the face based on eye positions.
        output_size: The desired output size of the cropped face (width, height).

    Returns:
        The cropped square face region as a NumPy array, or None if cropping fails.
    """
    if detection_result == mp.tasks.vision.FaceLandmarkerResult(face_landmarks=[],
                                                                face_blendshapes=[],
                                                                facial_transformation_matrixes=[]):
        return None
    
    x_min, x_max, y_min, y_max = get_face_bbox(detection_result, margin)
    x_min = int(x_min * width)
    x_max = int(x_max * width)
    y_min = int(y_min * height)
    y_max = int(y_max * height)

    if mask : 
        frame = mask_frame(frame, detection_result, width, height)

    if align:
        frame = align_face(frame, detection_result, width, height)

    face = frame[y_min:y_max, x_min:x_max]
    new_width = x_max - x_min
    new_height = y_max - y_min
    new_size = output_size or (max(new_width, new_height), max(new_width, new_height))

    square_face = cv2.resize(
        face, new_size, interpolation=cv2.INTER_LINEAR
    )

    return square_face
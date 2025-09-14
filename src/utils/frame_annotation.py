import cv2
import matplotlib.pyplot as plt
import numpy as np


def compute_face_regions(
    frame,
    detection_result,
    landmark_groups=None,
    margin_h=0.0,
    margin_w=0.0,
):
    """
    Compute landmark points or bounding boxes from face landmarks.

    Parameters
    ----------
    frame : np.ndarray
        Input frame.
    detection_result : FaceLandmarkerResult
        Detection result containing face landmarks.
    landmark_groups : list[list[int]] or None
        Groups of landmark indices. If None, use all landmarks.
    margin_h : float
        Extra margin (proportion of face height).
    margin_w : float
        Extra margin (proportion of face width).

    Returns
    -------
    regions : dict
        {
            "points": list of (x,y) landmark coords,
            "boxes": list of (x, y, w, h) bounding boxes
        }
    """
    height, width = frame.shape[:2]
    regions = {"points": [], "boxes": []}

    if not hasattr(detection_result, "face_landmarks"):
        return regions

    for face_landmarks in detection_result.face_landmarks:
        xs = [int(lm.x * width) for lm in face_landmarks]
        ys = [int(lm.y * height) for lm in face_landmarks]
        face_w = max(xs) - min(xs)
        face_h = max(ys) - min(ys)

        # default: use all landmarks
        if landmark_groups is None:
            landmark_groups = [list(range(len(face_landmarks)))]

        for group in landmark_groups:
            pts = np.array(
                [
                    [
                        int(face_landmarks[i].x * width),
                        int(face_landmarks[i].y * height),
                    ]
                    for i in group
                    if i < len(face_landmarks)
                ],
                dtype=np.int32,
            )
            if pts.shape[0] < 1:
                continue

            regions["points"].extend(pts.tolist())

            if pts.shape[0] >= 3:
                hull = cv2.convexHull(pts)
                x, y, w, h = cv2.boundingRect(hull)

                # add margins
                dw = int(margin_w * face_w)
                dh = int(margin_h * face_h)
                x = max(x - dw // 2, 0)
                y = max(y - dh // 2, 0)
                w = min(w + dw, width - x)
                h = min(h + dh, height - y)

                regions["boxes"].append((x, y, w, h))

    return regions


def draw_regions(
    frame,
    regions,
    mode="points",
    color=(0, 255, 0),
    radius=2,
    thickness=-1,
):
    """
    Draw or mask regions on a frame.

    Parameters
    ----------
    frame : np.ndarray
        Input frame.
    regions : dict
        Output from `compute_face_regions`.
    mode : {"points", "boxes", "mask"}
        Drawing mode.
    color : tuple[int]
        BGR color.
    radius : int
        Circle radius for points.
    thickness : int
        Thickness for points/boxes. -1 = filled.

    Returns
    -------
    annotated_frame : np.ndarray
    """
    annotated_frame = frame.copy()

    if mode == "points":
        for x, y in regions["points"]:
            cv2.circle(annotated_frame, (x, y), radius, color, thickness)

    elif mode == "boxes":
        for x, y, w, h in regions["boxes"]:
            cv2.rectangle(annotated_frame, (x, y), (x + w, y + h), color, 2)

    elif mode == "mask":
        mask = np.zeros_like(annotated_frame)
        for x, y, w, h in regions["boxes"]:
            mask[y : y + h, x : x + w] = annotated_frame[y : y + h, x : x + w]
        annotated_frame = mask

    return annotated_frame


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

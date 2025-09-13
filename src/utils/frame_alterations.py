import cv2
import numpy as np

def add_landmarks_to_frame(
    frame,
    detection_result,
    landmark_groups=None,
    mode="points",
    color=(0, 255, 0),
    radius=2,
    thickness=-1,
    margin_h=0.0,
    margin_w=0.0,
):
    """
    Draws or masks face landmarks on a single video frame.

    Parameters
    ----------
    frame : np.ndarray
        The video frame as a NumPy array.
    detection_result : FaceLandmarkerResult
        The detection result containing face landmarks.
    landmark_groups : list[list[int]] or None
        - For mode="points": list of landmark indices (single list).
        - For mode="boxes"/"mask": list of lists, each defining a region.
        - If None: all landmarks are used.
    mode : {"points", "boxes", "mask"}
        Drawing mode:
        - "points": draw landmarks as small circles.
        - "boxes": draw bounding boxes around convex hulls of landmark groups.
        - "mask": everything outside bounding boxes is blacked out.
    color : tuple[int]
        BGR color for drawing.
    radius : int
        Radius of each landmark circle (for "points").
    thickness : int
        Thickness for landmarks/boxes. -1 means filled circle.
    margin_h : float
        Extra margin (proportion of face height) to add to bounding box height.
    margin_w : float
        Extra margin (proportion of face width) to add to bounding box width.

    Returns
    -------
    annotated_frame : np.ndarray
        The frame with landmarks drawn according to the selected mode.
    """
    annotated_frame = frame.copy()
    height, width = annotated_frame.shape[:2]

    if not hasattr(detection_result, "face_landmarks"):
        return annotated_frame

    for face_landmarks in detection_result.face_landmarks:
        # Face dimensions (approx: full bounding box of landmarks)
        xs = [int(lm.x * width) for lm in face_landmarks]
        ys = [int(lm.y * height) for lm in face_landmarks]
        face_w = max(xs) - min(xs)
        face_h = max(ys) - min(ys)

        if mode == "points":
            indices = (
                range(len(face_landmarks))
                if landmark_groups is None
                else landmark_groups
            )
            for i in indices:
                if i < len(face_landmarks):
                    lm = face_landmarks[i]
                    x, y = int(lm.x * width), int(lm.y * height)
                    cv2.circle(annotated_frame, (x, y), radius, color, thickness)

        elif mode in {"boxes", "mask"}:
            if landmark_groups is None:
                landmark_groups = [list(range(len(face_landmarks)))]

            mask = np.zeros_like(annotated_frame) if mode == "mask" else None

            for group in landmark_groups:
                pts = np.array(
                    [[int(face_landmarks[i].x * width), int(face_landmarks[i].y * height)]
                     for i in group if i < len(face_landmarks)] ,
                    dtype=np.int32,
                )
                if pts.shape[0] < 3:
                    continue  # need at least 3 points for convex hull

                hull = cv2.convexHull(pts)
                x, y, w, h = cv2.boundingRect(hull)

                # add margins
                dw = int(margin_w * face_w)
                dh = int(margin_h * face_h)
                x = max(x - dw // 2, 0)
                y = max(y - dh // 2, 0)
                w = min(w + dw, width - x)
                h = min(h + dh, height - y)

                if mode == "boxes":
                    cv2.rectangle(annotated_frame, (x, y), (x + w, y + h), color, 2)
                elif mode == "mask":
                    mask[y:y+h, x:x+w] = annotated_frame[y:y+h, x:x+w]

            if mode == "mask":
                annotated_frame = mask

    return annotated_frame
import cv2
import matplotlib.pyplot as plt
import numpy as np


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

    # Define a list of distinctive colors (BGR)
    color_palette = [
        (255, 0, 0),    # Blue
        (0, 255, 0),    # Green
        (0, 0, 255),    # Red
        (255, 255, 0),  # Cyan
        (255, 0, 255),  # Magenta
        (0, 255, 255),  # Yellow
        (128, 0, 128),  # Purple
        (0, 128, 128),  # Teal
        (128, 128, 0),  # Olive
        (0, 0, 128),    # Maroon
    ]

    if mode == "points":
        for x, y in regions["points"]:
            cv2.circle(annotated_frame, (x, y), radius, color, thickness)

    elif mode == "boxes":
        for idx, (x, y, w, h) in enumerate(regions["boxes"]):
            group_color = color_palette[idx % len(color_palette)]
            cv2.rectangle(annotated_frame, (x, y), (x + w, y + h), group_color, 2)

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

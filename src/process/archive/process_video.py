# List of required packages
required_packages = [
    "opencv-python", 
    "mediapipe", 
    "pandas", 
    "numpy", 
    "tqdm", 
    "multiprocessing", 
    "logging", 
    "scikit-learn", 
    "mord", 
    "tensorflow<2.16.0", 
    "matplotlib", 
    "tensorflow-addons", 
    "imbalanced-learn"
]

import subprocess
import sys

# Function to install packages if not already installed
def install_package(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

# Check and install missing packages
for package in required_packages:
    try:
        __import__(package.split("<")[0])  # Handle version constraints
    except ImportError:
        print(f"{package} not found, installing...")
        install_package(package)

# # Ensure compatibility with TensorFlow Addons and Keras
# try:
#     import tensorflow as tf
#     if tf.__version__ >= "2.16.0":
#         print("Downgrading TensorFlow for compatibility with TensorFlow Addons...")
#         install_package("tensorflow<2.16.0")
# except ImportError:
#     install_package("tensorflow<2.16.0")

# try:
#     import tensorflow_addons as tfa
# except ImportError:
#     print("Installing TensorFlow Addons...")
#     install_package("tensorflow-addons")

# try:
#     import keras
# except ImportError:
#     print("Installing Keras...")
#     install_package("keras")

# Import necessary libraries
import os
import cv2
import mediapipe as mp
import pandas as pd
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool
import logging
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from mord import LogisticAT  # Ordinal logistic regression
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import tensorflow_addons as tfa
from keras import backend as K
from keras import losses
from imblearn.over_sampling import SMOTE, SMOTENC
from tensorflow.keras.layers import Lambda
from sklearn.utils.class_weight import compute_class_weight

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("process_video.log"),
        logging.StreamHandler()
    ]
)

# Initialize MediaPipe FaceLandmarker
def initialize_landmarker():
    base_options = mp.tasks.BaseOptions(model_asset_path='/home/mojtaba.kananisarche/Zh/UTA/Process/face_landmarker_v2_with_blendshapes.task')
    options = mp.tasks.vision.FaceLandmarkerOptions(
        base_options=base_options,
        output_face_blendshapes=True,
        output_facial_transformation_matrixes=False,
        num_faces=1
    )
    return mp.tasks.vision.FaceLandmarker.create_from_options(options)

def draw_landmarks_on_frame(frame, detection_result):
    """
    Draws face landmarks on a single video frame.

    Parameters:
    - frame: numpy.ndarray
        The video frame as a NumPy array.
    - detection_result: FaceLandmarkerResult
        The detection result containing face landmarks.

    Returns:
    - annotated_frame: numpy.ndarray
        The frame with face landmarks drawn.
    """
    annotated_frame = frame.copy()
    for face_landmarks in detection_result.face_landmarks:
        for landmark in face_landmarks:
            x = int(landmark.x * annotated_frame.shape[1])
            y = int(landmark.y * annotated_frame.shape[0])
            cv2.circle(annotated_frame, (x, y), radius=2, color=(0, 255, 0), thickness=-1)
    return annotated_frame

def process_video(video_path, save=False, start_second=0, end_second=None, use_tqdm=True):
    """
    Processes a video, detects landmarks in each frame within the specified range, 
    and optionally saves the annotated video.

    Parameters:
    - video_path: str
        Path to the input video.
    - save: bool or str
        Whether to save the annotated video. If a string, it specifies the output path.
    - start_second: int
        Start time in seconds for processing the video.
    - end_second: int or None
        End time in seconds for processing the video. If None, processes until the end of the video.
    - use_tqdm: bool
        Whether to use a progress bar.

    Returns:
    - blendshape_results: list
        A list of detected blendshape scores for each frame.
    """
    detector = initialize_landmarker()
    
    logging.info(f"Starting processing for video: {video_path}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logging.warning(f"Failed to open video: {video_path}")
        return []

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    start_frame = int(start_second * fps)
    end_frame = int(end_second * fps) if end_second is not None else total_frames

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    blendshape_results = []
    frames = []
    detection_results = []
    

    with tqdm(total=end_frame - start_frame, desc=f"Processing {os.path.basename(video_path)}", unit="frame", disable=not use_tqdm) as progress_bar:
        while cap.isOpened() and cap.get(cv2.CAP_PROP_POS_FRAMES) < end_frame:
            ret, frame = cap.read()
            
            if cap.get(cv2.CAP_PROP_POS_FRAMES) % 1000 == 0:
                logging.info(f"frame number {cap.get(cv2.CAP_PROP_POS_FRAMES)} in video: {video_path}")
                
            if not ret:
                logging.warning(f"Failed to read frame at position {cap.get(cv2.CAP_PROP_POS_FRAMES)} in video: {video_path}")
                break
            
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
            detection_result = detector.detect(mp_image)
            detection_results.append(detection_result)
     
            frame_blendshapes = [
                {blendshape.category_name: blendshape.score for blendshape in face_blendshapes}
                for face_blendshapes in detection_result.face_blendshapes
            ]
            blendshape_results.append(frame_blendshapes)
            frames.append(frame)

            progress_bar.update(1)

    cap.release()
    logging.info(f"Completed processing for video: {video_path}")

    if save:
        output_path = save if isinstance(save, str) else f"{os.path.splitext(video_path)[0]}_annotated.mp4"
        logging.info(f"Saving annotated video to: {output_path}")

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        for frame, detection_result in zip(frames, detection_results):
            annotated_frame = draw_landmarks_on_frame(frame, detection_result)
            out.write(annotated_frame)

        out.release()
        logging.info(f"Annotated video saved at: {output_path}")

    return blendshape_results

# Wrapper function for processing a single video
def process_video_wrapper(args):
    return process_single_video(*args)

def process_single_video(video_file, input_folder, output_folder, save, start_second, end_second, use_tqdm):
    input_path = os.path.join(input_folder, video_file)
    output_csv = os.path.join(output_folder, os.path.splitext(video_file)[0] + "_blendshape.csv")

    if os.path.exists(output_csv):
        logging.info(f"Skipping already processed video: {input_path}")
        return

    try:
        blendshape_results = process_video(input_path, save, start_second, end_second, use_tqdm)
        keys = next((frame[0].keys() for frame in blendshape_results if frame), [])

        pd.DataFrame([
            {key: round(value, 2) if isinstance(value, float) else value for key, value in frame[0].items()} if frame else {key: 0 for key in keys}
            for frame in blendshape_results
        ]).to_csv(output_csv, index=False)

        logging.info(f"Saved blendshape results to: {output_csv}")
    except Exception as e:
        logging.error(f"Error processing {video_file}: {e}")

def process_videos_and_save_parallel(input_folder, output_folder, save=False, start_second=0, end_second=None, use_tqdm=False):
    """
    Process all videos in a folder, create a DataFrame from blendshape_results, and save as CSV.
    """
    os.makedirs(output_folder, exist_ok=True)
    video_files = [f for f in os.listdir(input_folder) if f.lower().endswith(('.mp4', '.mov', '.avi', '.MOV'))]

    args_list = [(video_file, input_folder, output_folder, save, start_second, end_second, use_tqdm) for video_file in video_files]
    logging.info(args_list)
    
#     for args in args_list:
#         process_video_wrapper(args)
    with ProcessPoolExecutor(6) as executor:
        executor.map(process_video_wrapper, args_list)
    
    
if __name__ == "__main__":
    input_folder = "/work/long_lab/Mojtaba/data/DROZY/DROZY/videos_30fps/"
    output_folder = "/work/long_lab/Mojtaba/data/DROZY/DROZY/videos_30fps/blendshapes/"
    process_videos_and_save_parallel(input_folder, output_folder, save=False, start_second=0, end_second=None, use_tqdm=True)

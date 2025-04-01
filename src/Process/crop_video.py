import os
import cv2
import mediapipe as mp
import pandas as pd
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool
import logging
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor


# Initialize MediaPipe FaceLandmarker
def initialize_landmarker():
    base_options = mp.tasks.BaseOptions(model_asset_path='/home/mojtaba.kananisarche/Zh/UTA/Process/face_landmarker_v2_with_blendshapes.task')
    options = mp.tasks.vision.FaceLandmarkerOptions(
        base_options=base_options,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
        num_faces=1
    )
    return mp.tasks.vision.FaceLandmarker.create_from_options(options)


FACE_OVAL=[10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103,67, 109]


def get_face_bbox(detection_result, margin=0.1):
    face_oval_landmarks_x, face_oval_landmarks_y = zip(*[(detection_result.face_landmarks[0][i].x, 
                                                          detection_result.face_landmarks[0][i].y) for i in FACE_OVAL])
    
    x_min = min(face_oval_landmarks_x)
    x_max = max(face_oval_landmarks_x)
    y_min = min(face_oval_landmarks_y)
    y_max = max(face_oval_landmarks_y)
    
    face_height = x_max - x_min
    face_width  = y_max - y_min
    
    x_min = max(x_min - margin*face_width,  0)
    x_max = min(x_max + margin*face_width,  1)
    y_min = max(y_min - margin*face_height, 0)
    y_max = min(y_max + margin*face_height, 1)
    
    return x_min, x_max, y_min, y_max


# Wrapper function for processing a single video
def process_video_wrapper(args):
    return process_single_video(*args)

def process_single_video(video_file, input_folder, output_folder, save=False, start_second=0, end_second=None, use_tqdm=False, indices=FACE_OVAL):
    input_path  = os.path.join(input_folder, video_file)
    output_path = os.path.join(output_folder, os.path.splitext(video_file)[0] + "_sq_cropped.mp4")

    if os.path.exists(output_path):
        logging.info(f"Skipping already processed video: {input_path}")
        return None

    try:
        detector = initialize_landmarker()

        logging.info(f"Starting processing for video: {input_path}")

        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            logging.warning(f"Failed to open video: {input_path}")
            return []

        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        start_frame = int(start_second * fps)
        end_frame = int(end_second * fps) if end_second is not None else total_frames

        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        frames = []

        temp_detection_result = None

        with tqdm(total=end_frame - start_frame, desc=f"Processing {os.path.basename(input_path)}", unit="frame", disable=not use_tqdm) as progress_bar:
            while cap.isOpened() and cap.get(cv2.CAP_PROP_POS_FRAMES) < end_frame:
                ret, frame = cap.read()

                if cap.get(cv2.CAP_PROP_POS_FRAMES) % 1000 == 0:
                    logging.info(f"frame number {cap.get(cv2.CAP_PROP_POS_FRAMES)} in video: {input_path}")

                if not ret:
                    logging.warning(f"Failed to read frame at position {cap.get(cv2.CAP_PROP_POS_FRAMES)} in video: {input_path}")
                    break

                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
                detection_result = detector.detect(mp_image)
                try:
                    x_min, x_max, y_min, y_max = get_face_bbox(detection_result)
                    temp_detection_result = detection_result
                except:
                    if temp_detection_result == None:
                        continue
                    x_min, x_max, y_min, y_max = get_face_bbox(temp_detection_result)


                x_min = int(x_min*width)
                x_max = int(x_max*width)
                y_min = int(y_min*height)
                y_max = int(y_max*height)

                # Crop the frame using the bounding box coordinates
                face = frame[y_min:y_max, x_min:x_max]

                new_width = x_max - x_min
                new_height = y_max - y_min
                new_size = max(new_width, new_height)

                # Create a square image with bilinear interpolation
                square_face = cv2.resize(face, (new_size, new_size), interpolation=cv2.INTER_LINEAR)

                frames.append(square_face)
                progress_bar.update(1)  

            cap.release()

        logging.info(f"Completed processing for video: {input_path}")


        if save:
            logging.info(f"Saving annotated video to: {output_path}")

            video_height_width = max([frame.shape[0] for frame in frames])
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (video_height_width, video_height_width))

            with tqdm(total=end_frame - start_frame, desc=f"Saving {os.path.basename(input_path)}", unit="frame", disable=not use_tqdm) as progress_bar:    
                for frame in frames:
                    frame_resized = cv2.resize(frame, (video_height_width, video_height_width), interpolation=cv2.INTER_LINEAR)
                    out.write(frame_resized)
                    progress_bar.update(1)
                out.release()


            logging.info(f"Cropped video saved at: {output_path}")

        return output_path

    except Exception as e:
        logging.error(f"Error processing {video_file}: {e}")
        return None
    
    
    
def process_videos_and_save_parallel(input_folder, output_folder, save=False, start_second=0, end_second=None, use_tqdm=False, indices=FACE_OVAL):
    """
    Process all videos in a folder, create a DataFrame from blendshape_results, and save as CSV.
    """
    os.makedirs(output_folder, exist_ok=True)
    video_files = [f for f in os.listdir(input_folder) if f.lower().endswith(('.mp4', '.mov', '.avi', '.MOV'))]

    args_list = [(video_file, input_folder, output_folder, save, start_second, end_second, use_tqdm, indices) for video_file in video_files]
    logging.info(args_list)
    
#     for args in args_list:
#         print(args)
#         process_video_wrapper(args)
    with ProcessPoolExecutor(10) as executor:
        executor.map(process_video_wrapper, args_list)
        
        
        
if __name__ == "__main__":
    input_folder = "/home/mojtaba.kananisarche/Zh/UTA/Video/Raw_30fps"
    output_folder = "/home/mojtaba.kananisarche/Zh/UTA/Video/Raw_30fps_Cropped"
    
    process_videos_and_save_parallel(input_folder  = input_folder,
                                     output_folder = output_folder, 
                                     save          = True, 
                                     start_second  = 0, 
                                     end_second    = None, 
                                     use_tqdm      = True, 
                                     indices       = FACE_OVAL)

    

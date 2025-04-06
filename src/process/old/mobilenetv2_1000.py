import logging

from tensorflow import keras
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
import cv2
import os

import pandas as pd
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor


# Wrapper function for processing a single video
def process_video_wrapper(args):
    return process_single_video(*args)


def process_single_video(video_file, 
                         input_folder, 
                         output_folder, 
                         model_input_shape=(224,224,3), 
                         alpha = 1.0,
                         include_top = True,
                         weights = "imagenet",
                         input_tensor = None,
                         pooling = None,
                         classes = 1000,
                         classifier_activation = None,
                         save=True, 
                         start_second=0, 
                         end_second=None, 
                         use_tqdm=True):
    
    
    mnv2 = MobileNetV2(input_shape=model_input_shape,
                       alpha=alpha,
                       include_top=include_top,
                       weights=weights,
                       input_tensor=input_tensor,
                       pooling=pooling,
                       classes=classes,
                       classifier_activation=classifier_activation)
    
    
    input_path  = os.path.join(input_folder, video_file)
    output_path = os.path.join(output_folder, os.path.splitext(video_file)[0] + "_mobilenetv2.csv")

    if os.path.exists(output_path):
        logging.warning(f"Skipping already processed video: {input_path}")
        return None

    try:
        logging.warning(f"Starting processing for video: {input_path}")

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

        feature_vectors = [] #only gets filled if save is false
        column_names = [f'f_{i}' for i in range(classes)]
        with tqdm(total=end_frame - start_frame, desc=f"Processing {os.path.basename(input_path)}", unit="frame", disable=not use_tqdm) as progress_bar:
            while cap.isOpened() and cap.get(cv2.CAP_PROP_POS_FRAMES) < end_frame:
                ret, frame = cap.read()

                if cap.get(cv2.CAP_PROP_POS_FRAMES) % 1000 == 0:
                    logging.warning(f"frame number {cap.get(cv2.CAP_PROP_POS_FRAMES)} in video: {input_path}")

                if not ret:
                    logging.warning(f"Failed to read frame at position {cap.get(cv2.CAP_PROP_POS_FRAMES)} in video: {input_path}")
                    break

                frame = cv2.resize(frame, 
                                   model_input_shape[:2], 
                                   interpolation=cv2.INTER_LINEAR)

                frame = np.expand_dims(frame, 
                                       axis=0)

                frame = preprocess_input(frame, 
                                         data_format="channels_last")

                feature_vector = mnv2.predict(frame, 
                                              verbose = 0)

                feature_vector_df = pd.DataFrame(np.round(feature_vector, 
                                                          3), 
                                                 columns=column_names)

                write_header = not os.path.exists(output_path)

                if save:
                    feature_vector_df.to_csv(output_path, 
                                             mode='a', 
                                             header=write_header, 
                                             index=False)
                else:
                    feature_vectors.append(feature_vector)

                progress_bar.update(1)  

            cap.release()

        logging.info(f"Completed processing for video: {input_path}")

        return output_path

    except Exception as e:
        logging.warning(f"Error processing {video_file}: {e}")
        return None
    
    
    
def process_videos_and_save_parallel(input_folder, 
                                     output_folder, 
                                     model_input_shape=(224,224,3), 
                                     alpha = 1.0,
                                     include_top = True,
                                     weights = "imagenet",
                                     input_tensor = None,
                                     pooling = None,
                                     classes = 1000,
                                     classifier_activation = None,
                                     save=True, 
                                     start_second=0, 
                                     end_second=None, 
                                     use_tqdm=True):
    """
    Process all videos in a folder, create a DataFrame from blendshape_results, and save as CSV.
    """
    os.makedirs(output_folder, exist_ok=True)
    video_files = [f for f in os.listdir(input_folder) if f.lower().endswith(('.mp4', '.mov', '.avi', '.MOV'))]

    args_list = [(video_file, 
                 input_folder, 
                 output_folder, 
                 model_input_shape, 
                 alpha,
                 include_top,
                 weights,
                 input_tensor,
                 pooling,
                 classes,
                 classifier_activation,
                 save, 
                 start_second, 
                 end_second, 
                 use_tqdm) for video_file in video_files]
    
    logging.warning(args_list)
    
#     for args in args_list[:1]:
#         print(args)
#         return process_video_wrapper(args)
    with ProcessPoolExecutor(30) as executor:
        executor.map(process_video_wrapper, args_list)


if __name__ == "__main__":
    logging.warning("main")
    input_folder = "/home/mojtaba.kananisarche/Zh/UTA/Video/Raw_30fps_Cropped"
    
#     output_folder = "/home/mojtaba.kananisarche/Zh/UTA/feature_vectors/mobilenetv2"
#     process_videos_and_save_parallel(input_folder=input_folder, 
#                                      output_folder=output_folder, 
#                                      model_input_shape=(224,224,3), 
#                                      alpha = 1.0,
#                                      include_top = True,
#                                      weights = "imagenet",
#                                      input_tensor = None,
#                                      pooling = None,
#                                      classes = 1000,
#                                      classifier_activation = None,
#                                      save=True, 
#                                      start_second=0, 
#                                      end_second=None, 
#                                      use_tqdm=True)
    
#     output_folder = "/home/mojtaba.kananisarche/Zh/UTA/feature_vectors/mobilenetv2_topremoved_maxpooling"
#     process_videos_and_save_parallel(input_folder=input_folder, 
#                                      output_folder=output_folder, 
#                                      model_input_shape=(224,224,3), 
#                                      alpha = 1.0,
#                                      include_top = False,
#                                      weights = "imagenet",
#                                      input_tensor = None,
#                                      pooling = 'max',
#                                      classes = 1280,
#                                      classifier_activation = None,
#                                      save=True, 
#                                      start_second=0, 
#                                      end_second=None, 
#                                      use_tqdm=True)
    
    
    output_folder = "/home/mojtaba.kananisarche/Zh/UTA/feature_vectors/mobilenetv2_topremoved_avgpooling"
    process_videos_and_save_parallel(input_folder=input_folder, 
                                     output_folder=output_folder, 
                                     model_input_shape=(224,224,3), 
                                     alpha = 1.0,
                                     include_top = False,
                                     weights = "imagenet",
                                     input_tensor = None,
                                     pooling = 'avg',
                                     classes = 1280,
                                     classifier_activation = None,
                                     save=True, 
                                     start_second=0, 
                                     end_second=None, 
                                     use_tqdm=True)
    
from __future__ import annotations

import os
import logging
import cv2
import traceback

from typing import List, Optional, Dict, Union
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from tqdm import tqdm

from Zh.src.utils.crop.cropper import crop_face_from_frame
from Zh.src.utils.face_detection.face_detector import FaceDetector
from Zh.src.utils.video import get_video_info

import mediapipe as mp
import numpy as np


class VideoProcessor:
    """
    A class to process video files, detecting and cropping faces, with support for multiple videos.
    """

    def __init__(self, input_folder: str, output_folder: str, use_tqdm: bool = False):
        """
        Initializes the VideoProcessor.

        Args:
            input_folder: The directory containing input video files.
            output_folder: The directory to save processed video files.
            use_tqdm: Whether to use tqdm progress bars.
        """
        self.input_folder = input_folder
        self.output_folder = output_folder
        self.use_tqdm = use_tqdm
        self.detector = FaceDetector(False, False, 1).landmarker

    def _augment_frame(self, frame: np.ndarray, augmentations: Dict[str, List[Dict]]) -> Dict[str, np.ndarray]:
        """
        Applies configurable augmentations to a single frame and returns a dictionary of augmented frames.

        Args:
            frame: The input frame as a NumPy array.
            augmentations: A dictionary specifying augmentations and their parameters.

        Returns:
            A dictionary where keys are augmentation names and values are augmented frames.
        """
        augmented_frames = {}
        original_frame = frame.copy()

        for aug_name, aug_configs in augmentations.items():
            for aug_params in aug_configs:
                augmented_frame = original_frame.copy()

                if aug_name == "gaussianblur":
                    kernel_size = aug_params.get("kernel_size", 3)
                    aug_key = f"{aug_name}_kernelsize{kernel_size}"
                    augmented_frame = cv2.GaussianBlur(augmented_frame, (kernel_size, kernel_size), 0)

                elif aug_name == "brightness":
                    alpha = aug_params.get("alpha", 1.1)
                    beta = aug_params.get("beta", 0)
                    aug_key = f"{aug_name}_alpha{alpha}_beta{beta}"
                    augmented_frame = np.clip(augmented_frame * alpha + beta, 0, 255).astype(np.uint8)

                elif aug_name == "contrast":
                    alpha = aug_params.get("alpha", 1.2)
                    beta = aug_params.get("beta", 0)
                    aug_key = f"{aug_name}_alpha{alpha}_beta{beta}"
                    augmented_frame = np.clip(augmented_frame * alpha + beta, 0, 255).astype(np.uint8)

                elif aug_name == "flip":
                    flip_code = aug_params.get("flip_code", 1)
                    aug_key = f"{aug_name}_flipcode{flip_code}"
                    augmented_frame = cv2.flip(augmented_frame, flip_code)

                elif aug_name == "grayscale":
                    aug_key = f"{aug_name}"
                    augmented_frame = cv2.cvtColor(augmented_frame, cv2.COLOR_BGR2GRAY)
                    augmented_frame = cv2.cvtColor(augmented_frame, cv2.COLOR_GRAY2BGR)

                elif aug_name == "rotate":
                    angle = aug_params.get("angle", 10)
                    aug_key = f"{aug_name}_angle{angle}"
                    height, width = augmented_frame.shape[:2]
                    matrix = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1)
                    augmented_frame = cv2.warpAffine(augmented_frame, matrix, (width, height))

                augmented_frames[aug_key] = augmented_frame

        return augmented_frames    
    
    def _process_frame(self, frame: cv2.Mat, width: int, height: int, temp_detection_result: Optional["mp.tasks.vision.FaceDetectorResult"] = []) -> Optional[cv2.Mat]:
        """
        Processes a single frame, detecting and cropping the face.

        Args:
            frame: The input frame.
            width: The width of the video frame.
            height: The height of the video frame.
            temp_detection_result: Previous detection result if available.

        Returns:
            The cropped face frame, or None if face detection fails.
        """
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        detection_result = self.detector.detect(mp_image)

        cropped_face = crop_face_from_frame(frame, detection_result, width, height, margin=0.1)
        if cropped_face is not None: return cropped_face, detection_result
        if temp_detection_result == []: return None, []
        cropped_face = crop_face_from_frame(frame, temp_detection_result, width, height, margin=0.1)
        return cropped_face, temp_detection_result
        
        
    def _save_video(self, frames: List[np.ndarray], output_path: str, fps: float, start_frame: int, end_frame: int) -> None:
        """
        Saves processed frames as a video.

        Args:
            frames: List of processed frames (NumPy arrays).
            output_path: Path to save the output video.
            fps: Frames per second of the video.
            start_frame: Start frame of the processing.
            end_frame: End frame of the processing.
        """
        if not frames:
            logging.warning(f"No frames to save for {output_path}. Skipping video save.")
            return

        first_frame_height, first_frame_width = frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(output_path, fourcc, fps, (first_frame_width, first_frame_height))

        with tqdm(total=end_frame - start_frame, desc=f"Saving {os.path.basename(output_path)}", unit="frame", disable=not self.use_tqdm) as progress_bar:
            for frame in frames:
                resized_frame = cv2.resize(frame, (first_frame_width, first_frame_height), interpolation=cv2.INTER_LINEAR)
                out.write(resized_frame)
                progress_bar.update(1)
            out.release()
        logging.info(f"Augmented video saved at: {output_path}")
        
    def process_single_video(self, 
                            video_file: str, 
                            start_second: int = 0, 
                            end_second: Optional[int] = None,
                            augmentations: Optional[Dict[str, List[Dict]]] = None,
                            save: bool = True,
                            crop: bool = True) -> Optional[Dict[str, str]]:
        """
        Processes a single video file, detecting and cropping faces, with configurable augmentations.

        Args:
            video_file: The name of the video file.
            start_second: Start processing from this second.
            end_second: End processing at this second.
            augmentations: A dictionary specifying augmentations and their parameters.
            save: If True, the outputs will be saved as videos in corresponding folders.
            crop: If True, crops the faces before applying augmentations.

        Returns:
            A dictionary containing the output video paths for each augmentation, or None if processing fails.
        """
        input_path = os.path.join(self.input_folder, video_file)
        output_base = os.path.splitext(video_file)[0]
        output_paths = {}

        if augmentations is None:
            augmentations = {}

        # try:
        logging.info(f"Starting processing for video: {input_path}")
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            logging.warning(f"Failed to open video: {input_path}")
            return None

        fps, width, height, total_frames = get_video_info(cap)
        start_frame = int(start_second * fps)
        end_frame = int(end_second * fps) if end_second is not None else total_frames
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        all_augmented_frames = {}
        cropped_frames = []
        temp_detection_result = []

        
        with tqdm(total=end_frame - start_frame, desc=f"Processing {os.path.basename(input_path)}", unit="frame", disable=not self.use_tqdm) as progress_bar:
            while cap.isOpened() and cap.get(cv2.CAP_PROP_POS_FRAMES) < end_frame:
                print(f'frame = {cap.get(cv2.CAP_PROP_POS_FRAMES)}')
                ret, frame = cap.read()
                if not ret:
                    logging.warning(f"Failed to read frame at position {cap.get(cv2.CAP_PROP_POS_FRAMES)} in video: {input_path}")
                    break

                if cap.get(cv2.CAP_PROP_POS_FRAMES) % 1000 == 0:
                    logging.info(f"frame number {cap.get(cv2.CAP_PROP_POS_FRAMES)} in video: {input_path}")

                cropped_frame, temp_detection_result = self._process_frame(frame, width, height, temp_detection_result) if crop else frame
                if cropped_frame is not None:
                    if crop:
                        cropped_frames.append(cropped_frame)

                    augmented_frames = self._augment_frame(cropped_frame, augmentations)
                    for aug_key, aug_frame in augmented_frames.items():
                        if aug_key not in all_augmented_frames:
                            all_augmented_frames[aug_key] = []
                        all_augmented_frames[aug_key].append(aug_frame)

                progress_bar.update(1)

            cap.release()

        logging.info(f"Completed processing for video: {input_path}")

        if save:
            if crop and cropped_frames:
                print('saving')
                cropped_folder = os.path.join(self.output_folder, f"cropped")
                os.makedirs(cropped_folder, exist_ok=True)
                cropped_output_path = os.path.join(cropped_folder, f"{output_base}_cropped.mp4")
                self._save_video(cropped_frames, cropped_output_path, fps, start_frame, end_frame)
                output_paths["cropped"] = cropped_output_path
                print(f'saved {cropped_output_path}')

            for aug_key, aug_frames in all_augmented_frames.items():
                aug_folder = os.path.join(self.output_folder, f"{aug_key}")
                os.makedirs(aug_folder, exist_ok=True)
                aug_output_path = os.path.join(aug_folder, f"{output_base}_{aug_key}.mp4")
                self._save_video(aug_frames, aug_output_path, fps, start_frame, end_frame)
                output_paths[aug_key] = aug_output_path

            if not output_paths:
                logging.warning(f"No faces detected in video: {input_path}")
                return None

            return output_paths

        else:
            return output_paths

        # except Exception as e:
        #     logging.error(f"Error processing {video_file}: %s", traceback.format_exc()) # log the full traceback
        #     return None

    def process_multiple_videos(self, 
                                video_files: List[str], 
                                parallel: bool = False, 
                                start_second: int = 0, 
                                end_second: Optional[int] = None,
                                augmentations: Optional[Dict[str, List[Dict]]] = None,
                                save: bool = True) -> List[Optional[str]]:
        """
        Processes multiple video files, either sequentially or in parallel.

        Args:
            video_files: A list of video file names.
            parallel: If True, processes videos in parallel using threads.
            start_second: Start processing from this second.
            end_second: End processing at this second.
            augmentations: A dictionary specifying augmentations and their parameters.
            save: If True, the outputs will be saved as videos in corresponding folders.

        Returns:
            A list of output video paths (or None if processing failed).
        """

        video_files  = os.listdir(self.input_folder) if video_files == None else video_files
    
        if parallel:
            with ProcessPoolExecutor() as executor:
                results = list(executor.map(lambda file: self.process_single_video(file, start_second, end_second, augmentations, save), video_files))
            return results
        else:
            results = [self.process_single_video(file, start_second, end_second, save) for file in video_files]
            return results
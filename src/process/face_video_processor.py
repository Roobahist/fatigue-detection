from __future__ import annotations

import gc
import logging
import multiprocessing
import os
import sys
import traceback
import tracemalloc
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from functools import partial
from multiprocessing import Pool
from typing import Dict, List, Optional, Union

import cv2
import mediapipe as mp
import memory_profiler
import numpy as np
import objgraph as o
from deepface import DeepFace as DF
from memory_profiler import profile
from tqdm import tqdm

from Zh.src.utils.cropper import crop_face_from_frame
from Zh.src.utils.csvwriter import CSVWriter
from Zh.src.utils.face_detector import FaceDetector
from Zh.src.utils.memory import cleanup
from Zh.src.utils.memory import print_all_memory as vm
from Zh.src.utils.video import get_video_info
from Zh.src.utils.serialized_parallel import process

# log_file = open("/home/mojtaba.kananisarche/Zh/temp/mem3.log", "w")

# def _process_video_helper(
#     video_processor_instance, file, start_second, end_second, augmentations, save, crop
# ):
#     try:
#         video_processor_instance.detector = FaceDetector(False, False, 1).landmarker
#         return video_processor_instance.process_single_video(
#             file, start_second, end_second, augmentations, save, crop
#         )
#     except:
#         return None


class VideoProcessor:
    """
    A class to process video files, detecting and cropping faces, with support for multiple videos.
    """

    # @profile(stream=log_file)
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

    # @profile(stream=log_file)
    def _augment_frame(
        self, frame: np.ndarray, augmentations: Dict[str, List[Dict]]
    ) -> Dict[str, np.ndarray]:
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
                    augmented_frame = cv2.GaussianBlur(
                        augmented_frame, (kernel_size, kernel_size), 0
                    )

                elif aug_name == "brightness":
                    alpha = aug_params.get("alpha", 1.1)
                    beta = aug_params.get("beta", 0)
                    aug_key = f"{aug_name}_alpha{alpha}_beta{beta}"
                    augmented_frame = np.clip(
                        augmented_frame * alpha + beta, 0, 255
                    ).astype(np.uint8)

                elif aug_name == "contrast":
                    alpha = aug_params.get("alpha", 1.2)
                    beta = aug_params.get("beta", 0)
                    aug_key = f"{aug_name}_alpha{alpha}_beta{beta}"
                    augmented_frame = np.clip(
                        augmented_frame * alpha + beta, 0, 255
                    ).astype(np.uint8)

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
                    augmented_frame = cv2.warpAffine(
                        augmented_frame, matrix, (width, height)
                    )

                augmented_frames[aug_key] = augmented_frame

        return augmented_frames

    # @profile(stream=log_file)
    def _process_frame(
        self,
        frame: cv2.Mat,
        width: int,
        height: int,
        temp_detection_result: Optional["mp.tasks.vision.FaceDetectorResult"] = [],
    ) -> Optional[cv2.Mat]:
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
        try:
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
            detector = FaceDetector(False, False, 1).landmarker
            detection_result = detector.detect(mp_image)

            cropped_face = crop_face_from_frame(
                frame, detection_result, width, height, margin=0.1
            )
            if cropped_face is not None:
                return cropped_face, detection_result
            if temp_detection_result == []:
                return None, []
            cropped_face = crop_face_from_frame(
                frame, temp_detection_result, width, height, margin=0.1
            )
            return cropped_face, temp_detection_result

        except Exception as e:
            logging.error(f"Error processing a frame")
            return None

        finally:
            # print(locals())
            cleanup(
                ["mp_image", "detector", "detection_result"],
                local_vars=locals(),
                global_vars=globals(),
            )

    # @profile(stream=log_file)
    def _save_video(
        self,
        frames: List[np.ndarray],
        output_path: str,
        fps: float,
        start_frame: int,
        end_frame: int,
    ) -> None:
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
            logging.warning(
                f"No frames to save for {output_path}. Skipping video save."
            )
            return

        first_frame_height, first_frame_width = frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(
            output_path, fourcc, fps, (first_frame_width, first_frame_height)
        )

        with tqdm(
            total=end_frame - start_frame,
            desc=f"Saving {os.path.basename(output_path)}",
            unit="frame",
            disable=not self.use_tqdm,
        ) as progress_bar:
            for frame in frames:
                resized_frame = cv2.resize(
                    frame,
                    (first_frame_width, first_frame_height),
                    interpolation=cv2.INTER_LINEAR,
                )
                out.write(resized_frame)
                progress_bar.update(1)
            out.release()
        logging.info(f"Augmented video saved at: {output_path}")

    # @profile(stream=log_file)
    def process_single_video(
        self,
        video_file: str,
        start_second: int = 0,
        end_second: Optional[int] = None,
        augmentations: Optional[Dict[str, List[Dict]]] = None,
        save: bool = True,
        crop: bool = True,
        find_embedding: bool = True,
        embedding_model: str = "Facenet",
    ) -> Optional[Dict[str, str]]:
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

        try:
            logging.info(f"Starting processing for video: {input_path}")
            cap = cv2.VideoCapture(input_path)
            if not cap.isOpened():
                logging.warning(f"Failed to open video: {input_path}")
                return None

            fps, width, height, total_frames, _ = get_video_info(cap)
            start_frame = int(start_second * fps)
            end_frame = (
                int(end_second * fps) if end_second is not None else total_frames
            )

            if save:
                try:
                    first_frame_height, first_frame_width = -1, -1
                    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
                    temp_detection_result = []
                    while (
                        cap.isOpened()
                        and first_frame_width == -1
                        and first_frame_height == -1
                    ):
                        ret, frame = cap.read()
                        if not ret:
                            logging.warning(
                                f"Failed to read frame at position {cap.get(cv2.CAP_PROP_POS_FRAMES)} in video: {input_path}"
                            )
                            break
                        cropped_frame, temp_detection_result = (
                            self._process_frame(
                                frame, width, height, temp_detection_result
                            )
                            if crop
                            else frame
                        )
                        if cropped_frame is not None:
                            first_frame_height, first_frame_width = cropped_frame.shape[
                                :2
                            ]
                            augmented_frames = self._augment_frame(
                                cropped_frame, augmentations
                            )
                    logging.warning(
                        f"New video dimensions found: {first_frame_height}, {first_frame_width}"
                    )

                except Exception as e:
                    logging.error(
                        f"Error processing {video_file}: %s", traceback.format_exc()
                    )
                    return None

                if crop:
                    cropped_folder = os.path.join(
                        self.output_folder, "video", f"cropped"
                    )
                    os.makedirs(cropped_folder, exist_ok=True)
                    cropped_output_path = os.path.join(
                        cropped_folder, f"{output_base}_cropped.mp4"
                    )
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    out_crop = cv2.VideoWriter(
                        cropped_output_path,
                        fourcc,
                        fps,
                        (first_frame_width, first_frame_height),
                    )
                    if find_embedding:
                        cropped_embedding_folder = os.path.join(
                            self.output_folder,
                            "embedding",
                            f"{embedding_model}",
                            f"cropped",
                        )
                        os.makedirs(cropped_embedding_folder, exist_ok=True)
                        cropped_embedding_path = os.path.join(
                            cropped_embedding_folder, f"{output_base}_cropped.csv"
                        )
                        cropped_embedding_writer = CSVWriter(cropped_embedding_path)

                aug_writers = {}
                aug_embedding_writers = {}

                for aug_key in augmented_frames.keys():
                    aug_folder = os.path.join(self.output_folder, "video", f"{aug_key}")
                    os.makedirs(aug_folder, exist_ok=True)
                    aug_output_path = os.path.join(
                        aug_folder, f"{output_base}_{aug_key}.mp4"
                    )
                    aug_writers[aug_key] = cv2.VideoWriter(
                        aug_output_path,
                        fourcc,
                        fps,
                        (first_frame_width, first_frame_height),
                    )
                    if find_embedding:
                        aug_embedding_folder = os.path.join(
                            self.output_folder,
                            "embedding",
                            f"{embedding_model}",
                            f"{aug_key}",
                        )
                        os.makedirs(aug_embedding_folder, exist_ok=True)
                        aug_embedding_path = os.path.join(
                            aug_embedding_folder, f"{output_base}_{aug_key}.csv"
                        )
                        aug_embedding_writers[aug_key] = CSVWriter(aug_embedding_path)

                cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
                temp_detection_result = []
                with tqdm(
                    total=end_frame - start_frame,
                    desc=f"Processing {os.path.basename(input_path)}",
                    unit="frame",
                    disable=not self.use_tqdm,
                ) as progress_bar:
                    while (
                        cap.isOpened() and cap.get(cv2.CAP_PROP_POS_FRAMES) < end_frame
                    ):
                        ret, frame = cap.read()
                        if not ret:
                            logging.warning(
                                f"Failed to read frame at position {cap.get(cv2.CAP_PROP_POS_FRAMES)} in video: {input_path}"
                            )
                            break
                        if cap.get(cv2.CAP_PROP_POS_FRAMES) % 1000 == 0:
                            logging.info(
                                f"frame number {cap.get(cv2.CAP_PROP_POS_FRAMES)} in video: {input_path}"
                            )

                        cropped_frame, temp_detection_result = (
                            self._process_frame(
                                frame, width, height, temp_detection_result
                            )
                            if crop
                            else frame
                        )
                        if cropped_frame is not None:
                            if crop:
                                resized_frame = cv2.resize(
                                    cropped_frame,
                                    (first_frame_width, first_frame_height),
                                    interpolation=cv2.INTER_LINEAR,
                                )
                                out_crop.write(resized_frame)
                                if find_embedding:
                                    cropped_embedding_writer.write(
                                        [
                                            round(x, 3)
                                            for x in DF.represent(
                                                img_path=resized_frame,
                                                enforce_detection=False,
                                                detector_backend="skip",
                                                model_name=embedding_model,
                                                normalization=(
                                                    "Facenet"
                                                    if embedding_model == "Facenet"
                                                    else "base"
                                                ),
                                            )[0]["embedding"]
                                        ]
                                    )

                            augmented_frames = self._augment_frame(
                                cropped_frame, augmentations
                            )
                            for aug_key, aug_frame in augmented_frames.items():
                                resized_frame = cv2.resize(
                                    aug_frame,
                                    (first_frame_width, first_frame_height),
                                    interpolation=cv2.INTER_LINEAR,
                                )
                                aug_writers[aug_key].write(resized_frame)
                                if find_embedding:
                                    aug_embedding_writers[aug_key].write(
                                        [
                                            round(x, 3)
                                            for x in DF.represent(
                                                img_path=resized_frame,
                                                enforce_detection=False,
                                                detector_backend="skip",
                                                model_name=embedding_model,
                                                normalization=(
                                                    "Facenet"
                                                    if embedding_model == "Facenet"
                                                    else "base"
                                                ),
                                            )[0]["embedding"]
                                        ]
                                    )
                        progress_bar.update(1)

                if crop:
                    out_crop.release()
                    if find_embedding:
                        cropped_embedding_writer.close()

                for aug_key, writer in aug_writers.items():
                    writer.release()
                    if find_embedding:
                        aug_embedding_writers[aug_key].close()

            cap.release()
            return None

        except Exception as e:
            logging.error(f"Error processing {video_file}: %s", traceback.format_exc())
            return None

        finally:
            cleanup(
                [
                    "cap",
                    "out_crop",
                    "aug_writers",
                    "cropped_frame",
                    "temp_detection_result",
                    "augmented_frames",
                    "frame",
                    "cropped_embedding_writer",
                    "aug_embedding_writers",
                ],
                local_vars=locals(),
                global_vars=globals(),
            )

    # @profile(stream=log_file)
    def process_multiple_videos(
        self,
        video_files: List[str],
        start_second: int = 0,
        end_second: Optional[int] = None,
        augmentations: Optional[Dict[str, List[Dict]]] = None,
        save: bool = True,
        crop: bool = True,
        find_embedding: bool = True,
        embedding_model: str = "Facenet",
        parallel: bool = False,
        max_workers: Optional[int] = 10,
        maxtasksperchild: int = 1,
        use_tqdm: bool = False,
    ) -> List[Optional[Dict[str, str]]]:
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
        video_files = (
            os.listdir(self.input_folder) if video_files == [] else video_files
        )

        args_list = [
            {
                'video_file' : video_file,
                'start_second' : start_second,
                'end_second' : end_second,
                'augmentations' : augmentations,
                'save' : save,
                'crop' : crop,
                'find_embedding' : find_embedding,
                'embedding_model' : embedding_model,
            }
            for video_file in video_files
        ]

        _process_single_video = partial(self.process_single_video)

        process(
            worker_function = _process_single_video,
            worker_params_list = args_list,
            handler_function = None,
            parallel = parallel,
            max_workers = max_workers,
            maxtasksperchild = maxtasksperchild,
            use_tqdm = use_tqdm,
        )

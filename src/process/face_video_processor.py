from __future__ import annotations

import gc
import logging
import os
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from functools import partial
from multiprocessing import Pool
from typing import Dict, List, Optional, Tuple, Union

import cv2
import mediapipe as mp
import numpy as np
from deepface import DeepFace as DF
from tqdm import tqdm

from Zh.src.utils.csvwriter import CSVWriter
from Zh.src.utils.face import calculate_ROI_face, process_face
from Zh.src.utils.face_detector import FaceDetector
from Zh.src.utils.file_folder import create_folders
from Zh.src.utils.memory import cleanup
from Zh.src.utils.serialized_parallel import process
from Zh.src.utils.video import VideoReader, VideoWriter


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
        frame_augmentations = {}
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

                frame_augmentations[aug_key] = augmented_frame

        return frame_augmentations

    def _process_frame(
        self,
        frame: np.ndarray,
        mask: bool = True,
        align: bool = True,
        crop: bool = True,
        margin: float = 0.05,
        output_size: Union[Tuple[int, int], str] = (224, 224),
        temp_detection_result: Union[object, None] = None,
        landmark_groups: list[Optional[list[int]]] = [],
        margin_h: list[Optional[float], float] = 0.0,
        margin_w: list[Optional[float], float] = 0.0,
        cropped_roi_output_sizes: list[Optional[Union[Tuple[int, int], str]]] = [],
    ) -> tuple[Optional[np.ndarray], Union[object, None]]:
        """
        Processes a single frame, detecting and cropping the face, if the face is not detected in the current frame,
        it tries to use the previous detection result if available.

        Args:
            frame: The input frame.
            mask: Whether to apply a mask to the face region.
            align: Whether to align the face based on eye positions.
            crop: Whether to crop the face region.
            margin: The margin to add around the bounding box as a fraction of the face's width/height.
            output_size: The desired output size of the cropped face (width, height).
            temp_detection_result: Previous detection result if available.
            landmark_groups: A list of landmark groups to consider for cropping the regions of interest.
            margin_h: The margin to add around the bounding box as a fraction of the face's height.
            margin_w: The margin to add around the bounding box as a fraction of the face's width.
            return_cropped_roi: Whether to return the cropped region of interest.
            cropped_roi_output_sizes: A list of output sizes for the cropped region of interest.

        Returns:
            A tuple containing the cropped face frame and detection result, or (None, None) if face detection fails.
        """
        try:
            fd = FaceDetector(
                output_face_blendshapes=False,
                output_facial_transformation_matrixes=False,
                num_faces=1,
            )
            detection_result = fd.detect(frame)

            cropped_face, detection_result = process_face(
                frame=frame,
                detection_result=detection_result,
                mask=mask,
                align=align,
                crop=crop,
                margin=margin,
                output_size=output_size,
            )
            if cropped_face is not None:
                regions = calculate_ROI_face(
                    frame=cropped_face,
                    detection_result=detection_result,
                    landmark_groups=landmark_groups,
                    margin_h=margin_h,
                    margin_w=margin_w,
                    return_cropped_roi=True,
                    cropped_roi_output_sizes=cropped_roi_output_sizes,
                )
                return cropped_face, detection_result, regions

            # If current detection failed but we have a previous result, try using that
            if temp_detection_result is not None:
                cropped_face, temp_detection_result = process_face(
                    frame=frame,
                    detection_result=temp_detection_result,
                    mask=mask,
                    align=align,
                    crop=crop,
                    margin=margin,
                    output_size=output_size,
                )
                if cropped_face is not None:
                    regions = calculate_ROI_face(
                        frame=cropped_face,
                        detection_result=temp_detection_result,
                        landmark_groups=landmark_groups,
                        margin_h=margin_h,
                        margin_w=margin_w,
                        return_cropped_roi=True,
                        cropped_roi_output_sizes=cropped_roi_output_sizes,
                    )
                    return cropped_face, temp_detection_result, regions

            return None, None

        except Exception as e:
            logging.error(f"Error processing a frame: {traceback.format_exc()}")
            return None, None
        finally:
            cleanup(
                ["mp_image", "detector", "detection_result"],
                local_vars=locals(),
                global_vars=globals(),
            )

    def _initialize_video_output(
        self,
        output_base: str,
        frame_augmentations: Dict[str, np.ndarray],
        mask: bool,
        align: bool,
        crop: bool,
        margin: float,
        landmark_groups: list[Optional[list[int]]],
        fourcc: str = "mp4v",
        fps: float = 30.0,
        output_size: tuple[int, int] = (224, 224),
    ) -> tuple[
        Optional[cv2.VideoWriter],
        Dict[str, cv2.VideoWriter],
        Optional[CSVWriter],
        Dict[str, CSVWriter],
    ]:
        """
        Initializes video writers and CSV writers for output.

        Args:
            output_base: Base name for output files.
            frame_augmentations: Dictionary of augmented frames.
            mask: Whether to apply a mask to the face region.
            align: Whether to align the face based on eye positions.
            crop: Whether to crop the face region.
            margin: The margin to add around the bounding box as a fraction of the face's width/height.
            landmark_groups: List of landmark groups to consider for creating writers.
            fourcc: Four-character code for the video codec.
            fps: Frames per second of the video.
            output_size: Size of frames (width, height).

        Returns:
            A tuple containing video writers and CSV writers.
        """

        # Build a descriptive string for the main processing pipeline
        process_tag = ""
        process_tag += "_masked" if mask else "_unmasked"
        process_tag += "_aligned" if align else "_unaligned"
        process_tag += f"_cropped{margin}" if crop else "_uncropped"
        # remove the first underscore
        if process_tag.startswith("_"):
            process_tag = process_tag[1:]

        processed_folder = os.path.join(self.output_folder, process_tag)
        create_folders(processed_folder)

        ROIs_names = ["base"] + [f"ROI_{i}" for i in range(len(landmark_groups))]
        augmentations_names = ["base"] + list(frame_augmentations.keys())

        writers = {}
        for ROI_name in ROIs_names:
            for aug_name in augmentations_names:
                create_folders(os.path.join(processed_folder, ROI_name, aug_name))

                key = f"{ROI_name}__{aug_name}"
                output_path = os.path.join(
                    processed_folder,
                    ROI_name,
                    aug_name,
                    f"{output_base}__{process_tag}__{ROI_name}__{aug_name}.mp4",
                )
                writers[key] = VideoWriter(
                    output_path=output_path,
                    fourcc=fourcc,
                    fps=fps,
                    frame_size=output_size,
                )

        return writers

    # def _get_embedding(self, frame: np.ndarray, embedding_model: str) -> List[float]:
    #     """
    #     Gets embedding for a frame using DeepFace.

    #     Args:
    #         frame: Input frame.
    #         embedding_model: Name of the embedding model.

    #     Returns:
    #         List of embedding values.
    #     """
    #     return [
    #         round(x, 3)
    #         for x in DF.represent(
    #             img_path=frame,
    #             enforce_detection=False,
    #             detector_backend="skip",
    #             model_name=embedding_model,
    #             normalization="Facenet" if embedding_model == "Facenet" else "base",
    #         )[0]["embedding"]
    #     ]

    def process_single_video(
        self,
        video_file: str,
        start_second: int = 0,
        end_second: Optional[int] = None,
        mask: bool = True,
        align: bool = True,
        crop: bool = True,
        margin: float = 0.05,
        augmentations: Optional[Dict[str, List[Dict]]] = None,
        output_size: Union[Tuple[int, int], str] = (224, 224),
        landmark_groups: list[Optional[list[int]]] = [],
        margin_h: list[Optional[float], float] = 0.0,
        margin_w: list[Optional[float], float] = 0.0,
        cropped_roi_output_sizes: list[Optional[Union[Tuple[int, int], str]]] = [],
    ) -> Optional[Dict[str, str]]:
        """
        Processes a single video file, detecting and cropping faces, with configurable augmentations.

        Args:
            video_file: The name of the video file.
            start_second: Start processing from this second.
            end_second: End processing at this second.
            augmentations: A dictionary specifying augmentations and their parameters.
            mask: Whether to apply a mask to the face region.
            align: Whether to align the face based on eye positions.
            crop: If True, crops the faces before applying augmentations.
            margin: The margin to add around the bounding box as a fraction of the face's width/height.
            output_size: The desired output size of the cropped face (width, height).
            landmark_groups: A list of landmark groups to consider for cropping the regions of interest.
            margin_h: The margin to add around the bounding box as a fraction of the face's height.
            margin_w: The margin to add around the bounding box as a fraction of the face's width.
            return_cropped_roi: Whether to return the cropped region of interest.
            cropped_roi_output_sizes: A list of output sizes for the cropped region of interest.

        Returns:
            A dictionary containing the output video paths for each augmentation, or None if processing fails.
        """
        print(f"Processing video: {video_file}")

        input_path = os.path.join(self.input_folder, video_file)
        output_base = os.path.splitext(video_file)[0]

        try:
            logging.info(f"Starting processing for video: {input_path}")

            # Initialize VideoReader
            vr = VideoReader(video_path=input_path)

            video_info = vr.get_info()
            fps = video_info["fps"]
            total_frames = video_info["total_frames"]

            # Calculate start and end frames
            start_frame = int(start_second * fps)
            end_frame = (
                int(end_second * fps) if end_second is not None else total_frames
            )

            # if output_size is "original" or "original_square", we need to find the size from the first valid frame
            if isinstance(output_size, str):
                temp_detection_result = None
                for frame_index in range(start_frame, end_frame):
                    ret, frame = vr.get_frame(frame_index=frame_index, jump=False)

                    if not ret:
                        continue

                    processed_frame, _, _ = self._process_frame(
                        frame=frame,
                        temp_detection_result=temp_detection_result,
                        mask=mask,
                        align=align,
                        crop=crop,
                        margin=margin,
                        output_size=output_size,
                    )

                    if processed_frame is None:
                        continue

                    output_size = (
                        processed_frame.shape[1],
                        processed_frame.shape[0],
                    )

            if isinstance(output_size, str):
                print(
                    "No valid frame found in the video to determine output size, defaulting to (224, 224)."
                )
                output_size = (224, 224)

            print(f"Output size set to: {output_size}")

            # augment the first frame to get the keys of augmentations
            frame_augmentations = self._augment_frame(
                frame=vr.get_frame(frame_index="next", jump=False)[1],
                augmentations=augmentations,
            )

            # Initialize output writers
            processed_video_writer, processed_augmented_videos_writers = (
                self._initialize_video_output(
                    output_base=output_base,
                    frame_augmentations=frame_augmentations,
                    mask=mask,
                    align=align,
                    crop=crop,
                    margin=margin,
                    fps=fps,
                    output_size=output_size,
                )
            )

            # Process frames
            temp_detection_result = None

            for frame_index in tqdm(
                range(start_frame, end_frame),
                desc=f"Processing {video_file}",
                disable=not self.use_tqdm,
                unit="frame",
                total=end_frame - start_frame,
                miniters=100,
            ):
                ret, frame = vr.get_frame(frame_index=frame_index, jump=True)
                if not ret:
                    print(
                        f"{video_file} Failed to read frame at index {frame_index}, skipping."
                    )
                    continue

                # Process frame
                processed_frame, temp_detection_result, regions = self._process_frame(
                    frame=frame,
                    temp_detection_result=temp_detection_result,
                    mask=mask,
                    align=align,
                    crop=crop,
                    margin=margin,
                    output_size=output_size,
                    landmark_groups=landmark_groups,
                    margin_h=margin_h,
                    margin_w=margin_w,
                    cropped_roi_output_sizes=cropped_roi_output_sizes,
                )

                if processed_frame is not None:
                    processed_video_writer.write_frame(processed_frame)

                    # Process and write augmented frames
                    frame_augmentations = self._augment_frame(
                        frame=processed_frame, augmentations=augmentations
                    )
                    for aug_key, aug_frame in frame_augmentations.items():
                        processed_augmented_videos_writers[aug_key].write_frame(
                            aug_frame
                        )

            return None

        except Exception as e:
            logging.error(f"Error processing {video_file}: %s", traceback.format_exc())
            return None
        finally:
            processed_video_writer.release()

            for aug_key in frame_augmentations.keys():
                processed_augmented_videos_writers[aug_key].release()

            vr.release()

            cleanup(
                [
                    "vr",
                    "processed_video_writer",
                    "processed_augmented_videos_writers",
                    "processed_frame",
                    "temp_detection_result",
                    "frame_augmentations",
                    "frame",
                    "cropped_embedding_writer",
                    "aug_embedding_writers",
                ],
                local_vars=locals(),
                global_vars=globals(),
            )

    def process_multiple_videos(
        self,
        video_files: List[str],
        start_second: int = 0,
        end_second: Optional[int] = None,
        augmentations: Optional[Dict[str, List[Dict]]] = None,
        mask: bool = True,
        align: bool = True,
        crop: bool = True,
        margin: float = 0.05,
        output_size: Union[Tuple[int, int], str] = (224, 224),
        landmark_groups: list[Optional[list[int]]] = [],
        margin_h: list[Optional[float], float] = 0.0,
        margin_w: list[Optional[float], float] = 0.0,
        cropped_roi_output_sizes: list[Optional[Union[Tuple[int, int], str]]] = [],
        parallel: bool = False,
        max_workers: Optional[int] = None,
        maxtasksperchild: int = 1,
        use_tqdm: bool = False,
    ) -> List[Optional[Dict[str, str]]]:
        """
        Processes multiple video files, either sequentially or in parallel.

        Args:
            video_files: A list of video file names.
            start_second: Start processing from this second.
            end_second: End processing at this second.
            augmentations: A dictionary specifying augmentations and their parameters.
            mask: Whether to apply a mask to the face region.
            align: Whether to align the face based on eye positions.
            crop: If True, crops the faces before applying augmentations.
            margin: The margin to add around the bounding box as a fraction of the face's width/height.
            output_size: The desired output size of the cropped face (width, height).
            landmark_groups: A list of landmark groups to consider for cropping the regions of interest.
            margin_h: The margin to add around the bounding box as a fraction of the face's height.
            margin_w: The margin to add around the bounding box as a fraction of the face's width.
            return_cropped_roi: Whether to return the cropped region of interest.
            cropped_roi_output_sizes: A list of output sizes for the cropped region of interest.
            parallel: If True, processes videos in parallel.
            max_workers: Maximum number of workers for parallel processing.
            maxtasksperchild: Maximum tasks per child process.
            use_tqdm: Whether to use tqdm progress bars.

        Returns:
            A list of output video paths (or None if processing failed).
        """
        video_files = os.listdir(self.input_folder) if not video_files else video_files
        logging.info(f"Processing {len(video_files)} videos.")

        args_list = [
            {
                "video_file": video_file,
                "start_second": start_second,
                "end_second": end_second,
                "mask": mask,
                "align": align,
                "crop": crop,
                "margin": margin,
                "augmentations": augmentations,
                "output_size": output_size,
                "landmark_groups": landmark_groups,
                "margin_h": margin_h,
                "margin_w": margin_w,
                "cropped_roi_output_sizes": cropped_roi_output_sizes,
            }
            for video_file in video_files
        ]

        process(
            worker_function=self.process_single_video,
            worker_params_list=args_list,
            handler_function=None,
            parallel=parallel,
            max_workers=max_workers,
            maxtasksperchild=maxtasksperchild,
            use_tqdm=use_tqdm,
        )


if __name__ == "__main__":
    import os
    import tracemalloc

    import memory_profiler
    import objgraph as o

    from Zh.src.process.face_video_processor import *
    from Zh.src.utils.memory import print_all_memory as vm

    pid = os.getpid()
    print(pid)

    # breakpoint()

    input_folder = "/home/mojtaba.kananisarche/Zh/data/UTA/1-raw_30fps"
    # output_folder='/work/long_lab/Mojtaba/data/UTA/1.4-augmentated_masked'
    output_folder = "/home/mojtaba.kananisarche/Zh/temp/out"

    vp = VideoProcessor(
        input_folder=input_folder, output_folder=output_folder, use_tqdm=False
    )

    augmentations = {
        "gaussianblur": [{"kernel_size": 3}, {"kernel_size": 5}],
        "brightness": [{"alpha": 1.2, "beta": 10}, {"alpha": 0.8, "beta": -10}],
        "contrast": [{"alpha": 1.5, "beta": 0}, {"alpha": 0.7, "beta": 0}],
        "flip": [
            {"flip_code": 1},  # Vertical flip
        ],
        "grayscale": [{}],  # No parameters needed
    }
    vp.process_multiple_videos(
        # video_files      = [],                           # all
        video_files=os.listdir(input_folder)[:2],
        start_second=0,
        # end_second       = None,
        end_second=1,
        # augmentations    = augmentations,
        augmentations={},
        mask=True,
        align=True,
        crop=True,
        margin=0.05,
        output_size=(224, 224),
        landmark_groups=[],
        margin_h=0.0,
        margin_w=0.0,
        cropped_roi_output_sizes=[(224, 224)],
        parallel=False,
        max_workers=None,
        maxtasksperchild=None,
        use_tqdm=True,
    )

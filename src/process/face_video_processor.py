from __future__ import annotations

import logging
import os
import traceback
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
from tqdm import tqdm

from Zh.src.utils.csvwriter import CSVWriter
from Zh.src.utils.face import calculate_ROI_face, process_face
from Zh.src.utils.face_detector import FaceDetector
from Zh.src.utils.file_folder import create_folders
from Zh.src.utils.memory import cleanup
from Zh.src.utils.serialized_parallel import process
from Zh.src.utils.video import VideoReader, VideoWriter
from Zh.src.utils.common import get_frame_size


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

                if aug_name == "base":
                    aug_key = f"{aug_name}"
                    frame_augmentations[aug_key] = augmented_frame

                elif aug_name == "gaussianblur":
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
                    augmented_frame = cv2.cvtColor(augmented_frame, cv2.COLOR_GRAY2BGR) # to go back to 3 channels

                elif aug_name == "rotate":
                    angle = aug_params.get("angle", 10)
                    aug_key = f"{aug_name}_angle{angle}"
                    width, height = get_frame_size(augmented_frame)
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
        crop_mode: str = "center_based",
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
            crop_mode: The mode to use for cropping the region of interest.

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

            cropped_face, updated_detection_result = process_face(
                frame=frame,
                detection_result=detection_result,
                mask=mask,
                align=align,
                crop=crop,
                margin=margin,
                output_size=output_size,
            )

            if cropped_face is None:
                detection_result = temp_detection_result
                cropped_face, updated_detection_result = process_face(
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
                    detection_result=updated_detection_result,
                    landmark_groups=landmark_groups,
                    margin_h=margin_h,
                    margin_w=margin_w,
                    return_cropped_roi=True,
                    cropped_roi_output_sizes=cropped_roi_output_sizes,
                    crop_mode=crop_mode,
                )
                return cropped_face, detection_result, regions

            else:
                return None, None, None

        except Exception as e:
            logging.error(f"Error processing a frame: {traceback.format_exc()}")
            return None, None, None

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

        video_writers = {}
        for ROI_name in [f"ROI{i}" for i in range(len(landmark_groups) + 1)]:
            for aug_key in list(frame_augmentations.keys()):
                create_folders(os.path.join(processed_folder, ROI_name, aug_key))

                video_writer_key = f"{ROI_name}__{aug_key}"
                output_path = os.path.join(
                    processed_folder,
                    ROI_name,
                    aug_key,
                    f"{output_base}__{process_tag}__{video_writer_key}.mp4",
                )

                video_output_size = output_size if ROI_idx == 0 else get_frame_size(regions['cropped_rois'][ROI_idx-1])
                video_writers[video_writer_key] = VideoWriter(
                    output_path=output_path,
                    fourcc=fourcc,
                    fps=fps,
                    frame_size=output_size,
                )

        return video_writers

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
        crop_mode: str = "center_based",
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
            crop_mode: The mode for cropping the region of interest.

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

            for frame_index in range(start_frame, end_frame):
                ret, frame = vr.read_frame(frame_index=frame_index, jump=False)

                if not ret:
                    continue

                temp_processed_frame, _, temp_regions = self._process_frame(
                    frame=frame,
                    mask=mask,
                    align=align,
                    crop=crop,
                    margin=margin,
                    output_size=output_size,
                    temp_detection_result=None,
                    landmark_groups=landmark_groups,
                    margin_h=margin_h,
                    margin_w=margin_w,
                    cropped_roi_output_sizes=cropped_roi_output_sizes,
                    crop_mode=crop_mode,
                )

                if temp_processed_frame is None:
                    continue

                output_size = get_frame_size(temp_processed_frame)

                temp_frame_augmentations = self._augment_frame(
                    frame=frame,
                    augmentations=augmentations,
                )

                break

            if isinstance(output_size, str):
                logging.error(
                    f"Invalid output_size: {output_size}. Please provide a tuple or 'original' or 'original_square'."
                )
                return False

            # Initialize output writers
            video_writers = self._initialize_video_output(
                output_base=output_base,
                frame_augmentations=temp_frame_augmentations,
                mask=mask,
                align=align,
                crop=crop,
                margin=margin,
                landmark_groups=landmark_groups,
                fps=fps,
                output_size=output_size,
            )

            # create_folders("/home/mojtaba.kananisarche/Zh/temp/pkls")
            # for file in os.listdir("/home/mojtaba.kananisarche/Zh/temp/pkls"):
            #     os.remove(f"/home/mojtaba.kananisarche/Zh/temp/pkls/{file}")

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
                ret, frame = vr.read_frame(frame_index=frame_index, jump=True)
                if not ret:
                    print(
                        f"{video_file} Failed to read frame at index {frame_index}, skipping."
                    )
                    continue

                # Process frame
                # import pickle
                # with open("/home/mojtaba.kananisarche/Zh/temp/pkls/temp_detection_result_before.pkl", "wb") as f:
                #     pickle.dump(temp_detection_result, f)
                # with open("/home/mojtaba.kananisarche/Zh/temp/pkls/frame.pkl", "wb") as f:
                #     pickle.dump(frame, f)
                # with open("/home/mojtaba.kananisarche/Zh/temp/pkls/frame_index.pkl", "wb") as f:
                #     pickle.dump(frame_index, f)

                processed_frame, temp_detection_result, regions = self._process_frame(
                    frame=frame,
                    mask=mask,
                    align=align,
                    crop=crop,
                    margin=margin,
                    output_size=output_size,
                    temp_detection_result=temp_detection_result,
                    landmark_groups=landmark_groups,
                    margin_h=margin_h,
                    margin_w=margin_w,
                    cropped_roi_output_sizes=cropped_roi_output_sizes,
                    crop_mode=crop_mode,
                )

                # with open("/home/mojtaba.kananisarche/Zh/temp/pkls/temp_detection_result_after.pkl", "wb") as f:
                #     pickle.dump(temp_detection_result, f)
                # with open("/home/mojtaba.kananisarche/Zh/temp/pkls/processed_frame.pkl", "wb") as f:
                #     pickle.dump(processed_frame, f)

                if processed_frame is not None:
                    for roi_idx in range(
                        len([processed_frame] + regions["cropped_rois"])
                    ):
                        roi = ([processed_frame] + regions["cropped_rois"])[roi_idx]
                        frame_augmentations = self._augment_frame(
                            frame=roi, augmentations=augmentations
                        )
                        for aug_key in list(frame_augmentations.keys()):
                            video_writer_key = f"ROI{roi_idx}__{aug_key}"
                            video_writers[video_writer_key].write_frame(
                                frame_augmentations[aug_key]
                            )
            for video_writer in video_writers.values():
                video_writer.release()
            vr.release()
            return True

        except Exception as e:
            logging.error(f"Error processing {video_file}: %s", traceback.format_exc())
            return False

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
        crop_mode: str = "center_based",
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
            crop_mode: The mode to use for cropping the region of interest.
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
                "crop_mode": crop_mode,
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
    import argparse

    from Zh.src.utils.face_indices import *

    logging.info("Process ID:", os.getpid())

    parser = argparse.ArgumentParser(description="Process video files.")
    parser.add_argument(
        "input_folder",
        type=str,
        help="Path to the input folder containing video files.",
    )
    parser.add_argument(
        "output_folder",
        type=str,
        help="Path to the output folder to save processed videos.",
    )
    parser.add_argument(
        "video_files",
        nargs="*",
        help="List of video file names to process.",
    )
    parser.add_argument(
        "--start_second",
        type=int,
        default=0,
        help="Start processing from this second.",
    )
    parser.add_argument(
        "--end_second",
        type=int,
        default=None,
        help="End processing at this second.",
    )
    parser.add_argument(
        "--mask",
        action="store_true",
        help="Apply a mask to the face region.",
    )
    parser.add_argument(
        "--align",
        action="store_true",
        help="Align the face based on eye positions.",
    )
    parser.add_argument(
        "--crop",
        action="store_true",
        help="Crop the face region.",
    )
    parser.add_argument(
        "--margin",
        type=float,
        default=0.05,
        help="Margin to add around the bounding box as a fraction of the face's width/height.",
    )
    parser.add_argument(
        "--parallel",
        action="store_true",
        help="Process videos in parallel.",
    )
    parser.add_argument(
        "--max_workers",
        type=int,
        default=None,
        help="Maximum number of workers for parallel processing.",
    )
    parser.add_argument(
        "--maxtasksperchild",
        type=int,
        default=None,
        help="Maximum tasks per child process.",
    )
    parser.add_argument(
        "--no_tqdm",
        action="store_false",
        dest="use_tqdm",
        default=True,
        help="Disable tqdm progress bars.",
    )
    args = parser.parse_args()

    vp = VideoProcessor(
        input_folder=args.input_folder,
        output_folder=args.output_folder,
        use_tqdm=args.use_tqdm,
    )

    augmentations = {
        "base": [{}],
        "gaussianblur": [{"kernel_size": 3}, {"kernel_size": 5}],
        "brightness": [{"alpha": 1.2, "beta": 10}, {"alpha": 0.8, "beta": -10}],
        "contrast": [{"alpha": 1.5, "beta": 0}, {"alpha": 0.7, "beta": 0}],
        "flip": [
            {"flip_code": 1},  # Vertical flip
        ],
        "grayscale": [{}],  # No parameters needed
    }

    landmark_groups = [LEFT_EYE, RIGHT_EYE, LIPS]

    cropped_roi_output_sizes = (224, 224)

    output_size = (224, 224)

    margin_h=[0.3,0.3,0.4]
    
    margin_w=[0.3,0.3,0.5]

    crop_mode = "center_based"

    vp.process_multiple_videos(
        video_files=args.video_files,
        start_second=args.start_second,
        end_second=args.end_second,
        augmentations=augmentations,
        mask=args.mask,
        align=args.align,
        crop=args.crop,
        margin=args.margin,
        output_size=output_size,
        landmark_groups=landmark_groups,
        margin_h=margin_h,
        margin_w=margin_w,
        cropped_roi_output_sizes=cropped_roi_output_sizes,
        crop_mode=crop_mode,
        parallel=args.parallel,
        max_workers=args.max_workers,
        maxtasksperchild=args.maxtasksperchild,
        use_tqdm=args.use_tqdm,
    )

from __future__ import annotations

import logging
import os
import traceback
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
from tqdm import tqdm

from Zh.src.utils.common import get_frame_size
from Zh.src.utils.csvwriter import CSVWriter
from Zh.src.utils.face import calculate_ROI_face, process_face
from Zh.src.utils.face_detector import FaceDetector
from Zh.src.utils.file_folder import create_folders
from Zh.src.utils.memmap_handler import MemmapHandler
from Zh.src.utils.memory import cleanup
from Zh.src.utils.serialized_parallel import process
from Zh.src.utils.video import VideoReader, VideoWriter


def _add_unfinished_suffix(path: str) -> str:
    root, ext = os.path.splitext(path)
    return f"{root}_unfinished{ext}"


def _resolve_log_dir(output_folder: str) -> str:
    output_dir = os.environ.get("OUTPUT_DIR", "").strip()
    if output_dir:
        return os.path.join(output_dir, "logs")
    return os.path.join(output_folder, "logs")


def _setup_logger(log_dir: str, name: str) -> logging.Logger:
    os.makedirs(log_dir, exist_ok=True)

    logger = logging.getLogger(f"{name}_{os.getpid()}")
    logger.setLevel(logging.INFO)
    logger.propagate = False

    if logger.handlers:
        return logger

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(log_dir, f"{name}_pid{os.getpid()}_{ts}.log")
    fmt = logging.Formatter("[%(asctime)s][%(levelname)s] %(message)s")

    fh = logging.FileHandler(log_path)
    fh.setLevel(logging.INFO)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setLevel(logging.INFO)
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    logger.info(f"log_path={log_path}")
    return logger


def _init_valid_column_to_unwritten(
    mh: MemmapHandler, valid_col: int, chunk_rows: int = 200_000
) -> None:
    n = int(mh.shape[0])
    start = 0
    while start < n:
        end = min(start + chunk_rows, n)
        block = np.full((end - start, 1), -1.0, dtype=np.float32)
        mh.write(
            data=block,
            start=start,
            slice=(slice(start, end), slice(valid_col, valid_col + 1)),
        )
        start = end


def _find_resume_row_by_valid(
    mh: MemmapHandler, valid_col: int, chunk_rows: int = 200_000
) -> int:
    """
    Returns first row index where valid == -1.0 (unwritten).
    If all rows are written (valid in {0,1}), returns mh.shape[0].
    """
    n = int(mh.shape[0])
    if n == 0:
        return 0

    start = 0
    while start < n:
        end = min(start + chunk_rows, n)
        arr = mh.read(slice=(slice(start, end), slice(valid_col, valid_col + 1)))
        if arr is None:
            return start
        v = np.asarray(arr).reshape(-1)
        idx = np.where(v == -1.0)[0]
        if idx.size > 0:
            return start + int(idx[0])
        start = end
    return n


def _validate_before_finalize(
    unfinished_path: str,
    expected_rows: int,
    expected_cols: int,
    valid_col: int,
    chunk_rows: int = 200_000,
) -> Tuple[bool, str]:
    try:
        mh = MemmapHandler(path=unfinished_path, mode="r")
    except Exception as e:
        return False, f"open_failed {type(e).__name__}: {e}"

    # if int(mh.shape[0]) != int(expected_rows):
    #     return (
    #         False,
    #         f"row_count_mismatch shape_rows={mh.shape[0]} expected_rows={expected_rows}",
    #     )

    if int(mh.shape[1]) != int(expected_cols):
        return (
            False,
            f"col_count_mismatch shape_cols={mh.shape[1]} expected_cols={expected_cols}",
        )

    n = int(expected_rows)
    start = 0
    while start < n:
        end = min(start + chunk_rows, n)
        arr = mh.read(slice=(slice(start, end), slice(valid_col, valid_col + 1)))
        if arr is None:
            return False, f"valid_read_failed range={start}:{end}"

        v = np.asarray(arr).reshape(-1)

        bad_val = np.where((v != -1.0) & (v != 0.0) & (v != 1.0))[0]
        if bad_val.size > 0:
            i = start + int(bad_val[0])
            return (
                False,
                f"valid_out_of_domain row={i} value={float(v[int(bad_val[0])])}",
            )

        unwritten = np.where(v == -1.0)[0]
        if unwritten.size > 0:
            i = start + int(unwritten[0])
            return False, f"unwritten_rows_remaining first_unwritten_row={i}"

        start = end

    return True, ""


def _touch_success_files(logger: logging.Logger) -> None:
    output_dir = os.environ.get("OUTPUT_DIR", "").strip()
    if not output_dir:
        return

    os.makedirs(output_dir, exist_ok=True)

    default_success = os.path.join(output_dir, "EXPERIMENT_COMPLETE")
    try:
        with open(default_success, "a"):
            os.utime(default_success, None)
        logger.info(f"success_file_touched path={default_success}")
    except Exception as e:
        logger.error(
            f"success_file_touch_failed path={default_success} err={type(e).__name__}: {e}"
        )

    env_success = os.environ.get("SUCCESS_FILE", "").strip()
    if env_success:
        env_path = os.path.join(output_dir, env_success)
        try:
            with open(env_path, "a"):
                os.utime(env_path, None)
            logger.info(f"success_file_touched path={env_path}")
        except Exception as e:
            logger.error(
                f"success_file_touch_failed path={env_path} err={type(e).__name__}: {e}"
            )


class VideoProcessor:
    """
    A class to process video files, detecting and cropping faces, with support for multiple videos.
    """

    def __init__(self, input_folder: str, output_folder: str, use_tqdm: bool = False):
        self.input_folder = input_folder
        self.output_folder = output_folder
        self.use_tqdm = use_tqdm

        log_dir = _resolve_log_dir(output_folder=self.output_folder)
        self.logger = _setup_logger(log_dir=log_dir, name="face_features")

        # Important: FaceDetector contains MediaPipe TaskRunner which is not picklable.
        # We lazily create it inside each worker process.
        self.fd: Optional[FaceDetector] = None

    def __getstate__(self):
        state = dict(self.__dict__)
        state["fd"] = None
        state["logger"] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)

        log_dir = _resolve_log_dir(output_folder=self.output_folder)
        self.logger = _setup_logger(log_dir=log_dir, name="face_features")

        self.fd = None

    def _get_fd(self) -> FaceDetector:
        if self.fd is None:
            self.fd = FaceDetector(
                output_face_blendshapes=True,
                output_facial_transformation_matrixes=False,
                num_faces=1,
            )
            self.logger.info("face_detector_initialized")
        return self.fd

    def _augment_frame(
        self, frame: np.ndarray, augmentations: Dict[str, List[Dict]]
    ) -> Dict[str, np.ndarray]:
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
                    augmented_frame = cv2.cvtColor(augmented_frame, cv2.COLOR_GRAY2BGR)

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
        frame_index: int,
        frame: np.ndarray,
        mask: bool = True,
        align: bool = True,
        crop: bool = True,
        margin: float = 0.05,
        output_size: Union[Tuple[int, int], str] = (224, 224),
        temp_detection_result: Union[object, None] = None,
        landmark_groups: Optional[List[Optional[List[int]]]] = None,
        margin_h: Union[List[Optional[float]], float] = 0.0,
        margin_w: Union[List[Optional[float]], float] = 0.0,
        cropped_roi_output_sizes: Optional[
            List[Optional[Union[Tuple[int, int], str]]]
        ] = None,
        crop_mode: str = "center_based",
    ) -> tuple[Optional[np.ndarray], Union[object, None], dict, dict]:
        if landmark_groups is None:
            landmark_groups = []
        if cropped_roi_output_sizes is None:
            cropped_roi_output_sizes = []

        try:
            detection_result = self._get_fd().detect(frame)
            face_blendshapes = (
                detection_result.face_blendshapes[0]
                if hasattr(detection_result, "face_blendshapes")
                and len(detection_result.face_blendshapes) > 0
                else None
            )

            if face_blendshapes is None:
                blendshape_features = {}
            else:
                blendshape_features = {
                    i.category_name: float(i.score) for i in face_blendshapes
                }

            cropped_face, process_features, updated_detection_result = process_face(
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
                cropped_face, process_features, updated_detection_result = process_face(
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

                return (
                    cropped_face,
                    detection_result,
                    regions,
                    {
                        "frame_index": frame_index,
                        **process_features,
                        **blendshape_features,
                    },
                )

            else:
                return None, None, None, None

        except Exception:
            self.logger.error(
                f"frame_process_failed frame_index={frame_index} err={traceback.format_exc()}"
            )
            return None, None, None, None

    def _initialize_video_output(
        self,
        output_base: str,
        frame_augmentations: Dict[str, np.ndarray],
        regions: Dict[str, np.ndarray],
        process_tag: str,
        landmark_groups: List[Optional[List[int]]],
        fourcc: str = "mp4v",
        fps: float = 30.0,
        output_size: tuple[int, int] = (224, 224),
    ) -> Dict[str, VideoWriter]:
        processed_folder = os.path.join(self.output_folder, process_tag)
        create_folders(processed_folder)

        video_writers = {}
        for ROI_idx in range(len(landmark_groups) + 1):
            for aug_key in list(frame_augmentations.keys()):
                ROI_name = f"ROI{ROI_idx}"
                create_folders(os.path.join(processed_folder, ROI_name, aug_key))

                video_writer_key = f"{ROI_name}__{aug_key}"
                output_path = os.path.join(
                    processed_folder,
                    ROI_name,
                    aug_key,
                    f"{output_base}__{process_tag}__{video_writer_key}.mp4",
                )

                video_output_size = (
                    output_size
                    if ROI_idx == 0
                    else get_frame_size(regions["cropped_rois"][ROI_idx - 1])
                )
                video_writers[video_writer_key] = VideoWriter(
                    output_path=output_path,
                    fourcc=fourcc,
                    fps=fps,
                    frame_size=video_output_size,
                )

        return video_writers

    def _initialize_feature_output(
        self,
        output_base: str,
        process_tag: str,
        shape: tuple[int, int],
        dtype: str = "float32",
        mode: str = "w+",
        unfinished: bool = True,
    ) -> Tuple[MemmapHandler, str, str]:
        processed_folder = os.path.join(self.output_folder, process_tag)
        features_folder = os.path.join(processed_folder, "features")
        create_folders(features_folder)

        final_path = os.path.join(features_folder, f"{output_base}__features.dat")
        path = _add_unfinished_suffix(final_path) if unfinished else final_path

        return (
            MemmapHandler(
                path=path,
                shape=shape,
                dtype=dtype,
                mode=mode,
            ),
            path,
            final_path,
        )

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
        landmark_groups: Optional[List[Optional[List[int]]]] = None,
        margin_h: Union[List[Optional[float]], float] = 0.0,
        margin_w: Union[List[Optional[float]], float] = 0.0,
        cropped_roi_output_sizes: Optional[
            List[Optional[Union[Tuple[int, int], str]]]
        ] = None,
        crop_mode: str = "center_based",
        export_videos: bool = True,
        export_features: bool = True,
    ) -> Optional[Dict[str, str]]:
        if landmark_groups is None:
            landmark_groups = []
        if cropped_roi_output_sizes is None:
            cropped_roi_output_sizes = []

        input_path = os.path.join(self.input_folder, video_file)
        output_base = os.path.splitext(video_file)[0]

        self.logger.info(f"video_start video_file={video_file} input_path={input_path}")

        try:
            vr = VideoReader(video_path=input_path)

            video_info = vr.get_info()
            fps = video_info["fps"]
            total_frames = video_info["total_frames"]

            start_frame = int(start_second * fps)
            end_frame = (
                int(end_second * fps) if end_second is not None else total_frames
            )
            expected_rows = int(end_frame - start_frame)
            if expected_rows <= 0:
                self.logger.warning(
                    f"video_skip_empty video_file={video_file} start_frame={start_frame} end_frame={end_frame}"
                )
                vr.release()
                return False

            process_tag = ""
            process_tag += "_masked" if mask else "_unmasked"
            process_tag += "_aligned" if align else "_unaligned"
            process_tag += f"_cropped{margin}" if crop else "_uncropped"
            if process_tag.startswith("_"):
                process_tag = process_tag[1:]

            temp_frame_augmentations = None
            temp_regions = None
            temp_features = None

            for frame_index in range(start_frame, end_frame):
                ret, frame = vr.read_frame(frame_index=frame_index, jump=False)
                if not ret:
                    continue

                temp_processed_frame, _, temp_regions, temp_features = (
                    self._process_frame(
                        frame_index=frame_index,
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
                )

                if temp_processed_frame is None or temp_features is None:
                    continue

                output_size = get_frame_size(temp_processed_frame)
                temp_frame_augmentations = self._augment_frame(
                    frame=frame, augmentations=augmentations
                )
                break

            if temp_features is None:
                self.logger.error(f"video_fail_no_schema video_file={video_file}")
                vr.release()
                return False

            if isinstance(output_size, str):
                self.logger.error(
                    f"video_fail_invalid_output_size video_file={video_file} output_size={output_size}"
                )
                vr.release()
                return False

            if export_videos:
                video_writers = self._initialize_video_output(
                    output_base=output_base,
                    frame_augmentations=temp_frame_augmentations,
                    regions=temp_regions,
                    process_tag=process_tag,
                    landmark_groups=landmark_groups,
                    fps=fps,
                    output_size=output_size,
                )

            feature_writer = None
            feature_path_unfinished = ""
            feature_path_final = ""

            schema_keys = list(temp_features.keys())
            if "valid" in schema_keys:
                schema_keys.remove("valid")
            schema_keys.append("valid")
            schema_dim = len(schema_keys)
            valid_col = schema_dim - 1

            resume_row = 0

            if export_features:
                processed_folder = os.path.join(self.output_folder, process_tag)
                features_folder = os.path.join(processed_folder, "features")
                create_folders(features_folder)

                final_path = os.path.join(
                    features_folder, f"{output_base}__features.dat"
                )
                unfinished_path = _add_unfinished_suffix(final_path)

                if os.path.exists(final_path):
                    self.logger.info(
                        f"video_skip_features_finalized video_file={video_file} path={final_path}"
                    )
                    export_features = False
                else:
                    if os.path.exists(unfinished_path):
                        feature_writer = MemmapHandler(path=unfinished_path, mode="r+")
                        resume_row = _find_resume_row_by_valid(
                            feature_writer, valid_col=valid_col
                        )
                        self.logger.info(
                            f"video_resume_features video_file={video_file} resume_row={resume_row} path={unfinished_path}"
                        )
                    else:
                        feature_writer, unfinished_path, final_path = (
                            self._initialize_feature_output(
                                output_base=output_base,
                                process_tag=process_tag,
                                shape=(expected_rows, schema_dim),
                                dtype="float32",
                                mode="w+",
                                unfinished=True,
                            )
                        )
                        _init_valid_column_to_unwritten(
                            feature_writer, valid_col=valid_col
                        )
                        resume_row = 0
                        self.logger.info(
                            f"video_create_features video_file={video_file} path={unfinished_path} shape={(expected_rows, schema_dim)}"
                        )

                    feature_path_unfinished = unfinished_path
                    feature_path_final = final_path

            temp_detection_result = None

            start_loop_frame = start_frame
            if export_features and feature_writer is not None:
                start_loop_frame = start_frame + int(resume_row)

            cnt_read_fail = 0
            cnt_features_none = 0

            for frame_index in tqdm(
                range(start_loop_frame, end_frame),
                desc=f"Processing {video_file}",
                disable=not self.use_tqdm,
                unit="frame",
                total=end_frame - start_loop_frame,
                miniters=100,
            ):
                ret, frame = vr.read_frame(frame_index=frame_index, jump=True)
                local_idx = frame_index - start_frame

                if not ret:
                    cnt_read_fail += 1
                    # if export_features and feature_writer is not None:
                    #     row = {k: 0.0 for k in schema_keys}
                    #     row["frame_index"] = float(frame_index)
                    #     row["valid"] = 0.0
                    #     feature_writer.write(data=row, start=local_idx)
                    continue

                processed_frame, temp_detection_result, regions, features = (
                    self._process_frame(
                        frame_index=frame_index,
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
                )

                if export_features and feature_writer is not None:
                    if features is not None:
                        row = {}
                        for k in schema_keys:
                            if k == "valid":
                                continue
                            row[k] = float(features.get(k, 0.0))
                        row["valid"] = 1.0
                        feature_writer.write(
                            data=row,
                            start=local_idx - (cnt_read_fail + cnt_features_none),
                        )
                    else:
                        cnt_features_none += 1
                        # row = {k: 0.0 for k in schema_keys}
                        # row["frame_index"] = float(frame_index)
                        # row["valid"] = 0.0

                if export_videos:
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
                    else:
                        pass

            if export_features and feature_writer is not None:
                feature_writer.remove_rows(
                    drop_indices=list(
                        range(
                            expected_rows - (cnt_read_fail + cnt_features_none),
                            expected_rows,
                        )
                    )
                )

            if export_features and feature_writer is not None:
                feature_writer.close()

                ok, reason = _validate_before_finalize(
                    unfinished_path=feature_path_unfinished,
                    expected_rows=expected_rows,
                    expected_cols=schema_dim,
                    valid_col=valid_col,
                )

                if ok:
                    os.replace(feature_path_unfinished, feature_path_final)
                    os.replace(
                        feature_path_unfinished.replace(".dat", ".json"),
                        feature_path_final.replace(".dat", ".json"),
                    )  # metadata
                    self.logger.info(
                        f"video_finalize_features video_file={video_file} final_path={feature_path_final}"
                    )
                else:
                    self.logger.error(
                        f"video_not_finalized_features video_file={video_file} path={feature_path_unfinished} reason={reason}"
                    )

            if export_videos:
                for video_writer in video_writers.values():
                    video_writer.release()

            vr.release()

            if cnt_read_fail > 0 or cnt_features_none > 0:
                self.logger.warning(
                    f"video_issues video_file={video_file} read_failures={cnt_read_fail} features_none={cnt_features_none}"
                )

            self.logger.info(
                f"video_end video_file={video_file} expected_rows={expected_rows}"
            )
            return True

        except Exception:
            self.logger.error(
                f"video_fail video_file={video_file} err={traceback.format_exc()}"
            )
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
        landmark_groups: Optional[List[Optional[List[int]]]] = None,
        margin_h: Union[List[Optional[float]], float] = 0.0,
        margin_w: Union[List[Optional[float]], float] = 0.0,
        cropped_roi_output_sizes: Optional[
            List[Optional[Union[Tuple[int, int], str]]]
        ] = None,
        crop_mode: str = "center_based",
        export_videos: bool = True,
        export_features: bool = True,
        parallel: bool = False,
        max_workers: Optional[int] = None,
        maxtasksperchild: int = 1,
        use_tqdm: bool = False,
    ) -> List[Optional[Dict[str, str]]]:
        video_files = os.listdir(self.input_folder) if not video_files else video_files
        self.logger.info(
            f"batch_start n_videos={len(video_files)} input_folder={self.input_folder}"
        )

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
                "export_videos": export_videos,
                "export_features": export_features,
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

        incomplete = []
        for video_file in video_files:
            output_base = os.path.splitext(video_file)[0]

            process_tag = ""
            process_tag += "_masked" if mask else "_unmasked"
            process_tag += "_aligned" if align else "_unaligned"
            process_tag += f"_cropped{margin}" if crop else "_uncropped"
            if process_tag.startswith("_"):
                process_tag = process_tag[1:]

            features_folder = os.path.join(self.output_folder, process_tag, "features")
            final_path = os.path.join(features_folder, f"{output_base}__features.dat")
            unfinished_path = _add_unfinished_suffix(final_path)

            if os.path.exists(unfinished_path) or (
                export_features and not os.path.exists(final_path)
            ):
                incomplete.append(video_file)

        if len(incomplete) == 0:
            self.logger.info("batch_complete all_outputs_finalized=true")
            _touch_success_files(self.logger)
        else:
            self.logger.warning(
                f"batch_incomplete missing_or_unfinished_count={len(incomplete)} sample={incomplete[:5]}"
            )

        self.logger.info("batch_end")


if __name__ == "__main__":
    import argparse

    from Zh.src.utils.face_indices import *

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
        "video_files", nargs="*", help="List of video file names to process."
    )
    parser.add_argument(
        "--start_second", type=int, default=0, help="Start processing from this second."
    )
    parser.add_argument(
        "--end_second", type=int, default=None, help="End processing at this second."
    )
    parser.add_argument(
        "--mask", action="store_true", help="Apply a mask to the face region."
    )
    parser.add_argument(
        "--align", action="store_true", help="Align the face based on eye positions."
    )
    parser.add_argument("--crop", action="store_true", help="Crop the face region.")
    parser.add_argument(
        "--margin",
        type=float,
        default=0.05,
        help="Margin to add around the bounding box.",
    )
    parser.add_argument(
        "--export_videos",
        action="store_true",
        help="Export processed videos to the output folder.",
    )
    parser.add_argument(
        "--export_features",
        action="store_true",
        help="Export extracted features to the output folder.",
    )
    parser.add_argument(
        "--parallel", action="store_true", help="Process videos in parallel."
    )
    parser.add_argument(
        "--max_workers", type=int, default=None, help="Maximum number of workers."
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
        "flip": [{"flip_code": 1}],
        "grayscale": [{}],
    }

    landmark_groups = [LEFT_EYE, RIGHT_EYE, LIPS]
    cropped_roi_output_sizes = (224, 224)
    output_size = (224, 224)
    margin_h = [0.3, 0.3, 0.4]
    margin_w = [0.3, 0.3, 0.5]
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
        export_videos=args.export_videos,
        export_features=args.export_features,
        parallel=args.parallel,
        max_workers=args.max_workers,
        maxtasksperchild=args.maxtasksperchild,
        use_tqdm=args.use_tqdm,
    )

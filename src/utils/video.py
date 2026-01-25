import logging
import os
import subprocess
import traceback
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import cv2
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from Zh.src.utils.common import get_frame_size
from Zh.src.utils.cpu_gpu import detect_gpu_capability


def is_video_valid(video_path: str) -> bool:
    """
    Check if video file is valid and readable.

    Args:
        video_path: Path to the video file
        timeout: Timeout in seconds for ffprobe command

    Returns:
        bool: True if video is valid, False otherwise
    """
    # Basic file checks
    if not os.path.exists(video_path):
        logging.error(f"Video file does not exist: {video_path}")
        return False

    if os.path.getsize(video_path) == 0:
        logging.error(f"Video file is empty: {video_path}")
        return False

    # Use ffprobe to validate video streams
    command = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=codec_name,duration,nb_frames",
        "-of",
        "csv=p=0",
        video_path,
    ]

    try:
        result = subprocess.run(command, capture_output=True, text=True, check=True)

        output = result.stdout.strip()
        if not output:
            logging.error(f"ffprobe returned empty output: {video_path}")
            return False

        # Parse the CSV output: codec_name,duration,nb_frames
        parts = output.split(",")
        if len(parts) >= 1:
            codec_name = parts[0].strip()
            # Valid if we have a codec name (indicates video stream exists)
            logging.debug(f"Codec name: {codec_name}")
            return bool(codec_name and codec_name != "N/A")

        logging.error(f"ffprobe returned unexpected output: {video_path}")
        return False

    except Exception as e:
        logging.error(
            f"Error validating video {video_path}: %s", traceback.format_exc()
        )
        return False


def convert_video(
    input_path: str,
    output_directory_path: str,
    output_file_name: Optional[str] = None,
    target_fps: Optional[float] = 15,
    target_ext: Optional[str] = ".mp4",
    no_audio: bool = True,
    use_gpu: bool = False,
    optimize_for_speed: bool = True,
    preserve_duration: bool = True,
    quality_level: str = "fast",  # New parameter: "fast", "medium", "high", "best"
) -> bool:
    """
    Converts a video by changing its FPS, format, or both using FFmpeg.
    Heavily optimized for speed with intelligent parameter selection.

    Args:
        input_path (str): Path to the input video file.
        output_directory_path (str): Directory to save the processed video.
        output_file_name (str, optional): Base name for the output file.
        target_fps (float, optional): Desired frames per second. If None, FPS is unchanged.
        target_ext (str, optional): Target extension (e.g., ".mp4"). If None, format is unchanged.
        no_audio (bool): Whether to remove audio (default: True).
        use_gpu (bool): Use GPU acceleration. Defaults to False.
        optimize_for_speed (bool): Use fastest settings (may reduce quality slightly).
        preserve_duration (bool): Keep same duration when changing FPS (default: True).
        quality_level (str): Quality preset - "fast", "medium", "high", "best" (default: "medium").

    Returns:
        bool: True if processing succeeded, False otherwise.
    """
    # --- validate ---
    if not isinstance(input_path, str) or not isinstance(output_directory_path, str):
        raise TypeError("Input and output paths must be strings")

    if target_fps is not None:
        if not isinstance(target_fps, (int, float)):
            raise TypeError("Target FPS must be a number")
        if target_fps <= 0:
            raise ValueError("Target FPS must be greater than 0")

    if target_ext is not None:
        if not isinstance(target_ext, str):
            raise TypeError("Target extension must be a string")
        if not target_ext.startswith("."):
            raise ValueError("Target extension must start with '.'")

    target_ext = target_ext if target_ext is not None else Path(input_path).suffix
    output_file_name = (
        Path(output_file_name).stem
        if output_file_name is not None
        else Path(input_path).stem
    )
    output_file_name = output_file_name + f"_{target_fps}fps" + target_ext
    output_path = str(
        Path(output_directory_path, output_file_name).with_suffix(target_ext)
    )

    # Skip if already exists and valid
    if os.path.exists(output_path):
        if is_video_valid(output_path):
            logging.info(f"Video already processed: {os.path.basename(input_path)}")
            return True

    # Get video info for optimization decisions
    video_info = VideoReader(video_path=input_path).get_info()
    source_fps = video_info.get("fps", 0)
    width = video_info.get("width", 0)
    height = video_info.get("height", 0)
    duration = video_info.get("duration", 0)

    # Detect GPU capabilities
    gpu_caps = detect_gpu_capability() if use_gpu else {}

    # Quality settings based on level
    quality_settings = {
        "fast": {"crf": 28, "preset_cpu": "veryfast", "preset_gpu": "p1", "cq": 28},
        "medium": {"crf": 23, "preset_cpu": "fast", "preset_gpu": "p4", "cq": 23},
        "high": {"crf": 20, "preset_cpu": "medium", "preset_gpu": "p5", "cq": 20},
        "best": {"crf": 18, "preset_cpu": "slow", "preset_gpu": "p6", "cq": 18},
    }

    if quality_level not in quality_settings:
        quality_level = "fast"
        logging.warning(f"Invalid quality_level, using 'fast'")

    q_settings = quality_settings[quality_level]

    # --- build optimized command ---
    command = ["ffmpeg", "-y"]

    # Input optimization
    if use_gpu and gpu_caps.get("nvenc_available"):
        # Use CUDA for decoding when available
        command += ["-hwaccel", "cuda", "-hwaccel_output_format", "cuda"]

    # Threading optimization
    thread_count = min(os.cpu_count() or 4, 16)  # Cap at 16 threads
    command += ["-threads", str(thread_count)]

    command += ["-i", input_path]

    # Video filter optimizations
    filters = []

    # Handle odd dimensions (H.264 requirement: even width/height)
    if width > 0 and height > 0:
        # Calculate target dimensions (ensure even numbers)
        target_width = width + (width % 2)  # Make even by adding 1 if odd
        target_height = height + (height % 2)  # Make even by adding 1 if odd

        # Scale filter optimization for large videos with proper dimension handling
        if optimize_for_speed and width > 1920 and height > 1080:
            # Scale down large videos and ensure even dimensions
            filters.append(
                "scale='2*trunc(min(1920,iw)/2)':'2*trunc(min(1080,ih)/2)':flags=fast_bilinear"
            )
            logging.info(f"Scaling down large video from {width}x{height}")
        else:
            # Ensure even dimensions for all videos (H.264 requirement)
            if width % 2 != 0 or height % 2 != 0:
                filters.append(
                    f"scale={target_width}:{target_height}:flags=fast_bilinear"
                )
                logging.info(
                    f"Adjusting dimensions from {width}x{height} to {target_width}x{target_height}"
                )

    # FPS handling - CRITICAL FIX for duration preservation
    if target_fps is not None and source_fps > 0:
        if preserve_duration:
            # Method 1: Use fps filter to resample (preserves duration)
            filters.append(f"fps={target_fps}")
            logging.info(
                f"Converting FPS from {source_fps} to {target_fps} (duration preserved)"
            )
        else:
            # Method 2: Drop/duplicate frames (changes duration)
            if source_fps > target_fps:
                step = int(source_fps / target_fps)
                if step > 1:
                    filters.append(f"select='not(mod(n,{step}))'")
                    command += ["-vsync", "vfr"]
                    logging.info(
                        f"Dropping frames: {source_fps} -> {target_fps} (duration will change)"
                    )
            else:
                filters.append(f"fps={target_fps}")

    # Apply filters
    if filters:
        if use_gpu and gpu_caps.get("nvenc_available"):
            # GPU filtering
            filter_str = ",".join(filters)
            command += ["-vf", filter_str]
        else:
            command += ["-vf", ",".join(filters)]
    elif target_fps is not None:
        # Fallback: use -r flag if no filters applied
        command += ["-r", str(target_fps)]
        if preserve_duration:
            # Force frame duplication/dropping to match target duration
            command += ["-video_track_timescale", str(int(target_fps * 1000))]

    # Audio handling
    if no_audio:
        command += ["-an"]
    else:
        command += ["-c:a", "aac", "-b:a", "128k"]

    # Video encoding optimization
    if use_gpu and gpu_caps.get("nvenc_available"):
        # Optimized NVENC settings with quality control
        command += [
            "-c:v",
            "h264_nvenc",
            "-preset",
            q_settings["preset_gpu"],
            "-tune",
            "hq" if quality_level in ["high", "best"] else "ll",
            "-rc",
            "vbr_hq" if quality_level in ["high", "best"] else "vbr",
            "-cq",
            str(q_settings["cq"]),
            "-b:v",
            "0",
            "-maxrate",
            "20M" if quality_level in ["high", "best"] else "10M",
            "-bufsize",
            "40M" if quality_level in ["high", "best"] else "20M",
        ]
    elif use_gpu and gpu_caps.get("qsv_available"):
        # Intel Quick Sync optimization with quality control
        command += [
            "-c:v",
            "h264_qsv",
            "-preset",
            "medium" if quality_level in ["high", "best"] else "veryfast",
            "-global_quality",
            str(q_settings["cq"]),
        ]
    else:
        # CPU encoding with quality control
        command += [
            "-c:v",
            "libx264",
            "-preset",
            q_settings["preset_cpu"],
            "-crf",
            str(q_settings["crf"]),
        ]

        # Add quality-specific x264 options
        if quality_level == "fast":
            command += ["-tune", "fastdecode"]
            command += ["-x264opts", "ref=1:bframes=0:me=dia:subme=1:trellis=0"]
        elif quality_level == "medium":
            # Use defaults (good balance)
            pass
        elif quality_level in ["high", "best"]:
            command += ["-tune", "film"]
            if quality_level == "best":
                command += ["-x264opts", "ref=5:bframes=5:me=umh:subme=8:trellis=2"]
            else:
                command += ["-x264opts", "ref=3:bframes=3:me=hex:subme=6:trellis=1"]

    # Output format optimizations
    command += ["-pix_fmt", "yuv420p"]

    # Container optimization
    if target_ext == ".mp4":
        command += ["-movflags", "+faststart"]

    # Memory and I/O optimization
    command += [
        "-avoid_negative_ts",
        "make_zero",
        "-fflags",
        "+genpts",
        "-max_muxing_queue_size",
        "9999",
    ]

    command.append(output_path)

    # --- run with progress monitoring ---
    try:
        # Log the command for debugging
        logging.debug(f"FFmpeg command: {' '.join(command)}")

        # Run with error capture for better debugging
        process = subprocess.run(command, check=True, capture_output=True, text=True)

        # Verify the output has correct properties
        output_info = VideoReader(video_path=output_path).get_info()
        actual_fps = output_info.get("fps", 0)
        actual_duration = output_info.get("duration", 0)

        if preserve_duration and duration > 0:
            duration_diff = abs(actual_duration - duration)
            if duration_diff > 1.0:  # Allow 1 second tolerance
                logging.warning(
                    f"Duration changed: {duration:.2f}s -> {actual_duration:.2f}s"
                )

        if target_fps and abs(actual_fps - target_fps) > 0.5:
            logging.warning(
                f"FPS not exact: target={target_fps}, actual={actual_fps:.2f}"
            )

        logging.info(
            f"Successfully processed {os.path.basename(input_path)} "
            f"(fps: {source_fps:.1f}->{actual_fps:.1f}, duration: {duration:.1f}s->{actual_duration:.1f}s, "
            f"quality: {quality_level}, gpu={use_gpu})"
        )
        return True

    except subprocess.CalledProcessError as e:
        error_msg = e.stderr if e.stderr else str(e)

        # Check for specific error patterns
        if any(
            pattern in error_msg.lower()
            for pattern in [
                "moov atom not found",
                "invalid data found",
                "width not divisible by 2",
                "height not divisible by 2",
                "padded dimensions cannot be smaller",
            ]
        ):
            logging.warning(f"Format/dimension error: {os.path.basename(input_path)}")

        logging.error(
            f"FFmpeg error processing {os.path.basename(input_path)}: {error_msg}"
        )
        return False
    except Exception as e:
        logging.error(
            f"Error processing {os.path.basename(input_path)}: {traceback.format_exc()}"
        )
        return False


def fourcc_to_str(fourcc: Union[str, int]) -> str:
    if not isinstance(fourcc, int) and not isinstance(fourcc, str):
        raise TypeError("fourcc must be a string or integer")
    if isinstance(fourcc, str):
        return fourcc
    fourcc_bytes = fourcc.to_bytes(4, "little")
    fourcc_str = fourcc_bytes.decode("latin-1")
    return fourcc_str


def video_to_gif(
    input_path: str,
    output_path: str,
    start_time: Optional[float] = None,
    duration: Optional[float] = None,
    fps: Optional[float] = 10,
    scale_width: Optional[int] = None,
) -> bool:
    """
    Converts a video segment to GIF using FFmpeg.

    Args:
        input_path (str): Path to the input video file.
        output_path (str): Path to save the output GIF file.
        start_time (float, optional): Start time in seconds for the segment.
        duration (float, optional): Duration in seconds for the segment.
        fps (float, optional): Frames per second for the GIF. Defaults to 10.
        scale_width (int, optional): Width to scale the GIF. Height is auto-calculated to maintain aspect ratio.

    Returns:
        bool: True if conversion succeeded, False otherwise.
    """
    if os.path.isdir(output_path):
        output_path = os.path.join(output_path, Path(input_path).stem + ".gif")

    command = ["ffmpeg", "-y"]

    if start_time is not None:
        command += ["-ss", str(start_time)]
    if duration is not None:
        command += ["-t", str(duration)]

    command += ["-i", input_path]

    filters = []
    if scale_width is not None:
        filters.append(f"scale={scale_width}:-1")
    if fps is not None:
        filters.append(f"fps={fps}")

    if filters:
        command += ["-vf", ",".join(filters)]

    command += ["-loop", "0", output_path]

    try:
        subprocess.run(command, check=True)
        logging.info(f"Successfully converted video to GIF: {output_path}")
        return True
    except Exception as e:
        logging.error(
            f"Error converting video to GIF {output_path}: {traceback.format_exc()}"
        )
        return False


class VideoReaderInitializationError(Exception):
    pass


class VideoReader:
    def __init__(
        self,
        video_path: Optional[str] = None,
        cap_object: Optional[cv2.VideoCapture] = None,
        validate: bool = True,
    ):
        self.video_path = video_path

        if cap_object is not None and isinstance(cap_object, cv2.VideoCapture):
            if not cap_object.isOpened():
                raise VideoReaderInitializationError(
                    "cv2.VideoCapture object is not opened."
                )
            self.cap = cap_object
        elif video_path is not None:
            if validate and not is_video_valid(video_path):
                raise VideoReaderInitializationError(
                    f"Invalid video file: {video_path}"
                )
            self.cap = cv2.VideoCapture(video_path)
            if not self.cap.isOpened():
                raise VideoReaderInitializationError(
                    f"cv2 could not open video: {video_path}"
                )
        else:
            raise VideoReaderInitializationError(
                "Must provide either video_path or cap_object."
            )

        # metadata
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.duration = self.total_frames / self.fps if self.fps > 0 else 0
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fourcc = int(self.cap.get(cv2.CAP_PROP_FOURCC))
        self.codec = "".join([chr((self.fourcc >> 8 * i) & 0xFF) for i in range(4)])
        self.current_frame_index = -1

    def read_frame(
        self,
        frame_index: Union[int, str],
        jump: bool = True,
    ) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Retrieves the frame at the specified index.

        Args:
            frame_index (Union[int, str]): The 0-based index of the frame to read,
                                        or "next"/"prev"/"current" for relative access.
            jump (bool): If True, updates the internal current_frame_index and moves the pointer.
                        If False, resets pointer after read. Defaults to True.

        Returns:
            Tuple[bool, Optional[np.ndarray]]: Success flag and the frame (or None).
        """

        # Resolve frame_index to int
        # Not currenltly in use, but could be useful
        if frame_index == "init":
            frame_index = -1
            self.current_frame_index = -1
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            return True, None

        if frame_index == "next":
            frame_index = self.current_frame_index + 1
        elif frame_index == "prev":
            frame_index = self.current_frame_index - 1
        elif frame_index == "current":
            frame_index = self.current_frame_index
        if not isinstance(frame_index, int):
            print(
                "Error: frame_index must be an int or one of 'next', 'prev', 'current'."
            )
            return False, None

        # Validate frame_index
        if (
            frame_index < 0
            or frame_index >= self.total_frames
            and self.cap.isOpened()
            and self.cap
        ):
            print(
                f"Error: frame_index {frame_index} is out of bounds 0 - {self.total_frames-1} or video not opened."
            )
            return False, None

        # Save position if peeking
        original_index = (
            int(self.cap.get(cv2.CAP_PROP_POS_FRAMES)) if not jump else None
        )

        try:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ret, frame = self.cap.read()

            if not ret:
                return False, None

            if jump:
                self.current_frame_index = frame_index
            else:
                # Reset to previous position if not jumping
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, original_index)

            return True, frame

        except Exception as e:
            print(f"Error reading frame at index {frame_index}: {e}")
            return False, None

    def read_frames(
        self,
        frame_indices: List[int],
        jump: bool = True,
    ) -> Tuple[bool, List[np.ndarray]]:
        rets, frames = zip(
            *[self.read_frame(frame_index, jump) for frame_index in frame_indices]
        )
        return all(rets), list(frames)

    def get_info(self) -> Dict[str, Union[int, float]]:
        """
        Returns a dictionary containing information about the video.

        Returns:
            Dict[str, Union[int, float]]: A dictionary with video properties like 'total_frames', 'fps',
                  'width', 'height', and 'current_frame_index'.
        """
        return {
            "total_frames": self.total_frames,
            "fps": self.fps,
            "duration": self.duration,
            "width": self.width,
            "height": self.height,
            "aspect_ratio": self.width / self.height,
            "fourcc": self.fourcc,
            "codec": self.codec,
            "current_frame_index": self.current_frame_index,
        }

    def release(self):
        """
        Releases the video capture object. This should be called when you are
        finished with the video to free up resources.
        """
        if self.cap and self.cap.isOpened():
            self.cap.release()
            print("Video capture released.")


class VideoWriter:
    def __init__(
        self,
        output_path: str,
        fps: float,
        frame_size: Tuple[int, int],
        fourcc: Union[str, int] = "mp4v",
    ):
        """
        Initializes the VideoWriter object.
        Args:
            output_path (str): Path to save the output video.
            fps (float): Frames per second of the video.
            frame_size (Tuple[int, int]): Size of the video frames as (width, height).
            fourcc (str, int): FourCC code for the codec (default: "mp4v").
        Raises:
            ValueError: If frame_size is not a tuple of two integers.
        """
        self.output_path = output_path
        self.fps = fps
        self.frame_size = frame_size  # (width, height)
        self.fourcc = cv2.VideoWriter_fourcc(*fourcc_to_str(fourcc))
        self.writer = cv2.VideoWriter(output_path, self.fourcc, fps, frame_size)
        if not self.writer.isOpened():
            raise IOError(f"Could not open video writer for {output_path}")

    def write_frame(self, frame: np.ndarray) -> bool:
        """
        Writes a single frame to the video file.

        Args:
            frame (np.ndarray): The frame image (NumPy array) to write.

        Returns:
            bool: True if the frame was written successfully, False otherwise.
        """

        if get_frame_size(frame) != self.frame_size:
            frame = cv2.resize(
                frame,
                self.frame_size,
                interpolation=cv2.INTER_LINEAR,
            )

        try:
            self.writer.write(frame)
            return True
        except Exception as e:
            print(f"Error writing frame: {e}")
            return False

    def write_frames(self, frames: List[np.ndarray], use_tqdm: bool = True):
        """
        Writes multiple frames to the video file.

        Args:
            frames (List[np.ndarray]): List of frame images (NumPy arrays) to write.
            use_tqdm (bool): Whether to display a progress bar. Defaults to True.
        """
        for frame in tqdm(
            frames,
            desc=f"Writing to {os.path.basename(self.output_path)}",
            unit="frame",
            disable=not use_tqdm,
        ):
            if not self.write_frame(frame):
                print("Stopping write due to error.")
                break

    def release(self):
        """
        Releases the video writer object. This should be called when you are
        finished writing the video to free up resources.
        """
        if self.writer:
            self.writer.release()
            print(f"Video writer released and file saved at: {self.output_path}")

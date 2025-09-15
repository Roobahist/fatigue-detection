import logging
import os
import shutil
import subprocess
import traceback
from pathlib import Path
from typing import Optional

from Zh.src.utils.cpu_gpu import detect_gpu_capability
from Zh.src.utils.video import is_video_valid


def get_video_info(input_path: str) -> dict:
    """Get basic video information using ffprobe."""
    try:
        cmd = [
            "ffprobe",
            "-v",
            "quiet",
            "-print_format",
            "json",
            "-show_format",
            "-show_streams",
            "-select_streams",
            "v:0",
            input_path,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        import json

        data = json.loads(result.stdout)

        video_stream = data["streams"][0] if data["streams"] else {}
        format_info = data.get("format", {})

        return {
            "fps": eval(video_stream.get("r_frame_rate", "0/1")),
            "duration": float(format_info.get("duration", 0)),
            "width": int(video_stream.get("width", 0)),
            "height": int(video_stream.get("height", 0)),
            "codec": video_stream.get("codec_name", "unknown"),
        }
    except Exception as e:
        logging.warning(f"Could not get video info: {e}")
        return {}

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
    output_file_name = Path(output_file_name).stem if output_file_name is not None else Path(input_path).stem
    output_file_name = output_file_name + f"_{target_fps}fps" + target_ext
    output_path = str(Path(output_directory_path, output_file_name).with_suffix(target_ext))

    # Skip if already exists and valid
    if os.path.exists(output_path):
        if is_video_valid(output_path):
            logging.info(f"Video already processed: {os.path.basename(input_path)}")
            return True

    # Get video info for optimization decisions
    video_info = get_video_info(input_path)
    source_fps = video_info.get('fps', 0)
    width = video_info.get('width', 0)
    height = video_info.get('height', 0)
    duration = video_info.get('duration', 0)
    
    # Detect GPU capabilities
    gpu_caps = detect_gpu_capability() if use_gpu else {}

    # Quality settings based on level
    quality_settings = {
        "fast": {"crf": 28, "preset_cpu": "veryfast", "preset_gpu": "p1", "cq": 28},
        "medium": {"crf": 23, "preset_cpu": "fast", "preset_gpu": "p4", "cq": 23}, 
        "high": {"crf": 20, "preset_cpu": "medium", "preset_gpu": "p5", "cq": 20},
        "best": {"crf": 18, "preset_cpu": "slow", "preset_gpu": "p6", "cq": 18}
    }
    
    if quality_level not in quality_settings:
        quality_level = "fast"
        logging.warning(f"Invalid quality_level, using 'fast'")
    
    q_settings = quality_settings[quality_level]

    # --- build optimized command ---
    command = ["ffmpeg", "-y"]
    
    # Input optimization
    if use_gpu and gpu_caps.get('nvenc_available'):
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
            filters.append("scale='2*trunc(min(1920,iw)/2)':'2*trunc(min(1080,ih)/2)':flags=fast_bilinear")
            logging.info(f"Scaling down large video from {width}x{height}")
        else:
            # Ensure even dimensions for all videos (H.264 requirement)
            if width % 2 != 0 or height % 2 != 0:
                filters.append(f"scale={target_width}:{target_height}:flags=fast_bilinear")
                logging.info(f"Adjusting dimensions from {width}x{height} to {target_width}x{target_height}")
    
    # FPS handling - CRITICAL FIX for duration preservation
    if target_fps is not None and source_fps > 0:
        if preserve_duration:
            # Method 1: Use fps filter to resample (preserves duration)
            filters.append(f"fps={target_fps}")
            logging.info(f"Converting FPS from {source_fps} to {target_fps} (duration preserved)")
        else:
            # Method 2: Drop/duplicate frames (changes duration)
            if source_fps > target_fps:
                step = int(source_fps / target_fps)
                if step > 1:
                    filters.append(f"select='not(mod(n,{step}))'")
                    command += ["-vsync", "vfr"]
                    logging.info(f"Dropping frames: {source_fps} -> {target_fps} (duration will change)")
            else:
                filters.append(f"fps={target_fps}")

    # Apply filters
    if filters:
        if use_gpu and gpu_caps.get('nvenc_available'):
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
    if use_gpu and gpu_caps.get('nvenc_available'):
        # Optimized NVENC settings with quality control
        command += [
            "-c:v", "h264_nvenc",
            "-preset", q_settings["preset_gpu"],
            "-tune", "hq" if quality_level in ["high", "best"] else "ll",
            "-rc", "vbr_hq" if quality_level in ["high", "best"] else "vbr",
            "-cq", str(q_settings["cq"]),
            "-b:v", "0",
            "-maxrate", "20M" if quality_level in ["high", "best"] else "10M",
            "-bufsize", "40M" if quality_level in ["high", "best"] else "20M"
        ]
    elif use_gpu and gpu_caps.get('qsv_available'):
        # Intel Quick Sync optimization with quality control
        command += [
            "-c:v", "h264_qsv",
            "-preset", "medium" if quality_level in ["high", "best"] else "veryfast",
            "-global_quality", str(q_settings["cq"])
        ]
    else:
        # CPU encoding with quality control
        command += [
            "-c:v", "libx264",
            "-preset", q_settings["preset_cpu"],
            "-crf", str(q_settings["crf"])
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
    if target_ext == '.mp4':
        command += ["-movflags", "+faststart"]
    
    # Memory and I/O optimization
    command += [
        "-avoid_negative_ts", "make_zero",
        "-fflags", "+genpts",
        "-max_muxing_queue_size", "9999"
    ]

    command.append(output_path)

    # --- run with progress monitoring ---
    try:
        # Log the command for debugging
        logging.debug(f"FFmpeg command: {' '.join(command)}")
        
        # Run with error capture for better debugging
        process = subprocess.run(
            command, 
            check=True,
            capture_output=True,
            text=True
        )
        
        # Verify the output has correct properties
        output_info = get_video_info(output_path)
        actual_fps = output_info.get('fps', 0)
        actual_duration = output_info.get('duration', 0)
        
        if preserve_duration and duration > 0:
            duration_diff = abs(actual_duration - duration)
            if duration_diff > 1.0:  # Allow 1 second tolerance
                logging.warning(f"Duration changed: {duration:.2f}s -> {actual_duration:.2f}s")
        
        if target_fps and abs(actual_fps - target_fps) > 0.5:
            logging.warning(f"FPS not exact: target={target_fps}, actual={actual_fps:.2f}")
        
        logging.info(
            f"Successfully processed {os.path.basename(input_path)} "
            f"(fps: {source_fps:.1f}->{actual_fps:.1f}, duration: {duration:.1f}s->{actual_duration:.1f}s, "
            f"quality: {quality_level}, gpu={use_gpu})"
        )
        return True
        
    except subprocess.CalledProcessError as e:
        error_msg = e.stderr if e.stderr else str(e)
        
        # Check for specific error patterns
        if any(pattern in error_msg.lower() for pattern in [
            "moov atom not found", 
            "invalid data found",
            "width not divisible by 2",
            "height not divisible by 2",
            "padded dimensions cannot be smaller"
        ]):
            logging.warning(f"Format/dimension error: {os.path.basename(input_path)}")
        
        logging.error(f"FFmpeg error processing {os.path.basename(input_path)}: {error_msg}")
        return False
    except Exception as e:
        logging.error(f"Error processing {os.path.basename(input_path)}: {traceback.format_exc()}")
        return False
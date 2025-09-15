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

        # Check for rotation metadata
        rotation = 0
        side_data = video_stream.get('side_data_list', [])
        for side in side_data:
            if side.get('side_data_type') == 'Display Matrix':
                rotation_str = side.get('rotation', '0')
                try:
                    rotation = float(rotation_str)
                except:
                    rotation = 0

        return {
            "fps": eval(video_stream.get("r_frame_rate", "0/1")),
            "duration": float(format_info.get("duration", 0)),
            "width": int(video_stream.get("width", 0)),
            "height": int(video_stream.get("height", 0)),
            "codec": video_stream.get("codec_name", "unknown"),
            "rotation": rotation
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
    rotation = video_info.get('rotation', 0)
    duration = video_info.get('duration', 0)
    
    # Detect GPU capabilities
    gpu_caps = detect_gpu_capability() if use_gpu else {}

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
        needs_padding = (width % 2 != 0) or (height % 2 != 0)
        if needs_padding:
            # Pad to even dimensions
            new_width = width + (width % 2)
            new_height = height + (height % 2)
            filters.append(f"pad={new_width}:{new_height}:(ow-iw)/2:(oh-ih)/2:black")
            logging.info(f"Padding video from {width}x{height} to {new_width}x{new_height}")

    # Handle rotation metadata if present
    if abs(rotation) > 1:  # Only handle significant rotations
        if abs(rotation - 90) < 1:
            filters.append("transpose=1")
        elif abs(rotation - 180) < 1 or abs(rotation + 180) < 1:
            filters.append("transpose=2,transpose=2")
        elif abs(rotation - 270) < 1 or abs(rotation + 90) < 1:
            filters.append("transpose=2")
        logging.info(f"Applying rotation correction: {rotation} degrees")
    
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
    
    # Scale filter optimization for large videos
    if optimize_for_speed and width > 1920 and height > 1080:
        # Ensure even dimensions with fast scaling
        filters.append("scale='2*trunc(min(1920,iw)/2)':'2*trunc(min(1080,ih)/2)':flags=fast_bilinear")
    else:
        # Ensure even dimensions for all videos (H.264 requirement)
        if not any('pad=' in f for f in filters):  # Only add if not already padded
            filters.append("scale='2*trunc(iw/2)':'2*trunc(ih/2)'")

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

    # Remove rotation metadata to prevent conflicts
    command += ["-metadata:s:v", "rotate=0"]

    # Audio handling
    if no_audio:
        command += ["-an"]
    else:
        command += ["-c:a", "aac", "-b:a", "128k"]

    # Video encoding optimization
    if use_gpu and gpu_caps.get('nvenc_available'):
        # Optimized NVENC settings
        command += [
            "-c:v", "h264_nvenc",
            "-preset", "p1" if optimize_for_speed else "p4",
            "-tune", "ll",
            "-rc", "vbr",
            "-cq", "28",
            "-b:v", "0",
            "-maxrate", "10M",
            "-bufsize", "20M"
        ]
    elif use_gpu and gpu_caps.get('qsv_available'):
        # Intel Quick Sync optimization
        command += [
            "-c:v", "h264_qsv",
            "-preset", "veryfast",
            "-global_quality", "28"
        ]
    else:
        # Optimized CPU encoding
        if optimize_for_speed:
            command += [
                "-c:v", "libx264",
                "-preset", "veryfast",
                "-tune", "fastdecode",
                "-crf", "28",
                "-x264opts", "ref=1:bframes=0:me=dia:subme=1:trellis=0"
            ]
        else:
            command += [
                "-c:v", "libx264",
                "-preset", "fast",
                "-crf", "28"
            ]

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
            f"(fps: {source_fps:.1f}->{actual_fps:.1f}, duration: {duration:.1f}s->{actual_duration:.1f}s, gpu={use_gpu})"
        )
        return True
        
    except subprocess.CalledProcessError as e:
        error_msg = e.stderr if e.stderr else str(e)
        
        # Check for specific error patterns
        if any(pattern in error_msg.lower() for pattern in [
            "moov atom not found", 
            "invalid data found",
            "width not divisible by 2",
            "height not divisible by 2"
        ]):
            logging.warning(f"Format/dimension error: {os.path.basename(input_path)}")
        
        logging.error(f"FFmpeg error processing {os.path.basename(input_path)}: {error_msg}")
        return False
    except Exception as e:
        logging.error(f"Error processing {os.path.basename(input_path)}: {traceback.format_exc()}")
        return False
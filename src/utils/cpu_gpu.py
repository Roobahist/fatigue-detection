def setup_tensorflow():
    import tensorflow as tf

    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"TensorFlow: Using GPU: {gpus[0].name}")
        try:
            tf.config.experimental.set_memory_growth(gpus[0], True)
        except RuntimeError as e:
            print(f"TensorFlow: Memory growth setting error: {e}")
    else:
        print("TensorFlow: No GPU found. Using CPU.")


def setup_torch(device = "cuda"):
    import torch
    
    if device == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"PyTorch: Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("PyTorch: No GPU found. Using CPU.")
    return device



def detect_gpu_capability() -> dict:
    """Detect available GPU encoding capabilities."""
    gpu_info = {
        "nvenc_available": False,
        "qsv_available": False,
        "vaapi_available": False,
    }
    import subprocess
    import logging
    
    try:
        # Check for NVENC
        result = subprocess.run(
            ["ffmpeg", "-hide_banner", "-encoders"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        output = result.stdout

        if "h264_nvenc" in output:
            gpu_info["nvenc_available"] = True
        if "h264_qsv" in output:
            gpu_info["qsv_available"] = True
        if "h264_vaapi" in output:
            gpu_info["vaapi_available"] = True

    except Exception as e:
        logging.warning(f"Could not detect GPU capabilities: {e}")

    return gpu_info
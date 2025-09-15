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


def setup_torch():
    import torch
    
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"PyTorch: Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("PyTorch: No GPU found. Using CPU.")
    return device
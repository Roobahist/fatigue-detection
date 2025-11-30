from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np


def set_seed(seed=42):
    import os
    import random

    import numpy as np
    import tensorflow as tf

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    tf.config.experimental.enable_op_determinism()


def flatten(lst) -> list:
    flat = []
    for item in lst:
        if isinstance(item, list):
            flat.extend(flatten(item))
        else:
            flat.append(item)
    return flat


def order_dict_by_key_list(d, key_order):
    return {k: d[k] for k in key_order if k in d} | {
        k: d[k] for k in d if k not in key_order
    }


def pattern(lst):
    lookup = {}
    return [lookup.setdefault(val, len(lookup)) for val in lst]


def matches_pattern(list1, list2):
    # a = [1, 2, 3, 3, 2, 1]
    # b = ['x', 'y', 'z', 'z', 'y', 'x']
    # c = ['x', 'y', 'y', 'y', 'y', 'y']
    # print(matches_pattern(a, b))  # True
    # print(matches_pattern(a, c))  # False

    return pattern(list1) == pattern(list2)


def has_multiple_mappings(list1, list2):
    """
    Checks if any unique element in list2 corresponds to more than one unique element in list1.

    Args:
        list1 (list): The first list.
        list2 (list): The second list.

    Returns:
        bool: True if any unique element in list2 maps to more than one unique element in list1, False otherwise.

    list1 = [1, 2, 3, 3, 2, 1]
    list2 = ['x', 'y', 'z', 'z', 'y', 'x']
    print(has_multiple_mappings(list1, list2))  # False

    list1 = [1, 2, 3, 4, 2, 1]
    list2 = ['x', 'y', 'z', 'z', 'y', 'x']
    print(has_multiple_mappings(list1, list2))  # True
    """
    mapping = {}
    for a, b in zip(list1, list2):
        if b not in mapping:
            mapping[b] = set()
        mapping[b].add(a)

    # Check if any element in list2 maps to more than one unique element in list1
    return any(len(values) > 1 for values in mapping.values())


def save_npz(path: Union[str, Path], arrays: Dict[str, Any]) -> None:
    """
    Save arbitrary arrays or Python scalars into an .npz file.

    The `arrays` dict values must be convertible to NumPy arrays.
    """
    converted = {k: np.asarray(v) for k, v in arrays.items()}
    np.savez(path, **converted)


def load_npz(path: Union[str, Path]) -> Dict[str, np.ndarray]:
    """
    Load an .npz file and return a dict of NumPy arrays.
    """
    loaded = np.load(path, allow_pickle=False)
    return {k: loaded[k] for k in loaded.files}


def save_obj(obj, output_path, file_type):
    """
    Save an object to a file using pickle.

    Args:
        obj (Any): The object to be pickled.
        output_path (str): The path where the pickled file will be saved.
    """
    if file_type == "pickle":
        import pickle

        with open(output_path, "wb") as file:
            pickle.dump(obj, file)

    elif file_type == "json":
        import json

        with open(output_path, "w") as f:
            json.dump(obj, f, indent=4)


def get_frame_size(frame) -> tuple[int, int]:
    """
    Returns the (width, height) of a frame,
    formatted for cv2.VideoWriter.
    """
    h, w = frame.shape[:2]
    return (w, h)


def read_obj(input_path, file_type: Optional[str] = None):
    """
    Load an object from a file using pickle.

    Args:
        input_path (str): The path to the pickled file.
    """

    if input_path.endswith(".json"):
        file_type = "json"
    elif input_path.endswith(".pkl") or input_path.endswith(".pickle"):
        file_type = "pickle"

    if file_type == "pickle":
        import pickle

        with open(input_path, "rb") as file:
            obj = pickle.load(file)
        return obj

    if file_type == "json":
        import json

        with open(input_path, "r") as f:
            obj = json.load(f)
        return obj


def update_file_name(
    file_path: str,
    input_folder: str,
    output_folder: str,
    remove_suffix: str = "",
    suffix: str = "",
    remove_prefix: str = "",
    prefix: str = "",
    extension: str = "",
):
    import os

    relative_path = os.path.relpath(file_path, input_folder)
    base_name, ext = os.path.splitext(relative_path)
    if remove_suffix and base_name.endswith(remove_suffix):
        base_name = base_name[: -len(remove_suffix)]
    if remove_prefix and base_name.startswith(remove_prefix):
        base_name = base_name[len(remove_prefix) :]
    new_file_name = f"{prefix}{base_name}{suffix}{extension}"
    return os.path.join(output_folder, new_file_name)

import json
import os
import pickle
import random
from pathlib import Path
from typing import IO, Any, Callable, Dict, Optional, Union

import numpy as np


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def set_seed(seed=42):
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


def _atomic_write(path: str, mode: str, write_fn: Callable[[IO], None]) -> None:
    """
    Write to a temporary file and atomically replace the target file.

    `write_fn` is responsible for writing the content to the open file handle.
    """
    tmp_path = path + ".tmp"
    with open(tmp_path, mode) as f:
        write_fn(f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_path, path)


def save_obj(obj: Any, output_path: str, file_type: str, atomic: bool = False) -> None:
    """
    Save an object to a file in either pickle or JSON format.

    Args:
        obj: The object to be serialized.
        output_path: The path where the file will be saved.
        file_type: Serialization format, either "pickle" or "json".
        atomic: If True, perform an atomic write via a temporary file.
    """
    if file_type == "pickle":
        if atomic:
            _atomic_write(
                output_path,
                "wb",
                lambda f: pickle.dump(obj, f),
            )
        else:
            with open(output_path, "wb") as f:
                pickle.dump(obj, f)

    elif file_type == "json":
        if atomic:
            _atomic_write(
                output_path,
                "w",
                lambda f: json.dump(obj, f, indent=4, cls=NumpyEncoder),
            )
        else:
            with open(output_path, "w") as f:
                json.dump(obj, f, indent=4, cls=NumpyEncoder)

    else:
        raise ValueError(f"Unsupported file_type: {file_type!r}")


def get_frame_size(frame) -> tuple[int, int]:
    """
    Returns the (width, height) of a frame,
    formatted for cv2.VideoWriter.
    """
    h, w = frame.shape[:2]
    return (w, h)


def load_obj(input_path, file_type: Optional[str] = None):
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

        with open(input_path, "rb") as file:
            obj = pickle.load(file)
        return obj

    if file_type == "json":

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
    relative_path = os.path.relpath(file_path, input_folder)
    base_name, ext = os.path.splitext(relative_path)
    if remove_suffix and base_name.endswith(remove_suffix):
        base_name = base_name[: -len(remove_suffix)]
    if remove_prefix and base_name.startswith(remove_prefix):
        base_name = base_name[len(remove_prefix) :]
    new_file_name = f"{prefix}{base_name}{suffix}{extension}"
    return os.path.join(output_folder, new_file_name)

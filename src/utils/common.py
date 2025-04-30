from typing import List, Union


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
    return {k: d[k] for k in key_order if k in d} | {k: d[k] for k in d if k not in key_order}


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


def get_csv_length(file_path: str, skip_header: bool = False):
    import csv
    import os

    assert os.path.exists(file_path), "File does not exist."
    assert file_path.lower().endswith('.csv'), "This is not a CSV file."

    with open(file_path, 'r', encoding='utf-8') as file:
        reader = csv.reader(file)
        if skip_header:
            next(reader, None)  # Skip header
        row_count = sum(1 for row in reader if row)

    return row_count


def find_new_file_name(output_directory_path: str, 
                       file_name_prefix: str, 
                       file_name_extension: Union[str, None] = None) -> str:
    """
    Finds a new file name in the specified directory by appending a numeric suffix to the file name prefix.

    Args:
        output_directory_path (str): The directory where the file will be saved.
        file_name_prefix (str): The prefix of the file name.
        file_name_extension (str | None): The file extension (e.g., 'txt', 'csv'). If None, no extension is added.

    Returns:
        str: A new file name with a numeric suffix that does not already exist in the directory.

    Example:
        find_new_file_name('/path/to/dir', 'file', 'txt')
        # Returns something like 'file.txt' 'file_1.txt' if 'file_1.txt' does not exist.
    """
    import os

    from Zh.src.utils.file_folder import create_folders

    # Ensure the directory exists
    create_folders(output_directory_path)

    # Trying without counter 
    file_name = f"{file_name_prefix}.{file_name_extension}" if file_name_extension else file_name_prefix
    file_path = os.path.join(output_directory_path, file_name)
    if not os.path.exists(file_path):
        return file_name

    # Initialize the suffix counter
    counter = 1
    # Construct the file name with the optional extension
    while True:
        file_name = f"{file_name_prefix}_{counter}"
        if file_name_extension:
            file_name += f".{file_name_extension}"
        file_path = os.path.join(output_directory_path, file_name)

        # Check if the file already exists
        if not os.path.exists(file_path):
            return file_name
        counter += 1


def pop_row(df, idx, reset_index=False):
    """
    Pops one or more rows from a DataFrame based on index or slice.

    Parameters:
    - df: pandas DataFrame
    - idx: int, label, list of labels/ints, or slice
    - reset_index: bool, whether to reset the index of the returned DataFrame

    Returns:
    - popped_rows: pd.Series or pd.DataFrame (depending on the input)
    - updated_df: pd.DataFrame (without the popped row(s))
    """
    import pandas as pd
    
    if isinstance(idx, slice):
        popped = df.iloc[idx]
        remaining = df.drop(popped.index)
    elif isinstance(idx, (int, str)):
        popped = df.loc[idx]
        remaining = df.drop(idx)
    elif isinstance(idx, (list, tuple, pd.Index)):
        popped = df.loc[idx]
        remaining = df.drop(idx)
    else:
        raise ValueError("Unsupported index type: must be int, str, slice, or list")

    if reset_index:
        if isinstance(popped, pd.Series):
            popped = popped.to_frame().T.reset_index(drop=True).squeeze()
        else:
            popped = popped.reset_index(drop=True)
        remaining = remaining.reset_index(drop=True)

    return popped, remaining


def merge_intervals(intervals):
    intervals = sorted(intervals, key=lambda x: x[0])
    
    merged = []
    for interval in intervals:
        if not merged:
            merged.append(interval)
        else:
            last = merged[-1]
            if interval[0] < last[1]:  # real intersection (not just touching)
                print(f"Warning: intervals {last} and {interval} overlap!")
                merged[-1] = (last[0], max(last[1], interval[1]))
            elif interval[0] == last[1]:  # touching, safe to merge
                merged[-1] = (last[0], interval[1])
            else:
                merged.append(interval)
    return merged


def timeit(func):
    from functools import wraps
    import time

    @wraps(func)
    def timeit_wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        total_time = end_time - start_time
        print(f'Function {func.__name__}{args} {kwargs} Took {total_time:.4f} seconds')
        return result
    
    return timeit_wrapper
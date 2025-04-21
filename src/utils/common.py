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
    import os
    import csv

    assert os.path.exists(file_path), "File does not exist."
    assert file_path.lower().endswith('.csv'), "This is not a CSV file."

    with open(file_path, 'r', encoding='utf-8') as file:
        reader = csv.reader(file)
        if skip_header:
            next(reader, None)  # Skip header
        row_count = sum(1 for row in reader if row)

    return row_count
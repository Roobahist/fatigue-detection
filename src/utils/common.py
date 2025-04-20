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
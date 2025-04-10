import logging
import os

import pandas as pd

from Zh.experiments.e7.main import *
from Zh.src.utils.load_data import load_data

if __name__ == "__main__":
    logging.info("Starting")

    # path_to_data = "/work/long_lab/Mojtaba/data/UTA/1.3.2-augmented_embedded"
    path_to_data = "/work/long_lab/Mojtaba/data/UTA/1.3.3-preprocessed"
    mapping = {"10": 8.5, "5": 6.5, "0": 2}

    set_seed(42)
    # unpreprocessed_data = load_data(path_to_data=path_to_data, file_names=[])
    preprocessed_data = load_data(path_to_data=path_to_data, file_names=[])
    # preprocessed_data, columns_to_drop = preprocess_data(data=unpreprocessed_data, mapping=mapping)
    columns_to_drop = [
        "file_name",
        "type",
        "frame",
        "fold",
        "subject_id",
    ]

    m = Model(
        units=[50, 30],
        dropout_ratio=0.3,
        kernel_regularizer_l2=0,
        kernel_regularizer_l1=0,
        batch_size=32,
        num_classes=3,
        learning_rate=1e-3,
        weight_decay=1e-4,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-7,
        epochs=200,
        patience=100,
        max_slices=5000,
        mapping=mapping
    )

    m.test_model(
        data=preprocessed_data,
        columns_to_drop=columns_to_drop,
        parallel=True,
        max_workers=15,
        maxtasksperchild=1,
        use_tqdm=True,
        output_parent_dir="/home/mojtaba.kananisarche/Zh/outputs",
    )

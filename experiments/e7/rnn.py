import logging
import os
import sys

import pandas as pd

from Zh.experiments.e7.main import *
from Zh.src.utils.load_data import load_data

if __name__ == "__main__":
    logging.info("Starting ...")
    working_directory = sys.argv[1]
    job_id            = sys.argv[2]
    max_workers       = int(sys.argv[3])
    fold_start_index  = int(sys.argv[4])
    fold_end_index    = int(sys.argv[5])
    mapping           = {"10": 8.5, "5": 6.5, "0": 2}

    logging.info(f"working_directory : {working_directory} & job_id : {job_id}")
    # logging.info("Setting Up Environment ...")
    # os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0"
    # tf.get_logger().propagate = False
    # tf_logger = tf.get_logger()
    # tf_logger.setLevel(logging.INFO)  # Or DEBUG, WARNING etc.
    # tf_handler = logging.FileHandler(os.path.join(working_directory, f"tensorflow-{job_id}.log"))
    # tf_handler.setLevel(logging.INFO)
    # formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    # tf_handler.setFormatter(formatter)
    # tf_logger.handlers = []
    # tf_logger.addHandler(tf_handler)

    logging.info("Loading Data ...")
    # path_to_data = "/work/long_lab/Mojtaba/data/UTA/1.3.2-augmented_embedded"
    path_to_data = "/work/long_lab/Mojtaba/data/UTA/1.3.3-preprocessed"
    set_seed(42)
    # unpreprocessed_data = load_data(path_to_data=path_to_data, file_names=[])
    preprocessed_data = load_data(
        path_to_data=path_to_data,
        # file_names=["facenet_preprocessed_alternativemapping.csv"],
        file_names=["facenet_preprocessed_alternativemapping_croppedonly.csv"],
        # nrows=1000000,
    )
    # preprocessed_data, columns_to_drop = preprocess_data(data=unpreprocessed_data, mapping=mapping)

    logging.info("Creating Model ...")
    columns_to_drop = [
        "file_name",
        "file_name_1",
        "type",
        "frame",
        "fold",
        "subject_id",
    ]

    m = Model(
        units                   = [50, 50],
        dropout_ratio           = 0.3,
        kernel_regularizer_l2   = 0,
        kernel_regularizer_l1   = 0,
        batch_size              = 32,
        learning_rate           = 1e-3,
        weight_decay            = 1e-4,
        beta_1                  = 0.9,
        beta_2                  = 0.999,
        epsilon                 = 1e-7,
        epochs                  = 150,
        patience                = 50,
        slice_size              = 4500,
        step_size               = 3000,
        discard_threshold_ratio = 0.7,
        mapping                 = mapping,
    )

    logging.info("Testing Model ...")
    m.test_model(
        data              = preprocessed_data,
        columns_to_drop   = columns_to_drop,
        parallel          = True,
        max_workers       = max_workers,
        maxtasksperchild  = 1,
        use_tqdm          = True,
        output_parent_dir = "/home/mojtaba.kananisarche/Zh/outputs",
        fold_start_index  = fold_start_index,
        fold_end_index    = fold_end_index,
    )

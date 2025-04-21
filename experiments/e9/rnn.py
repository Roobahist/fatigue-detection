import logging
import sys

from Zh.experiments.e9.main import *
from Zh.src.utils.load_data import load_data
from Zh.src.utils.common import set_seed

if __name__ == "__main__":
    logging.info("Starting ...")
    max_workers       = int(sys.argv[1]) if len(sys.argv) > 1 else None
    working_directory = sys.argv[2] if len(sys.argv) > 2 else None
    job_id            = sys.argv[3] if len(sys.argv) > 3 else None
    fold_start_index  = int(sys.argv[4]) if len(sys.argv) > 4 else None
    fold_end_index    = int(sys.argv[5]) if len(sys.argv) > 5 else None
    mapping           = {"10": 8.5, "5": 6.5, "0": 2}

    logging.info(f"working_directory : {working_directory} & job_id : {job_id}")

    logging.info("Loading Data ...")
    # path_to_data = "/work/long_lab/Mojtaba/data/UTA/1.3.2-augmented_embedded"
    path_to_data = "/work/long_lab/Mojtaba/data/UTA/1.3.3-preprocessed"
    preprocessed_data = load_data(
        input_path=path_to_data,
        # file_names=["facenet_preprocessed_alternativemapping.csv"],
        file_names=["facenet_preprocessed_alternativemapping_croppedonly.csv"],
        # nrows=100000,
    )

    # unpreprocessed_data = load_data(path_to_data=path_to_data, file_names=[])
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

    set_seed(42)

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
        epochs                  = 250,
        patience                = 50,
        slice_size              = 4500,
        step_size               = 2000,
        discard_threshold_ratio = 0.7,
        mapping                 = mapping,
    )

    logging.info("Testing Model ...")
    m.test_model(
        data                          = preprocessed_data,
        columns_to_drop               = columns_to_drop,
        parallel                      = True,
        max_workers                   = max_workers,
        maxtasksperchild              = 1,
        use_tqdm                      = False,
        output_parent_dir             = "/home/mojtaba.kananisarche/Zh/outputs",
        fold_by_primary_column_name   = "fold",
        fold_by_secondary_column_name = None,
        fold_start_index              = fold_start_index,
        fold_end_index                = fold_end_index,
        fold_indices                  = []
    )

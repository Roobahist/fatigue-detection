import logging
import sys

from Zh.experiments.e9.main import *
from Zh.src.utils.common import set_seed

if __name__ == "__main__":
    logging.info("Starting ...")
    max_workers = int(sys.argv[1]) if len(sys.argv) > 1 else None
    working_directory = sys.argv[2] if len(sys.argv) > 2 else None
    job_id = sys.argv[3] if len(sys.argv) > 3 else None
    fold_start_index = int(sys.argv[4]) if len(sys.argv) > 4 else None
    fold_end_index = int(sys.argv[5]) if len(sys.argv) > 5 else None

    logging.info(f"working_directory : {working_directory} & job_id : {job_id}")

    input_directory_path = "/work/long_lab/Mojtaba/data/UTA/1.3.4-augmented_embedded_preprocessed_normalized_npy"

    mapping = {"10": 8.5, "5": 6.5, "0": 2}

    columns_to_drop = [
        "file_name",
        "file_name_1",
        "type",
        "frame",
        "fold",
        "subject_id",
    ]

    hp_config = {
        "num_lstm_layers": {"min_value": 1, "max_value": 2, "step": 1},
        "units": {"min_value": 10, "max_value": 30, "step": 10},
        "activation": "tanh",
        "l1_reg": {"min_value": 1e-6, "max_value": 1e-2, "sampling": "log"},
        "l2_reg": {"min_value": 1e-6, "max_value": 1e-2, "sampling": "log"},
        "dropout": {"min_value": 0.4, "max_value": 0.7, "step": 0.1},
        "use_batch_norm": True,
        "output": {"units": 1, "activation": "linear"},
        "learning_rate": {"min_value": 1e-4, "max_value": 1e-2, "sampling": "log"},
        "weight_decay": {"min_value": 1e-6, "max_value": 1e-2, "sampling": "log"},
        "beta_1": {"min_value": 0.8, "max_value": 0.95, "step": 0.05},
        "beta_2": {"min_value": 0.9, "max_value": 0.999, "step": 0.033},
        "epsilon": 1e-7,
        "amsgrad": True,
    }

    set_seed(42)

    logging.info("Creating Model ...")

    m = Model(
        batch_size=16,
        epochs=250,
        patience=100,
        slice_size=9000,
        step_size=1000,
        discard_threshold_ratio=0.8,
        mapping=mapping,
    )

    logging.info("Testing Model ...")
    m.test_model(
        input_directory_path=input_directory_path,
        split_info_csv_path=os.path.join(input_directory_path, "split_info.json"),
        column_names_json_path=os.path.join(input_directory_path, "columns.json"),
        columns_to_drop=columns_to_drop,
        parallel=True,
        max_workers=max_workers,
        maxtasksperchild=1,
        use_tqdm=False,
        output_directory_path="/home/mojtaba.kananisarche/Zh/outputs",
        output_folder_name=None,
        fold_by_primary_column_name="fold",
        fold_by_secondary_column_name=None,
        fold_start_index=fold_start_index,
        fold_end_index=fold_end_index,
        fold_indices=[0],
        data_gen_workers=10,
        tuner_max_trials=20,
        tuner_executions_per_trial=1,
        tuner_epochs=50,
        tuner_patience=25,
        tuner_model_name="LSTM",
        tuner_type="bayesian",
        loss_function="mse",
        metrics=["mae"],
        # hp_config                     = hp_config,
    )

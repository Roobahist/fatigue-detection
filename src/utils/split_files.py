import os
from typing import Dict, List, Optional

import pandas as pd

from Zh.src.utils.file_folder import get_file_paths


def _get_train_val_test_file_paths(
    input_path: str,
    output_path: str,
    primary_column_name: str,
    secondary_column_name: Optional[str] = None,
) -> List[Dict]:

    file_paths = get_file_paths(input_path=input_path, depth=2, EXT_LIST=[".csv"])
    data = pd.DataFrame()
    data["path"] = file_paths
    data = data[
        ~data["path"].apply(lambda x: os.path.basename(x).startswith("video_info"))
    ]

    data["fold"] = (
        data["path"]
        .apply(lambda x: os.path.basename(x).split("_")[0])
        .astype(int)
        .astype(str)
    )
    data["subject_id"] = (
        data["path"]
        .apply(lambda x: os.path.basename(x).split("_")[1])
        .astype(int)
        .astype(str)
    )
    data = data.sort_values(["fold", "subject_id"]).reset_index(drop=True)

    columns_to_fold_by = data[
        [primary_column_name]
        + ([secondary_column_name] if secondary_column_name else [])
    ]

    train_val_test_file_paths = []
    unique_main_column_values = sorted(columns_to_fold_by[primary_column_name].unique())

    index = 0

    if primary_column_name and secondary_column_name:
        for main_column_value in unique_main_column_values:
            corresponding_secondary_value = columns_to_fold_by[
                columns_to_fold_by[primary_column_name] == main_column_value
            ][secondary_column_name].unique()
            assert (
                len(corresponding_secondary_value) == 1
            ), f"Multiple Unique Secondary Values Found For Main Column Value {main_column_value} - {corresponding_secondary_value}"
            corresponding_secondary_value = corresponding_secondary_value[0]

            fold_train_val_test_file_paths = {
                "index": index,
                "info": {},
                "input_paths": {},
                "output_paths": {},
            }
            index += 1

            test_split = columns_to_fold_by[
                columns_to_fold_by[primary_column_name] == main_column_value
            ]
            fold_train_val_test_file_paths["input_paths"]["test"] = data.loc[
                test_split.index.to_list()
            ]["path"].to_list()
            fold_train_val_test_file_paths["info"]["test"] = {
                f"{primary_column_name}": sorted(
                    test_split[primary_column_name].unique().tolist()
                )
            }

            val_split = columns_to_fold_by[
                (columns_to_fold_by[primary_column_name] != main_column_value)
                & (
                    columns_to_fold_by[secondary_column_name]
                    == corresponding_secondary_value
                )
            ]
            fold_train_val_test_file_paths["input_paths"]["val"] = data.loc[
                val_split.index.to_list()
            ]["path"].to_list()
            fold_train_val_test_file_paths["info"]["val"] = {
                f"{primary_column_name}": sorted(
                    val_split[primary_column_name].unique().tolist()
                )
            }

            train_split = columns_to_fold_by[
                columns_to_fold_by[secondary_column_name]
                != corresponding_secondary_value
            ]
            fold_train_val_test_file_paths["input_paths"]["train"] = data.loc[
                train_split.index.to_list()
            ]["path"].to_list()
            fold_train_val_test_file_paths["info"]["train"] = {
                f"{primary_column_name}": sorted(
                    train_split[primary_column_name].unique().tolist()
                )
            }
            train_val_test_file_paths.append(fold_train_val_test_file_paths)

    elif primary_column_name:
        for main_column_value_i in unique_main_column_values:
            for main_column_value_j in unique_main_column_values:
                if main_column_value_i != main_column_value_j:
                    fold_train_val_test_file_paths = {
                        "index": index,
                        "info": {},
                        "input_paths": {},
                        "output_paths": {},
                    }
                    index += 1

                    test_split = columns_to_fold_by[
                        columns_to_fold_by[primary_column_name] == main_column_value_i
                    ]
                    fold_train_val_test_file_paths["input_paths"]["test"] = data.loc[
                        test_split.index.to_list()
                    ]["path"].to_list()
                    fold_train_val_test_file_paths["info"]["test"] = {
                        f"{primary_column_name}": sorted(
                            test_split[primary_column_name].unique().tolist()
                        )
                    }

                    val_split = columns_to_fold_by[
                        columns_to_fold_by[primary_column_name] == main_column_value_j
                    ]
                    fold_train_val_test_file_paths["input_paths"]["val"] = data.loc[
                        val_split.index.to_list()
                    ]["path"].to_list()
                    fold_train_val_test_file_paths["info"]["val"] = {
                        f"{primary_column_name}": sorted(
                            val_split[primary_column_name].unique().tolist()
                        )
                    }

                    train_split = columns_to_fold_by[
                        ~columns_to_fold_by[primary_column_name].isin(
                            [main_column_value_i, main_column_value_j]
                        )
                    ]
                    fold_train_val_test_file_paths["input_paths"]["train"] = data.loc[
                        train_split.index.to_list()
                    ]["path"].to_list()
                    fold_train_val_test_file_paths["info"]["train"] = {
                        f"{primary_column_name}": sorted(
                            train_split[primary_column_name].unique().tolist()
                        )
                    }

                    train_val_test_file_paths.append(fold_train_val_test_file_paths)

    for train_val_test_file_path in train_val_test_file_paths:

        train_val_test_file_path["output_paths"]["test"] = [
            os.path.join(
                output_path,
                str(train_val_test_file_path["index"]),
                path.replace(input_path + "/", ""),
            )
            for path in train_val_test_file_path["input_paths"]["test"]
        ]

        train_val_test_file_path["output_paths"]["val"] = [
            os.path.join(
                output_path,
                str(train_val_test_file_path["index"]),
                path.replace(input_path + "/", ""),
            )
            for path in train_val_test_file_path["input_paths"]["val"]
        ]

        train_val_test_file_path["output_paths"]["train"] = [
            os.path.join(
                output_path,
                str(train_val_test_file_path["index"]),
                path.replace(input_path + "/", ""),
            )
            for path in train_val_test_file_path["input_paths"]["train"]
        ]

    return train_val_test_file_paths

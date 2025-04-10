import os
from typing import List, Optional, Tuple

import pandas as pd

#e7
def preprocess_data(
    data: pd.DataFrame, mapping: Optional[dict] = None
) -> Tuple[pd.DataFrame, List[str]]:
    # Ensure the 'file_name' column exists
    if "file_name" not in data.columns:
        raise ValueError("The data must contain a 'file_name' column.")

    # Add the 'frame' column
    data["frame"] = data.groupby("file_name").cumcount()

    # Split 'file_name' to extract 'fold', 'subject_id', 'target', and 'type'
    split_columns = data["file_name"].str.split("_", expand=True)
    data["fold"] = split_columns[0].astype(int)
    data["subject_id"] = split_columns[1].astype(int).astype(str)
    data["target"] = split_columns[2]
    data["type"] = split_columns[4].str.replace(".csv", "", regex=False)

    # Handle the 'target' column
    if mapping:
        unique_targets = data["target"].unique()
        if not set(unique_targets).issubset(set(mapping.keys())):
            raise ValueError("Mapping does not include all unique target values.")
        data["target"] = data["target"].map(mapping).astype(float)
    else:
        data["target"] = data["target"].astype(float)

    # Reorder columns
    columns_order = ["file_name", "type", "frame", "fold", "subject_id", "target"]
    remaining_columns = [col for col in data.columns if col not in columns_order]
    data = data[columns_order + remaining_columns]
    columns_to_drop = [
        "file_name",
        "type",
        "frame",
        "fold",
        "subject_id",
    ]

    return data, columns_to_drop

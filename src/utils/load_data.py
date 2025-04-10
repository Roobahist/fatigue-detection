import os
from typing import List, Optional

import pandas as pd
import logging
# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def load_data(path_to_data: str, file_names: Optional[List[str]] = []):
    data = pd.DataFrame()
    headers = None
    is_numeric_headers = None

    # Check if the path is a file
    if os.path.isfile(path_to_data):
        if path_to_data.endswith(".csv") and (
            not file_names or os.path.basename(path_to_data) in file_names
        ):
            temp_data = pd.read_csv(path_to_data, header=None)
            headers = (
                temp_data.iloc[0].tolist()
                if temp_data.iloc[0].dtype == "object"
                else None
            )
            is_numeric_headers = headers is None
            if is_numeric_headers:
                temp_data.columns = [f"f_{i}" for i in range(temp_data.shape[1])]
            else:
                temp_data = pd.read_csv(path_to_data)
            temp_data["file_name"] = os.path.basename(path_to_data)
            data = temp_data
            logging.info(f"Loaded file: {path_to_data}, data shape: {data.shape}")
    # If the path is a directory
    elif os.path.isdir(path_to_data):
        for root, _, files in os.walk(path_to_data):
            for file in files:
                if file.endswith(".csv") and (not file_names or file in file_names):
                    file_path = os.path.join(root, file)
                    temp_data = pd.read_csv(file_path, header=None)
                    temp_headers = (
                        temp_data.iloc[0].tolist()
                        if temp_data.iloc[0].dtype == "object"
                        else None
                    )
                    temp_is_numeric_headers = temp_headers is None

                    if headers is None:
                        headers = temp_headers
                        is_numeric_headers = temp_is_numeric_headers
                        if is_numeric_headers:
                            temp_data.columns = [
                                f"f_{i}" for i in range(temp_data.shape[1])
                            ]
                        else:
                            temp_data = pd.read_csv(file_path)
                    else:
                        if is_numeric_headers != temp_is_numeric_headers:
                            raise ValueError(
                                "Mixed header conditions detected. Some files have headers while others do not."
                            )
                        if is_numeric_headers:
                            if temp_data.shape[1] != len(headers):
                                raise ValueError(
                                    "Files do not have the same number of columns."
                                )
                            temp_data.columns = [
                                f"f_{i}" for i in range(temp_data.shape[1])
                            ]
                        else:
                            if temp_headers != headers:
                                raise ValueError("Files do not have the same headers.")
                            temp_data = pd.read_csv(file_path)

                    temp_data["file_name"] = file  # Add file name as a column
                    data = pd.concat([data, temp_data], ignore_index=True)
                    logging.info(f"Concatenated file: {file}, current data shape: {data.shape}")
    else:
        raise ValueError(f"Invalid path: {path_to_data}")

    return data

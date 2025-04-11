import logging
import os
from typing import List, Optional, Tuple

import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def extract_headers(file_path: str) -> List[str]:
    """Extract headers from the first row of the file."""
    first_row = pd.read_csv(file_path, nrows=1, header=None)
    if first_row.iloc[0].dtype == "object":
        return first_row.iloc[0].tolist()
    return [f"f_{i}" for i in range(first_row.shape[1])]


def find_identifier_column_name(
    column_names: List[str], identifier_column_prefix: Optional[str] = "file_name"
) -> str:
    identifier_column_name = identifier_column_prefix
    i = 1
    while identifier_column_name in column_names:
        identifier_column_name = identifier_column_prefix + f"_{i}"
        i += 1
    return identifier_column_name


def load_single_file(
    file_path: str,
    headers: Optional[List[str]],
    is_numeric_headers: bool,
    nrows: Optional[int],
    skiprows: Optional[int] = 0,
    identifier_column_prefix: Optional[str] = "file_name",
) -> Tuple[pd.DataFrame, str]:
    """Load a single CSV file into a DataFrame."""
    temp_data = pd.read_csv(
        file_path,
        nrows=nrows,
        skiprows=skiprows if is_numeric_headers or skiprows != 0 else 1,
        header=None,
    )
    if is_numeric_headers:
        temp_data.columns = [f"f_{i}" for i in range(temp_data.shape[1])]
    else:
        temp_data.columns = headers
        temp_data = temp_data.iloc[1:].reset_index(drop=True)
    identifier_column_name = find_identifier_column_name(
        column_names=temp_data.columns,
        identifier_column_prefix=identifier_column_prefix,
    )
    temp_data[identifier_column_name] = os.path.basename(file_path)
    return temp_data, identifier_column_name


def validate_headers(
    headers: Optional[List[str]],
    temp_headers: List[str],
    is_numeric_headers: bool,
    temp_is_numeric_headers: bool,
):
    """Validate headers across multiple files."""
    if headers is None:
        return temp_headers, temp_is_numeric_headers
    if is_numeric_headers != temp_is_numeric_headers:
        raise ValueError(
            "Mixed header conditions detected. Some files have headers while others do not."
        )
    if not is_numeric_headers and headers != temp_headers:
        raise ValueError("Files do not have the same headers.")
    return headers, is_numeric_headers


def load_data(
    path_to_data: str,
    file_names: Optional[List[str]] = None,
    nrows: Optional[int] = None,
    skiprows: Optional[int] = 0,
    identifier_column_prefix: Optional[str] = "file_name",
) -> pd.DataFrame:
    """Load data from a file or directory of CSV files."""
    file_names = file_names or []
    data = pd.DataFrame()
    headers, is_numeric_headers = None, None

    if os.path.isfile(path_to_data):
        if path_to_data.endswith(".csv") and (
            not file_names or os.path.basename(path_to_data) in file_names
        ):
            headers = extract_headers(path_to_data)
            is_numeric_headers = headers is None
            data, identifier_column_name = load_single_file(
                path_to_data,
                headers,
                is_numeric_headers,
                nrows,
                skiprows,
                identifier_column_prefix,
            )
            logging.info(f"Loaded file: {path_to_data}, data shape: {data.shape}")
    elif os.path.isdir(path_to_data):
        for root, _, files in os.walk(path_to_data):
            for file in files:
                if file.endswith(".csv") and (not file_names or file in file_names):
                    file_path = os.path.join(root, file)
                    temp_headers = extract_headers(file_path)
                    temp_is_numeric_headers = temp_headers is None
                    headers, is_numeric_headers = validate_headers(
                        headers,
                        temp_headers,
                        is_numeric_headers,
                        temp_is_numeric_headers,
                    )
                    temp_data, identifier_column_name = load_single_file(
                        file_path,
                        headers,
                        is_numeric_headers,
                        nrows,
                        skiprows,
                        identifier_column_prefix,
                    )
                    data = pd.concat([data, temp_data], ignore_index=True)
                    logging.info(
                        f"Concatenated file: {file}, current data shape: {data.shape}"
                    )
    else:
        raise ValueError(f"Invalid path: {path_to_data}")

    if not data.empty:
        columns = [identifier_column_name] + [col for col in data.columns if col != identifier_column_name]
        data = data[columns]

    return data


# import logging
# import os
# from typing import List, Optional

# import pandas as pd

# # Configure logging
# logging.basicConfig(
#     level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
# )


# def load_data(
#     path_to_data: str,
#     file_names: Optional[List[str]] = [],
#     nrows: Optional[int] = None,
#     skiprows: Optional[int] = None,
# ):
#     data = pd.DataFrame()
#     headers = None
#     is_numeric_headers = None

#     def get_headers(file_path: str):
#         """Helper function to extract headers from the first row of the file."""
#         first_row = pd.read_csv(file_path, nrows=1, header=None)
#         return (
#             first_row.iloc[0].tolist()
#             if first_row.iloc[0].dtype == "object"
#             else [f"f_{i}" for i in range(first_row.shape[1])]
#         )

#     # Check if the path is a file
#     if os.path.isfile(path_to_data):
#         if path_to_data.endswith(".csv") and (
#             not file_names or os.path.basename(path_to_data) in file_names
#         ):
#             headers = get_headers(path_to_data)
#             is_numeric_headers = headers is None

#             temp_data = pd.read_csv(
#                 path_to_data, nrows=nrows, skiprows=skiprows, header=None
#             )
#             if is_numeric_headers:
#                 temp_data.columns = [f"f_{i}" for i in range(temp_data.shape[1])]
#             else:
#                 temp_data.columns = headers
#             column_name = "file_name"
#             i = 1
#             while column_name in temp_data.columns:
#                 column_name = f"file_name_{i}"
#                 i += 1
#             temp_data[column_name] = os.path.basename(path_to_data)
#             data = temp_data
#             logging.info(f"Loaded file: {path_to_data}, data shape: {data.shape}")
#     # If the path is a directory

#     elif os.path.isdir(path_to_data):
#         for root, _, files in os.walk(path_to_data):
#             for file in files:
#                 if file.endswith(".csv") and (not file_names or file in file_names):
#                     file_path = os.path.join(root, file)
#                     temp_headers = get_headers(file_path)
#                     temp_is_numeric_headers = temp_headers is None

#                     if headers is None:
#                         headers = temp_headers
#                         is_numeric_headers = temp_is_numeric_headers

#                     if is_numeric_headers != temp_is_numeric_headers:
#                         raise ValueError(
#                             "Mixed header conditions detected. Some files have headers while others do not."
#                         )

#                     temp_data = pd.read_csv(
#                         file_path, nrows=nrows, skiprows=skiprows, header=None
#                     )
#                     if is_numeric_headers:
#                         if temp_data.shape[1] != len(headers):
#                             raise ValueError(
#                                 "Files do not have the same number of columns."
#                             )
#                         temp_data.columns = headers
#                     else:
#                         if temp_headers != headers:
#                             raise ValueError("Files do not have the same headers.")
#                         temp_data.columns = headers
#                         temp_data = temp_data.iloc[1:].reset_index(drop=True)
#                     column_name = "file_name"
#                     i = 1
#                     while column_name in temp_data.columns:
#                         column_name = f"file_name_{i}"
#                         i += 1
#                     temp_data[column_name] = file
#                     data = pd.concat([data, temp_data], ignore_index=True)
#                     logging.info(
#                         f"Concatenated file: {file}, current data shape: {data.shape}"
#                     )
#     # Move the column named column_name to the first column
#     if not data.empty and column_name in data.columns:
#         columns = [column_name] + [col for col in data.columns if col != column_name]
#         data = data[columns]
#     else:
#         raise ValueError(f"Invalid path: {path_to_data}")

#     return data

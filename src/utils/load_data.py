import logging
import os
from typing import List, Optional, Tuple

import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def _extract_headers(file_path: str) -> List[str]:
    """Extract headers from the first row of the file."""
    first_row = pd.read_csv(file_path, nrows=1, header=None)
    if first_row.iloc[0].dtype == "object":
        return first_row.iloc[0].tolist()
    return [f"f_{i}" for i in range(first_row.shape[1])]


def _find_identifier_column_name(
    column_names: List[str], identifier_column_prefix: Optional[str] = "file_name"
) -> str:
    identifier_column_name = identifier_column_prefix
    i = 1
    while identifier_column_name in column_names:
        identifier_column_name = identifier_column_prefix + f"_{i}"
        i += 1
    return identifier_column_name


def _load_single_file(
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
        temp_data = temp_data.reset_index(drop=True)
    identifier_column_name = _find_identifier_column_name(
        column_names=temp_data.columns,
        identifier_column_prefix=identifier_column_prefix,
    )
    temp_data[identifier_column_name] = os.path.basename(file_path)
    return temp_data, identifier_column_name


def _validate_headers(
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
    input_path: str,
    file_names: Optional[List[str]] = None,
    nrows: Optional[int] = None,
    skiprows: Optional[int] = 0,
    identifier_column_prefix: Optional[str] = "file_name",
) -> pd.DataFrame:
    """Load data from a file or directory of CSV files."""
    file_names = file_names or []
    data = pd.DataFrame()
    headers, is_numeric_headers = None, None

    if os.path.isfile(input_path):
        if input_path.endswith(".csv") and (
            not file_names or os.path.basename(input_path) in file_names
        ):
            headers = _extract_headers(input_path)
            is_numeric_headers = headers is None
            data, identifier_column_name = _load_single_file(
                input_path,
                headers,
                is_numeric_headers,
                nrows,
                skiprows,
                identifier_column_prefix,
            )
            logging.info(f"Loaded file: {input_path}, data shape: {data.shape}")
    elif os.path.isdir(input_path):
        for root, _, files in os.walk(input_path):
            for file in files:
                if file.endswith(".csv") and (not file_names or file in file_names):
                    file_path = os.path.join(root, file)
                    temp_headers = _extract_headers(file_path)
                    temp_is_numeric_headers = temp_headers is None
                    headers, is_numeric_headers = _validate_headers(
                        headers,
                        temp_headers,
                        is_numeric_headers,
                        temp_is_numeric_headers,
                    )
                    temp_data, identifier_column_name = _load_single_file(
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
        raise ValueError(f"Invalid path: {input_path}")

    if not data.empty:
        columns = [identifier_column_name] + [
            col for col in data.columns if col != identifier_column_name
        ]
        data = data[columns]

    return data


def csv_to_parquet(file_path: str):
    """
    Convert a CSV file to a Parquet file.

    Args:
        file_path (str): Path to the CSV file.

    Returns:
        str: Path to the generated Parquet file.
    """
    if not file_path.endswith(".csv"):
        raise ValueError("The input file must be a CSV file.")

    parquet_file_path = file_path.replace(".csv", ".parquet")
    try:
        logging.info(f"Reading CSV {file_path} ...")
        df = pd.read_csv(file_path)
        logging.info(f"Parqueting ...")
        df.to_parquet(parquet_file_path, index=False)
        logging.info(f"Converted {file_path} to {parquet_file_path}")
        return parquet_file_path
    except Exception as e:
        logging.error(f"Failed to convert {file_path} to Parquet: {e}")
        raise

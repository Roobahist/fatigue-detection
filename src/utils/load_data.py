import logging
import os
from typing import List, Optional, Tuple

import pandas as pd
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def _extract_headers(file_path: str) -> List[str]:
    """Extract headers from the first row of the file."""
    if file_path.endswith('.csv'):
        first_row = pd.read_csv(file_path, nrows=1, header=None)
        if first_row.iloc[0].dtype == "object":
            return first_row.iloc[0].tolist()
    return None


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
    column_names: Optional[List[str]],
    nrows: Optional[int],
    skiprows: Optional[int] = 0,
    identifier_column_prefix: Optional[str] = "file_name",
) -> Tuple[pd.DataFrame, str]:
    """Load a single CSV or NPY file into a DataFrame."""
    if file_path.endswith('.csv'):
        # For CSV files
        csv_headers = _extract_headers(file_path)
        has_header = csv_headers is not None
        
        # Determine if we should skip the header row
        effective_skiprows = skiprows if not has_header else skiprows+1
        
        temp_data = pd.read_csv(
            file_path,
            nrows=nrows,
            skiprows=effective_skiprows,
            header=None,
        )
    elif file_path.endswith('.npy'):
        # For NPY files
        np_data = np.load(file_path)
        # Take only the specified number of rows if nrows is provided
        if skiprows > 0 and skiprows < np_data.shape[0]:
            np_data = np_data[skiprows:]
        if nrows is not None and nrows < np_data.shape[0]:
            np_data = np_data[:nrows]
        # Skip rows if specified
        temp_data = pd.DataFrame(np_data)
        csv_headers = None  # NPY files don't have headers
    else:
        raise ValueError(f"Unsupported file format: {file_path}")
    
    # Apply column names with priority:
    # 1. User-provided column_names
    # 2. Headers from CSV file
    # 3. Generic column names (f_0, f_1, etc.)
    if column_names is not None:
        # Raise error if column count doesn't match
        if len(column_names) != temp_data.shape[1]:
            raise ValueError(
                f"Column names count ({len(column_names)}) does not match data column count ({temp_data.shape[1]}) "
                f"in file: {file_path}"
            )
        temp_data.columns = column_names

    elif csv_headers is not None:
        # Use headers from CSV file if available and no column_names were provided
        if len(csv_headers) != temp_data.shape[1]:
            raise ValueError(
                f"CSV header count ({len(csv_headers)}) does not match data column count ({temp_data.shape[1]}) "
                f"in file: {file_path}"
            )
        temp_data.columns = csv_headers
        
    else:
        # Use generic column names as last resort
        temp_data.columns = [f"f_{i}" for i in range(temp_data.shape[1])]
    
    temp_data = temp_data.reset_index(drop=True)
    
    # Add file identifier column
    identifier_column_name = _find_identifier_column_name(
        column_names=temp_data.columns,
        identifier_column_prefix=identifier_column_prefix,
    )
    temp_data[identifier_column_name] = os.path.basename(file_path)
    if not temp_data.empty and identifier_column_name:
        # Reorder columns to put identifier first
        columns = [identifier_column_name] + [
            col for col in temp_data.columns if col != identifier_column_name
        ]
        temp_data = temp_data[columns]
    return temp_data


def load_data(
    input_path: str,
    file_names: Optional[List[str]] = None,
    column_names: Optional[List[str]] = None,
    nrows: Optional[int] = None,
    skiprows: Optional[int] = 0,
    identifier_column_prefix: Optional[str] = "file_name",
) -> pd.DataFrame:
    """
    Load data from a file or directory of CSV or NPY files.
    
    Parameters:
    -----------
    input_path : str
        Path to a file or directory containing files to load
    file_names : Optional[List[str]]
        List of specific file names to load (if None, load all supported files)
    column_names : Optional[List[str]]
        List of column names to use (overrides any headers in CSV files).
        Must match the exact number of columns in the data or an error will be raised.
    nrows : Optional[int]
        Number of rows to load
    skiprows : Optional[int]
        Number of rows to skip
    identifier_column_prefix : Optional[str]
        Prefix for the column that identifies the source file name
    
    Returns:
    --------
    pd.DataFrame
        Combined data from all loaded files
    """
    file_names = file_names or []
    data = pd.DataFrame()
    supported_extensions = ['.csv', '.npy']

    if os.path.isfile(input_path):
        # Handle single file
        file_ext = os.path.splitext(input_path)[1].lower()
        if file_ext in supported_extensions and (
            not file_names or os.path.basename(input_path) in file_names
        ):
            data = _load_single_file(
                input_path,
                column_names,
                nrows,
                skiprows,
                identifier_column_prefix,
            )
            logging.info(f"Loaded file: {input_path}, data shape: {data.shape}")

    elif os.path.isdir(input_path):
        # Handle directory of files
        for root, _, files in os.walk(input_path):
            for file in files:
                file_ext = os.path.splitext(file)[1].lower()
                if file_ext in supported_extensions and (not file_names or file in file_names):
                    file_path = os.path.join(root, file)
                    temp_data = _load_single_file(
                        file_path,
                        column_names,
                        nrows,
                        skiprows,
                        identifier_column_prefix,
                    )
                    
                    if all(temp_data.columns == data.columns):
                        data = pd.concat([data, temp_data], ignore_index=True)
                        logging.info(
                            f"Concatenated file: {file}, current data shape: {data.shape}"
                        )
                    
                    else:
                        raise ValueError(f"Column names dont match {temp_data.columns}, {data.columns}")
    else:
        raise ValueError(f"Invalid path: {input_path}")

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

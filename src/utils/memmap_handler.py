from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch


class MemmapHandler:
    """
    General memory-mapped array manager.

    Supports incremental writing, random access, and metadata persistence.
    """

    def __init__(
        self,
        path: Union[str, Path],
        shape: Optional[Tuple[int, ...]] = None,
        dtype: str = "float32",
        mode: str = "r+",
        auto_metadata: bool = True,
        misc: Optional[Dict[str, Any]] = {},
    ):
        """
        Initialize or open a memory-mapped file.

        Parameters
        ----------
        path : str or Path
            Path to the .dat file (metadata stored alongside as .json)
        shape : tuple[int], optional
            Shape of the array if creating a new file.
        dtype : str
            NumPy dtype.
        mode : str
            File mode: 'r+' (read/write), 'r' (read-only), 'w+' (overwrite), or 'c' (copy-on-write).
        auto_metadata : bool
            Whether to load/save metadata.json automatically.
        misc : dict, optional
            Additional metadata to store.
        """
        self.path = Path(path)
        self.meta_path = self.path.with_suffix(".json")
        self.dtype = np.dtype(dtype)
        self.mode = mode
        self.auto_metadata = auto_metadata
        self.shape = shape
        self.misc = misc
        self._memmap: Optional[np.memmap] = None
        self._column_names: Optional[List[str]] = None

        if mode.startswith("w"):
            if shape is None:
                raise ValueError("Shape must be provided when creating a new file.")
            self._create_new()
        else:
            if misc is not None:
                logging.info(
                    "Misc metadata provided will be ignored when loading existing file."
                )
            self._load_existing()

    # -------------------- Core setup --------------------

    def _create_new(self):
        """Create a new memmap file and metadata."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._memmap = np.memmap(
            self.path, dtype=self.dtype, mode="w+", shape=self.shape
        )
        self._memmap.flush()
        if self.auto_metadata:
            self._save_metadata(
                {"shape": self.shape, "dtype": str(self.dtype), "misc": self.misc}
            )

    def _load_existing(self):
        """Open an existing memmap file and load shape from metadata."""
        if not self.path.exists():
            raise FileNotFoundError(f"No such file: {self.path}")
        if self.meta_path.exists():
            meta = json.loads(self.meta_path.read_text())
            self.shape = tuple(meta["shape"])
            self.dtype = np.dtype(meta["dtype"])
            self._column_names = meta.get("column_names", None)
            self.misc = meta.get("misc", {})
        elif self.shape is None:
            raise ValueError("Shape required if metadata file is missing.")
        self._memmap = np.memmap(
            self.path, dtype=self.dtype, mode=self.mode, shape=self.shape
        )

    # -------------------- Metadata --------------------

    def _save_metadata(self, meta: dict[str, Any]):
        meta.update({"path": str(self.path)})
        self.meta_path.write_text(json.dumps(meta, indent=2))

    def update_metadata(self, **kwargs):
        """Update metadata file with extra info."""
        if not self.auto_metadata:
            return
        meta = {}
        if self.meta_path.exists():
            meta = json.loads(self.meta_path.read_text())
        meta.update(kwargs)
        self._save_metadata(meta)

    # -------------------- Data Preparation --------------------

    def _validate_or_set_columns(self, keys: List[str]):
        """Ensure consistency of column names across writes."""
        if self._column_names is None:
            self._column_names = keys
            if self.auto_metadata:
                self.update_metadata(column_names=keys)
        elif keys != self._column_names:
            raise ValueError(
                f"Inconsistent keys: expected {self._column_names}, got {keys}"
            )

    def _check_array_size(self, arr: np.ndarray):
        """
        Ensure that the incoming array has the same number of columns as existing metadata.
        """
        # Skip for 1D arrays (must be reshaped)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)

        num_cols = arr.shape[1]

        if self.shape[1] != num_cols:
            raise ValueError(f"Expected {self.shape[1]} columns but got {num_cols}")

    def _prepare_data(
        self, data: Union[np.ndarray, Dict[str, Any], List[Dict[str, Any]]]
    ) -> np.ndarray:
        """
        Normalize input data into a 2D NumPy array and update/check column metadata.
        """
        # Single dict
        if data is None:
            arr = np.empty((1, self.shape[1]), dtype=self.dtype)

        elif isinstance(data, dict):
            keys = list(data.keys())
            arr = np.array(list(data.values()), dtype=self.dtype).reshape(1, -1)
            self._validate_or_set_columns(keys)

        # List of dicts
        elif isinstance(data, list) and all(isinstance(d, dict) for d in data):
            for _, d in enumerate(data):
                keys = list(d.keys())
                self._validate_or_set_columns(keys)
            arr = np.array([[d[k] for k in keys] for d in data], dtype=self.dtype)

        # Already NumPy
        elif isinstance(data, np.ndarray):
            arr = data.astype(self.dtype)

        else:
            raise TypeError(
                "Data must be a NumPy array, dict, or list of dicts or None."
            )

        self._check_array_size(arr)
        return arr

    # -------------------- Writing --------------------

    def write(
        self,
        data: Union[np.ndarray, Dict[str, Any], List[Dict[str, Any]]],
        start: int = 0,
    ):
        """Overwrite a segment of the memmap starting from a given index."""
        arr = self._prepare_data(data)
        end = start + len(arr)
        self._memmap[start:end] = arr
        self._memmap.flush()

    def append(self, data: Union[np.ndarray, Dict[str, Any], List[Dict[str, Any]]]):
        """
        Append new data to the end of the file (auto-resizing).
        Only valid for 2D arrays.
        """
        arr = self._prepare_data(data)
        old_shape = self._memmap.shape
        if len(old_shape) != 2:
            raise ValueError("Append supported only for 2D arrays.")
        new_rows = arr.shape[0]
        new_shape = (old_shape[0] + new_rows, old_shape[1])
        self._memmap.flush()
        self._memmap._mmap.close()
        with open(self.path, "ab") as f:
            f.write(arr.tobytes())
        self._memmap = np.memmap(
            self.path, dtype=self.dtype, mode="r+", shape=new_shape
        )
        self.shape = new_shape
        if self.auto_metadata:
            self.update_metadata(shape=new_shape)

    # -------------------- Reading --------------------

    def read(self, slice=slice(None)) -> np.ndarray:
        """Read a slice of the data."""
        return np.array(self._memmap[slice])

    def __getitem__(self, idx: Union[int, slice, Tuple]) -> np.ndarray:
        """Array-style indexing."""
        return self._memmap[idx]

    def as_numpy(self) -> np.ndarray:
        """Return the underlying memmap as a NumPy array (still memory-mapped)."""
        return self._memmap

    # def as_torch(self) -> torch.Tensor:
    #     """Return as a PyTorch tensor (zero-copy)."""
    #     return torch.from_numpy(self._memmap)

    def as_dataframe(self, columns: Optional[list[str]] = None) -> pd.DataFrame:
        """Return as a Pandas DataFrame."""
        df = pd.DataFrame(self._memmap)
        if columns:
            df.columns = columns
        return df

    # -------------------- Utilities --------------------

    def flush(self):
        """Flush any pending writes to disk."""
        if self._memmap is not None:
            self._memmap.flush()

    def close(self):
        """Close the underlying mmap object."""
        if self._memmap is not None:
            self.flush()
            del self._memmap
            self._memmap = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def __repr__(self):
        return f"<MemmapHandler path='{self.path.name}' shape={self.shape} dtype={self.dtype} mode='{self.mode}'>"

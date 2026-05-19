from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union

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
        misc: Optional[Dict[str, Any]] = None,
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
        self.misc = misc if misc is not None else {}
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

        elif isinstance(data, list) and all(isinstance(d, np.ndarray) for d in data):
            arr = np.array(data, dtype=self.dtype)
            
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
        slice: Optional[Union[slice, Tuple]] = None,
        **kwargs,
    ):
        """
        Write to the memmap.

        Backward compatible modes:
          1) Row-range write (original behavior):
                mh.write(row_dict_or_array, start=idx)

          2) Slice write (new behavior, supports your use case):
                mh.write(block, slice=(slice(r0, r1), slice(c0, c1)))

        Notes:
          - The keyword "slice" is accepted (as used in your helper).
          - When using slice writes, column count checks are based on the slice shape,
            not on the full memmap column count.
        """
        # Support the caller's "slice=" keyword while keeping slice parameter name clean
        if slice is None and "slice" in kwargs:
            slice = kwargs["slice"]

        if self._memmap is None or self.shape is None:
            raise ValueError("Memmap is not initialized.")

        arr = self._prepare_data(data)

        if slice is not None:
            arr2 = self._normalize_target_and_data_for_slice(slice, arr)
            self._memmap[slice] = arr2
            self._memmap.flush()
            return

        # Original behavior: write contiguous rows starting at `start`
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)

        self._check_array_size_full_row_write(arr)

        end = start + len(arr)
        self._memmap[start:end] = arr
        self._memmap.flush()

    def append(self, data: Union[np.ndarray, Dict[str, Any], List[Dict[str, Any]]]):
        """
        Append new data to the end of the file (auto-resizing).
        Only valid for 2D arrays.
        """
        arr = self._prepare_data(data)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)

        if self._memmap is None:
            raise ValueError("Memmap is not initialized.")

        old_shape = self._memmap.shape
        if len(old_shape) != 2:
            raise ValueError("Append supported only for 2D arrays.")
        if self.shape is None:
            raise ValueError("Shape is not initialized.")

        self._check_array_size_full_row_write(arr)

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

    def remove_rows(
        self,
        *,
        drop_indices: Optional[Sequence[int]] = None,
        drop_mask: Optional[Union[np.ndarray, Sequence[bool]]] = None,
        predicate: Optional[Callable[[np.ndarray], Union[bool, np.bool_]]] = None,
        batch_size: int = 100_000,
        make_backup: bool = False,
    ) -> Dict[str, Any]:
        """
        Remove rows from a 2D memmap by rewriting the file.

        You can specify exactly one of:
        - drop_indices: row indices to remove
        - drop_mask: boolean mask of length N where True means "drop"
        - predicate: function that takes a single row (1D np.ndarray) and returns True to drop

        Notes
        -----
        - This operation rewrites the entire file to a temporary file, then replaces the original.
        - Works for 2D memmaps only (shape = (N, D)).
        - Row order is always preserved.

        Parameters
        ----------
        drop_indices:
            Indices of rows to remove.
        drop_mask:
            Boolean mask of rows to remove. Length must be N.
        predicate:
            Called on each row, should return True if the row must be removed.
        batch_size:
            Rows per batch when streaming the rewrite.
        make_backup:
            If True, keeps a copy of the original .dat and .json next to the file with suffix ".bak".

        Returns
        -------
        dict with keys:
            - old_shape
            - new_shape
            - removed_count
            - kept_count
            - backup_paths (optional)
        """
        if self._memmap is None or self.shape is None:
            raise ValueError("Memmap is not initialized.")
        if len(self.shape) != 2:
            raise ValueError("remove_rows is supported only for 2D memmaps.")
        if self.mode == "r":
            raise ValueError("remove_rows requires a writable mode (not 'r').")

        provided = sum(x is not None for x in (drop_indices, drop_mask, predicate))
        if provided != 1:
            raise ValueError("Provide exactly one of drop_indices, drop_mask, or predicate.")

        n_rows, n_cols = self.shape
        old_shape = self.shape

        # Build a "keep" mask or a set of rows to drop, with validation
        if drop_mask is not None:
            dm = np.asarray(drop_mask, dtype=bool)
            if dm.ndim != 1 or dm.shape[0] != n_rows:
                raise ValueError(
                    f"drop_mask must be 1D of length {n_rows}, got shape {dm.shape}"
                )
            keep_mask = ~dm
            drop_set = None
            use_predicate = False

        elif drop_indices is not None:
            idx = np.asarray(drop_indices, dtype=np.int64)
            if idx.ndim != 1:
                raise ValueError("drop_indices must be a 1D sequence of integers.")
            if idx.size == 0:
                return {
                    "old_shape": old_shape,
                    "new_shape": old_shape,
                    "removed_count": 0,
                    "kept_count": n_rows,
                }
            if np.any(idx < 0) or np.any(idx >= n_rows):
                bad = idx[(idx < 0) | (idx >= n_rows)]
                raise ValueError(f"drop_indices out of range: {bad[:10].tolist()}")

            drop_set = set(int(i) for i in idx.tolist())
            keep_mask = None
            use_predicate = False

        else:
            if not callable(predicate):
                raise ValueError("predicate must be callable.")
            keep_mask = None
            drop_set = None
            use_predicate = True

        tmp_dat = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp_json = tmp_dat.with_suffix(".json")
        backup_paths: Optional[Dict[str, str]] = None

        # Flush and close the current memmap
        self.flush()
        try:
            self._memmap._mmap.close()  # type: ignore[attr-defined]
        except Exception:
            pass
        
        # Keep reference to old memmap before setting to None
        old_memmap = self._memmap
        self._memmap = None
        del old_memmap

        # Open source memmap for reading
        src_mm = np.memmap(self.path, dtype=self.dtype, mode="r", shape=old_shape)

        kept = 0
        removed = 0

        # Write directly to file instead of using memmap initially
        with open(tmp_dat, "wb") as f:
            def _write_block(block: np.ndarray, global_row_start: int) -> None:
                nonlocal kept, removed

                if block.size == 0:
                    return

                if keep_mask is not None:
                    km = keep_mask[global_row_start : global_row_start + block.shape[0]]
                    out = block[km]
                    k = out.shape[0]
                    if k > 0:
                        f.write(out.tobytes())
                        kept += k
                    removed += block.shape[0] - k
                    return

                if drop_set is not None:
                    rows = np.arange(global_row_start, global_row_start + block.shape[0])
                    km = np.fromiter(
                        (int(r) not in drop_set for r in rows),
                        dtype=bool,
                        count=block.shape[0],
                    )
                    out = block[km]
                    k = out.shape[0]
                    if k > 0:
                        f.write(out.tobytes())
                        kept += k
                    removed += block.shape[0] - k
                    return

                km_list = []
                for i in range(block.shape[0]):
                    should_drop = bool(predicate(block[i]))  # type: ignore[misc]
                    km_list.append(not should_drop)
                km = np.asarray(km_list, dtype=bool)
                out = block[km]
                k = out.shape[0]
                if k > 0:
                    f.write(out.tobytes())
                    kept += k
                removed += block.shape[0] - k

            # Process data in batches
            for i in range(0, n_rows, batch_size):
                j = min(i + batch_size, n_rows)
                block = np.asarray(src_mm[i:j])
                _write_block(block, i)

        # Close the source memmap
        del src_mm

        # Create new shape based on kept rows
        new_shape = (kept, n_cols)

        # Write metadata for new file
        meta: Dict[str, Any] = {
            "shape": list(new_shape),
            "dtype": str(self.dtype),
            "misc": self.misc if self.misc is not None else {},
        }
        if self._column_names is not None:
            meta["column_names"] = self._column_names
        tmp_json.write_text(json.dumps({**meta, "path": str(self.path)}, indent=2))

        # Handle backup or deletion of original files
        if make_backup:
            bak_dat = self.path.with_suffix(self.path.suffix + ".bak")
            bak_json = self.meta_path.with_suffix(self.meta_path.suffix + ".bak")
            if self.path.exists():
                self.path.replace(bak_dat)
            if self.meta_path.exists():
                self.meta_path.replace(bak_json)
            backup_paths = {"dat": str(bak_dat), "json": str(bak_json)}
        else:
            if self.path.exists():
                self.path.unlink()
            if self.meta_path.exists():
                self.meta_path.unlink()

        # Replace original with temporary files
        tmp_dat.replace(self.path)
        tmp_json.replace(self.meta_path)

        # Reopen the memmap with new shape
        self.shape = new_shape
        self._memmap = np.memmap(
            self.path, dtype=self.dtype, mode="r+", shape=self.shape
        )

        return {
            "old_shape": old_shape,
            "new_shape": new_shape,
            "removed_count": removed,
            "kept_count": kept,
            **({"backup_paths": backup_paths} if backup_paths is not None else {}),
        }
    # -------------------- Reading --------------------

    def read(self, slice=slice(None)) -> np.ndarray:
        """Read a slice of the data."""
        return np.array(self._memmap[slice])

    def __getitem__(self, idx: Union[int, slice, Tuple]) -> np.ndarray:
        """Array-style indexing."""
        return self._memmap[idx]
    
    def iter_batches(
        self,
        batch_size: int,
        start: int = 0,
        end: Optional[int] = None,
        as_numpy: bool = True,
    ) -> Iterator[np.ndarray]:
        """
        Iterate over the data in batches along the first axis.

        Parameters
        ----------
        batch_size : int
            Number of rows per batch.
        start : int
            Starting row index (inclusive).
        end : int, optional
            Ending row index (exclusive). Defaults to len(self).
        as_numpy : bool
            If True, return batches as np.ndarray copies.
            If False, return memmap views (may be more memory-efficient,
            but some ops may force copies later anyway).
        """
        if self.shape is None:
            raise ValueError("Shape is not initialized.")
        n = self.shape[0]
        if end is None:
            end = n
        if end > n:
            raise ValueError(f"end={end} exceeds number of rows={n}")

        for i in range(start, end, batch_size):
            j = min(i + batch_size, end)
            batch = self[i:j]
            if as_numpy:
                batch = np.array(batch)
            yield batch

    def as_numpy(self) -> np.ndarray:
        """Return the underlying memmap as a NumPy array (still memory-mapped)."""
        return self._memmap

    def as_torch(self) -> torch.Tensor:
        """Return as a PyTorch tensor (zero-copy)."""
        return torch.from_numpy(self._memmap)

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

    def __len__(self) -> int:
        """Number of rows along the first dimension."""
        if self.shape is None:
            raise ValueError("Shape is not initialized.")
        return self.shape[0]

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def __repr__(self):
        return f"<MemmapHandler path='{self.path.name}' shape={self.shape} dtype={self.dtype} mode='{self.mode}'>"

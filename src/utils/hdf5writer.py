import os
import h5py
import numpy as np
from typing import Optional, Any, List, Dict
from datetime import datetime

class HDF5Writer:
    def __init__(self, file_path: str):
        """
        Initialize HDF5Writer. File opened in append mode.
        """
        self.file_path = file_path
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        self.h5file = h5py.File(self.file_path, "a")
        self.datasets = {}  # Cache for open handles

    def _get_or_create_dataset(self, key: str, value: np.ndarray):
        """
        Returns dataset if exists, otherwise creates it with appropriate shape.
        """
        if key in self.datasets:
            return self.datasets[key]

        if key in self.h5file:
            dset = self.h5file[key]
        else:
            initial_shape = (0,) + value.shape
            maxshape = (None,) + value.shape
            dset = self.h5file.create_dataset(
                name=key,
                shape=initial_shape,
                maxshape=maxshape,
                dtype=value.dtype,
                chunks=True,
                compression="gzip"
            )
        self.datasets[key] = dset
        return dset

    def write(self, key: str, value: np.ndarray):
        """
        Append one record to a specific dataset.
        """
        dset = self._get_or_create_dataset(key, value)
        new_shape = (dset.shape[0] + 1,) + dset.shape[1:]
        dset.resize(new_shape)
        dset[-1] = value
        self.h5file.flush()

    def batch_write(self, data_dict: dict):
        """
        Write multiple embeddings in one call (e.g. per frame).
        """
        for key, val in data_dict.items():
            self.write(key, val)

    def read(self, key: str, index: int) -> Optional[np.ndarray]:
        """
        Read a specific record from a dataset.
        """
        if key not in self.h5file:
            return None
        return self.h5file[key][index]

    def get_dataset_length(self, key: str) -> int:
        """
        Get number of entries in a given dataset.
        """
        if key not in self.h5file:
            return 0
        return self.h5file[key].shape[0]

    def get_all_keys(self) -> List[str]:
        """
        Return all dataset names (embedding strategy keys).
        """
        return list(self.h5file.keys())

    def clear(self, key: Optional[str] = None):
        """
        Clear a single dataset (if key is given) or all datasets (if key is None).
        """
        if key is not None:
            if key in self.h5file:
                del self.h5file[key]
            if key in self.datasets:
                del self.datasets[key]
        else:
            for k in list(self.h5file.keys()):
                del self.h5file[k]
            self.datasets.clear()

    def flush(self):
        """Manually flush to disk."""
        self.h5file.flush()

    def close(self):
        """Close the HDF5 file and release all handles."""
        self.h5file.close()
        self.datasets.clear()

    def add_metadata(self, metadata: Dict[str, str]):
        """
        Add a metadata entry to the HDF5 file under the '__metadata__' group.
        """
        meta_group = self.h5file.require_group("__metadata__")

        # Add timestamp automatically
        metadata["timestamp"] = datetime.utcnow().isoformat()

        next_idx = 0
        while f"entry_{next_idx}" in meta_group:
            next_idx += 1

        meta_entry = meta_group.create_group(f"entry_{next_idx}")
        for k, v in metadata.items():
            meta_entry.attrs[k] = v

    def read_all_metadata(self) -> List[Dict[str, str]]:
        """
        Read all metadata entries from the '__metadata__' group.
        """
        result = []
        if "__metadata__" not in self.h5file:
            return result

        meta_group = self.h5file["__metadata__"]
        for entry in meta_group.values():
            entry_dict = {k: entry.attrs[k] for k in entry.attrs}
            result.append(entry_dict)
        return result

    def check_metadata_consistency(self, existing: List[Dict[str, str]] = None, new: Dict[str, str] = None):
        """
        Placeholder: Task-specific metadata consistency check.
        Override in subclass or externally as needed.
        """
        existing = existing or []
        new = new or {}
        
        pass  # To be implemented for task-specific logic
import numpy as np
import pandas as pd
from typing import overload, Optional, Union, List, Tuple


def reshape_data_rnn(
    df: pd.DataFrame,
    feature_columns: List[str],
    slice_size: int,
    step_size: Optional[int] = None,
    groupby_column: str = "file_name",
    sortby_column: str = "frame",
    target_column: str = "target",
    discard_threshold_ratio: float = 0.6,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Reshape data for RNN input by grouping, slicing, and padding sequences - optimized version.

    Args:
        df (pd.DataFrame)              : Input dataframe.
        feature_columns (List[str])    : List of feature column names.
        slice_size (int)               : Maximum number of slices in each chunk.
        step_size (int)                : Step size for moving to the next chunk.
        groupby_column (str)           : Column to group by (e.g., video or file name).
        sortby_column (str)            : Column to sort within each group (e.g., frame index).
        target_column (str)            : Column containing target labels.
        discard_threshold_ratio (float): Last slice in each video will be discarded if the number of its frames is less than
                                         discard_threshold_ratio*slice_size and will be post padded with zero and kept otherwise.
    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: Feature sequences (X), corresponding labels (y), and group identifiers (g).
    """
    step_size = step_size or slice_size
    discard_threshold_frame = int(discard_threshold_ratio * slice_size)
    
    # Pre-allocate lists with estimated size to avoid resizing
    total_groups = len(df[groupby_column].unique())
    estimated_total_chunks = max(1, df.shape[0] // step_size) + total_groups #considering all last chunks in each group is higher than threshold

    print(estimated_total_chunks)
    
    # Pre-allocate arrays
    feature_count = len(feature_columns)
    all_chunks = np.zeros((estimated_total_chunks, slice_size, feature_count))
    all_labels = np.empty(estimated_total_chunks, dtype=df[target_column].dtype)
    all_groups = np.empty(estimated_total_chunks, dtype=df[groupby_column].dtype)
    
    chunk_idx = 0
    
    # Process groups in parallel if possible
    for group_name, group_df in df.groupby(groupby_column):
        # Sort efficiently
        group_df = group_df.sort_values(by=sortby_column)
        
        # Get the target value for this group
        y_group = group_df[target_column].iloc[0]  # Faster than using unique()
        
        # Extract features as a numpy array - much faster than .values.tolist()
        features = group_df[feature_columns].to_numpy()
        total_frames = len(features)
        
        # Process all full slices
        start_idx = 0

        for start_idx in range(0, total_frames - slice_size + 1, step_size):
            
            # Store chunk by view, not copy
            all_chunks[chunk_idx] = features[start_idx:start_idx + slice_size]
            all_labels[chunk_idx] = y_group
            all_groups[chunk_idx] = group_name
            chunk_idx += 1
        
        # Process the last partial slice if needed
        remaining_frames = total_frames - (start_idx + slice_size)
        if remaining_frames > discard_threshold_frame:
            
            all_chunks[chunk_idx, :remaining_frames] = features[start_idx + slice_size:]
            all_labels[chunk_idx] = y_group
            all_groups[chunk_idx] = group_name
            chunk_idx += 1
    
    # Trim arrays to actual size
    all_labels = all_labels[:chunk_idx]
    all_groups = all_groups[:chunk_idx]
    all_chunks = all_chunks[:chunk_idx]
    
    return all_chunks, all_labels, all_groups
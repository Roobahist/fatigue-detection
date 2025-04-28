import numpy as np
import pandas as pd
from typing import overload, Optional, Union, List, Tuple
from keras.src.utils import pad_sequences


# def pad_sequences(sequences, maxlen=None, dtype='int32', padding='pre', truncating='pre', value=0):
#     """
#     Pads sequences to the same length.

#     Args:
#         sequences: List of sequences (lists or arrays of varying lengths).
#         maxlen: Maximum length of all sequences after padding. If None, it uses the length of the longest sequence.
#         dtype: Type of the output sequences.
#         padding: 'pre' or 'post', pad either before or after each sequence.
#         truncating: 'pre' or 'post', remove values from sequences longer than maxlen at the beginning or end.
#         value: Padding value.

#     Returns:
#         A numpy array of shape (number_of_sequences, maxlen)
#     """
#     lengths = [len(s) for s in sequences]

#     if maxlen is None:
#         maxlen = max(lengths)

#     padded_sequences = np.full((len(sequences), maxlen), value, dtype=dtype)

#     for idx, seq in enumerate(sequences):
#         if len(seq) > maxlen:
#             if truncating == 'pre':
#                 trunc = seq[-maxlen:]
#             elif truncating == 'post':
#                 trunc = seq[:maxlen]
#             else:
#                 raise ValueError("truncating must be 'pre' or 'post'")
#         else:
#             trunc = seq

#         if padding == 'pre':
#             padded_sequences[idx, -len(trunc):] = trunc
#         elif padding == 'post':
#             padded_sequences[idx, :len(trunc)] = trunc
#         else:
#             raise ValueError("padding must be 'pre' or 'post'")

#     return padded_sequences


def reshape_data_rnn(
    df: pd.DataFrame,
    feature_columns: List[str],
    slice_size: int,
    step_size: Optional[int],
    groupby_column: str = "file_name",
    sortby_column: str = "frame",
    target_column: str = "target",
    discard_threshold_ratio: float = 0.6,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Reshape data for RNN input by grouping, slicing, and padding sequences.

    Args:
        df (pd.DataFrame)              : Input dataframe.
        feature_columns (List[str])    : List of feature column names.
        slice_size (int)               : Maximum number of slices in each chunk.
        step_size (int)                : Step size for moving to the next chunk.
        groupby_column (str)           : Column to group by (e.g., video or file name).
        sortby_column (str)            : Column to sort within each group (e.g., frame index).
        target_column (str)            : Column containing target labels.
        discard_threshold_ratio (float): Last slice in each video will be discarded if the number of its frames is less than
                                         discard_threshold_ratio*max_slice and will be post padded with zero and kept otherwise.
    Returns:
        Tuple[np.ndarray, np.ndarray]: Feature sequences (X) and corresponding labels (y).
    """
    step_size = step_size or slice_size
    X, y, g = [], [], []

    for group_identifier, group in df.groupby(groupby_column):
        # Sort by the specified column to maintain temporal order
        group = group.sort_values(by=sortby_column, ascending=True)

        y_group = group[target_column].unique()
        assert (
            len(y_group) == 1
        ), f"Expected one unique target value, but found {len(y_group)}"
        y_group = y_group[0]
        assert (target_column not in feature_columns), "Target Column Appeared in the features"
        assert (groupby_column not in feature_columns), "GroupBy Column Appeared in the features"
        # Extract features as a list of lists
        slices = group[feature_columns].values.tolist()
        # Create chunks with the specified step size

        start_idx = 0
        for start_idx in range(0, len(slices) - slice_size + 1, step_size):
            chunk = slices[start_idx : start_idx + slice_size]
            X.append(chunk)
            y.append(y_group)
            g.append(group_identifier)

        discard_threshold_frame = int(discard_threshold_ratio * slice_size)
        if len(slices) - (start_idx + slice_size) > discard_threshold_frame:
            chunk = slices[start_idx + slice_size : len(slices)]
            X.append(chunk)
            y.append(y_group)
            g.append(group_identifier)

    # Pad sequences to ensure uniform length
    X_padded = pad_sequences(X, maxlen=slice_size, padding="post", dtype="float32")

    return np.array(X_padded), np.array(y), np.array(g)


# @overload
# def count_slices_per_video(
#     df_or_lengths: pd.DataFrame,
#     slice_size: int,
#     step_size: Optional[int],
#     groupby_column: str = ...,
#     discard_threshold_ratio: float = ...
# ) -> List[int]:    ...

# @overload
# def count_slices_per_video(
#     df_or_lengths: List[int],
#     slice_size: int,
#     step_size: Optional[int],
#     discard_threshold_ratio: float = ...
# ) -> List[int]:    ...


# def count_slices_per_video(df_or_lengths: Union[pd.DataFrame, List[int]],
#                            slice_size: int,
#                            step_size: Optional[int],
#                            groupby_column: str = "file_name",
#                            discard_threshold_ratio: float = 0.6) -> List[int]:
#     """
#     Calculate the total number of slices for each video.

#     Args:
#         df_or_lengths (pd.DataFrame or List[int]): Input dataframe OR list of video lengths.
#         slice_size (int): Slice size.
#         step_size (int): Step between slices.
#         groupby_column (str): Column name to group by (only if using DataFrame).
#         discard_threshold_ratio (float): Minimum portion to keep the last slice.

#     Returns:
#         int: Total number of slices.
#     """
#     step_size = step_size or slice_size

#     if isinstance(df_or_lengths, pd.DataFrame):
#         video_lengths = [
#             len(group) for _, group in df_or_lengths.groupby(groupby_column)
#         ]
#     else:
#         video_lengths = df_or_lengths

#     num_slices = []
#     for num_frames in video_lengths:
#         video_num_slices = 0
#         num_full_slices = max(0, (num_frames - slice_size) // step_size + 1)
#         video_num_slices += num_full_slices

#         remaining_frames = num_frames - (num_full_slices * step_size +
#                                          slice_size - step_size)
#         discard_threshold_frame = int(discard_threshold_ratio * slice_size)
#         if remaining_frames >= discard_threshold_frame:
#             video_num_slices += 1
#         num_slices.append(video_num_slices)

#     return num_slices
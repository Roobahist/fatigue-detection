from typing import List

import numpy as np
import pandas as pd
from keras.src.trainers.data_adapters.py_dataset_adapter import PyDataset
from src.preprocess.reshape import count_slices_per_video, reshape_data_rnn
from src.utils.common import get_csv_length, pop_row
from src.utils.load_data import _extract_headers, load_data


class MyDataGenerator(PyDataset):
    """
    Custom data generator for Keras.
    """

    def __init__(self, 
                 paths                  : List[str],
                 video_info_csv_path    : str,
                 groupby_column         : str = "file_name",
                 sortby_column          : str = "frame",
                 target_column          : str = "target",
                 columns_to_drop        : List[str] = [],
                 batch_size             : int = 32,
                 slice_size             : int = 3000,
                 step_size              : int = 3000,
                 discard_threshold_ratio: float = 0.6,
                 shuffle                : bool = False,
                 **kwargs): 
        """
        Initialization function.  This is where you set up the basic parameters
        of your data generator.
        """
        super().__init__(**kwargs)

        self.paths                   = paths
        self.video_info_csv_path     = video_info_csv_path
        self.pointer                 = 0
        self.groupby_column          = groupby_column
        self.sortby_column           = sortby_column
        self.target_column           = target_column
        self.columns_to_drop         = columns_to_drop
        self.columns_to_drop_1       = self._get_columns_to_drop_1()
        self.feature_columns         = self._get_feature_columns()
        self.batch_size              = batch_size
        self.slice_size              = slice_size
        self.step_size               = step_size
        self.discard_threshold_ratio = discard_threshold_ratio
        self.shuffle                 = shuffle
        self.num_slices              = []
        self.data                    = pd.DataFrame(columns = ('X', 'y', 'g'))
        self.stash                   = pd.DataFrame(columns = ('X', 'y', 'g'))

        self.on_epoch_end()

    def _get_columns_to_drop_1(self):
        return [
            column_name
            for column_name in self.columns_to_drop
            if column_name not in [self.groupby_column, 
                                   self.sortby_column, 
                                   self.target_column]
        ]
    
    def _get_feature_columns(self):
        return [
            column_name
            for column_name in _extract_headers(self.paths[0])
            if column_name not in self.columns_to_drop
        ]
    
    def __len__(self):
        """
        Returns the number of batches per epoch.  Keras uses this to know how many
        times to call `__getitem__` in one epoch.
        """
        self.pointer = 0
        assert len(self.stash) != 0, f"Stash is not emptied properly! Length of stash : {len(self.stash)}"

        if not self.num_slices:
            video_info = pd.read_csv(self.video_info_csv_path)
            file_lengths = video_info[video_info['path'].isin(self.paths)]['total_frames'].to_list()
            num_slices = count_slices_per_video(df_or_lengths           = file_lengths, 
                                                slice_size              = self.slice_size,
                                                step_size               = self.step_size,
                                                discard_threshold_ratio = self.discard_threshold_ratio)
            self.num_slices = num_slices
        assert self.num_slices != len(self.paths), f"num_slices not calculated properly for all paths. num_slices : {len(self.num_slices)}, paths : {len(self.paths)}"

        return np.ceil(sum(self.num_slices)/self.batch_size)
    
    def _preprocess_data(self, file_data):
        file_data = file_data.drop(self.columns_to_drop_1, axis=1)
        X, y, g = reshape_data_rnn(df                      = file_data, 
                                   feature_columns         = self.feature_columns,
                                   slice_size              = self.slice_size,
                                   step_size               = self.step_size,
                                   groupby_column          = self.groupby_column,
                                   sortby_column           = self.sortby_column,
                                   target_column           = self.target_column,
                                   discard_threshold_ratio = self.discard_threshold_ratio)
        return X, y, g
    
    def _update_generator_memory(self, X_file, y_file, g_file):
        for (X, y, g) in zip(X_file, y_file, g_file):
            record_index = len(self.data)

            self.data.loc[record_index]  = {'X':'-', 'y':y, 'g':g}
            self.stash.loc[record_index] = {'X':X  , 'y':y, 'g':g}
    
    def on_epoch_end(self):
        """
        Called at the end of each epoch.  Use this to re-shuffle the data if needed.
        """
        if self.shuffle:
            self.stash = self.stash.sample(len(self.stash))

    def __data_generation(self):
        """
        Generates data containing batch_size samples. This is where the actual data
        loading and preprocessing happens.
        """
        while len(self.stash) < self.batch_size:
            file_data = load_data(input_path = self.paths[self.pointer])
            X_file, y_file, g_file = self._preprocess_data(file_data = file_data)
            self._update_generator_memory(X_file, y_file, g_file)
            self.pointer += 1

    def _pop_from_stash(self):
        popped, self.stash = pop_row(df = self.stash, 
                                     idx = slice(0,self.batch_size), 
                                     reset_index=False)
        return popped

    def __getitem__(self, index):
        """
        Generates one batch of data.  The core of the data generator.
        """
        self.__data_generation()
        popped = self._pop_from_stash()        
        return (np.stack(popped['X']), np.stack(popped['y']))
import random
from typing import List, Optional, Dict
from unittest import skip

import numpy as np
import pandas as pd
from keras.src.trainers.data_adapters.py_dataset_adapter import PyDataset
from src.preprocess.reshape import count_slices_per_video, reshape_data_rnn
from src.utils.common import merge_intervals, pop_row
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
        self.batch_size              = batch_size
        self.slice_size              = slice_size
        self.step_size               = step_size
        self.discard_threshold_ratio = discard_threshold_ratio
        self.shuffle                 = shuffle
        
        self.columns_to_drop_1       = self._get_columns_to_drop_1()
        self.feature_columns         = self._get_feature_columns()
        self.batch_info              = self._get_batch_info(
            video_info_csv_path     = self.video_info_csv_path,
            batch_size              = self.batch_size,
            slice_size              = self.slice_size,
            step_size               = self.step_size,
            groupby_column          = 'path',
            groupby_column_filter   = self.paths,
            counts_column           = 'total_frames',
            discard_threshold_ratio = self.discard_threshold_ratio)

        self.num_slices              = []

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
            if column_name not in 
            self.columns_to_drop + [self.target_column, 
                                    self.sortby_column, 
                                    self.groupby_column]
        ]

    def _get_batch_info(
        self,
        video_info_csv_path : str,
        batch_size : int,
        slice_size: int,
        step_size: Optional[int],
        groupby_column : str = 'path',
        groupby_column_filter : List[str] = [],
        counts_column : str = 'total_frames',
        discard_threshold_ratio: float = 0.6) -> Dict[str, List]:

        info = {
            'counts' : [],
            'batches': [],
        }

        video_info = load_data(video_info_csv_path)
        groupby_column_filter = groupby_column_filter or video_info[groupby_column].unique()
        video_info = video_info[video_info[groupby_column].isin(groupby_column_filter)]
        
        new_rows = []
        for _, row in video_info.iterrows():
            total = row[counts_column]
            start_frame = 0
            while start_frame < total:
                end_frame = min(start_frame + slice_size, total)
                current_slice_size = end_frame - start_frame
                if current_slice_size >= discard_threshold_ratio * slice_size:
                    new_row = row.copy()  # Create a copy to avoid modifying the original row
                    new_row['start_frame'] = start_frame
                    new_row['end_frame'] = end_frame
                    new_rows.append(new_row)
                start_frame += step_size

        video_info_processed = pd.DataFrame(new_rows).reset_index(drop=True)

        for group_identifier, group in video_info_processed.groupby(groupby_column):
            info['counts'].append({groupby_column : group_identifier, 
                                'count' : len(group)})

        for i in range(int(np.ceil(len(video_info_processed)/batch_size))):
            start_index = i * batch_size
            end_index   = min((i+1) * batch_size, len(video_info_processed))
            current_batch_info  = []
            for group_identifier, group in video_info_processed[start_index: end_index].groupby(groupby_column):
                current_batch_info.append(
                    {
                        groupby_column : group_identifier,
                        'raw_intervals' : list(zip(group['start_frame'], group['end_frame'])),
                        'merged_intervals' : merge_intervals(list(zip(group['start_frame'], group['end_frame'])))
                    }
                )
            info['batches'].append(current_batch_info)

        return info
    
    def __len__(self):
        """
        Returns the number of batches per epoch.  Keras uses this to know how many
        times to call `__getitem__` in one epoch.
        """
        assert len(self.batch_info['counts']) == len(self.paths), f"num_slices not calculated properly for all paths. num_slices : {len(self.batch_info['counts'])}, paths : {len(self.paths)}"

        return int(np.ceil(sum([file['count'] for file in self.batch_info['counts']])/self.batch_size))
    
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
    
    def on_epoch_end(self):
        """
        Called at the end of each epoch.
        #TODO implement proper shuffling. probably not needed as the batch indices are accessed without order
        """
        pass

    def __data_generation(self, index):
        """
        Generates data containing batch_size samples. This is where the actual data
        loading and preprocessing happens.
        """
        current_batch_info = self.batch_info['batches'][index]
        X, y, g = [], [], []

        for file in current_batch_info:
            file_data = load_data(input_path=file['path'], 
                      skiprows=file['merged_intervals'][0][0],
                      nrows=file['merged_intervals'][0][1] - file['merged_intervals'][0][0])
            X_file, y_file, g_file = self._preprocess_data(file_data=file_data)
            
            # Concatenate to the parent arrays
            X.extend(X_file)
            y.extend(y_file)
            g.extend(g_file)

        return {'X': np.array(X), 'y': np.array(y), 'g': np.array(g)}

    def __getitem__(self, index):
        """
        Generates one batch of data.  The core of the data generator.
        #TODO Calculate class weights based on the name of files and num_slices and return
        """
        print(f"index {index}")
        batch = self.__data_generation(index)
        return (batch['X'], batch['y'])
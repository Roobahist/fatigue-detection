import logging
import os
import traceback
from typing import List, Union

def create_folders(to_create_folder_paths:Union[str, List[str]]):
    try:
        if isinstance(to_create_folder_paths, str):
            to_create_folder_paths = [to_create_folder_paths]
        assert isinstance(to_create_folder_paths, list)
        for to_create_folder_path in to_create_folder_paths:
            os.makedirs(to_create_folder_path, exist_ok=True)
        return True
    except Exception as e:
        logging.error(f"Error Creating Folders: %s", traceback.format_exc())
        return False


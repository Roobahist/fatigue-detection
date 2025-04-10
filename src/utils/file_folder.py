import logging
import os
import traceback
from typing import List

def create_folders(to_create_folder_paths:List[str]):
    try:
        for to_create_folder_path in to_create_folder_paths:
            os.makedirs(to_create_folder_path, exist_ok=True)
        return True
    except Exception as e:
        logging.error(f"Error Creating Folders: %s", traceback.format_exc())
        return False

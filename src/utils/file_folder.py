import logging
import os
import traceback
from typing import List, Union


def create_folders(to_create_folder_paths: Union[str, List[str]]) -> bool:
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


def get_file_paths(
    folder_path: str, depth: int = 1, EXT_LIST: List[str] = []
) -> List[str]:
    EXT_LIST = [
        ext.lower() if ext.startswith(".") else "." + ext.lower() for ext in EXT_LIST
    ]

    if depth < 1:
        depth = 1

    file_paths = []
    for dirpath, _, file_names in os.walk(folder_path):
        if dirpath[len(folder_path) :].count(os.sep) < depth:
            file_paths.extend(
                [
                    os.path.join(dirpath, file_name)
                    for file_name in file_names
                    if not EXT_LIST
                    or os.path.splitext(file_name)[1].lower() in EXT_LIST
                ]
            )
    return file_paths

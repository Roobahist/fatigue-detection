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
            if os.path.exists(to_create_folder_path):
                # print(f"Folder already exists: {to_create_folder_path}")
                pass
            else:
                os.makedirs(to_create_folder_path)
                print(f"Folder created: {to_create_folder_path}")
    except Exception as e:
        logging.error(f"Error Creating Folders: %s", traceback.format_exc())


def get_file_paths(
    input_path: str, depth: int = 1, EXT_LIST: List[str] = []
) -> List[str]:
    EXT_LIST = [
        ext.lower() if ext.startswith(".") else "." + ext.lower() for ext in EXT_LIST
    ]

    if depth < 1:
        depth = 1

    file_paths = []
    for dirpath, _, file_names in os.walk(input_path):
        if dirpath[len(input_path) :].count(os.sep) < depth:
            file_paths.extend(
                [
                    os.path.join(dirpath, file_name)
                    for file_name in file_names
                    if not EXT_LIST
                    or os.path.splitext(file_name)[1].lower() in EXT_LIST
                ]
            )
    return file_paths


def copy_directory_structure(source_dir, target_dir):
    """
    Copy the directory structure from source_dir to target_dir.
    This only creates the directories, without copying any files.
    
    Args:
        source_dir (str): Path to the source directory
        target_dir (str): Path to the target directory where subdirectories will be created
    """
    # Check if source directory exists
    if not os.path.isdir(source_dir):
        raise ValueError(f"Source directory '{source_dir}' does not exist or is not a directory")
    
    # Check if target directory exists, create it if it doesn't
    create_folders(target_dir)
    if not os.path.isdir(target_dir):
        raise ValueError(f"Target path '{target_dir}' does not exist or is not a directory")
    
    # Walk through the source directory structure
    for root, _, _ in os.walk(source_dir):
        # Calculate the relative path from source_dir
        relative_path = os.path.relpath(root, source_dir)
        
        # Skip the root directory itself
        if relative_path == '.':
            continue
        
        # Create the corresponding directory in the target path
        new_dir_path = os.path.join(target_dir, relative_path)
        create_folders(new_dir_path)

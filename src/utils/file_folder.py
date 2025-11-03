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
                print(f"Folder already exists: {to_create_folder_path}")
                pass
            else:
                os.makedirs(to_create_folder_path)
                logging.info(f"Folder created: {to_create_folder_path}")
    except FileExistsError:
        logging.info(f"Folder already exists: {to_create_folder_path}")
    except Exception as e:
        logging.error(f"Error Creating Folders: %s", traceback.format_exc())


def get_file_paths(
    input_path: str, 
    return_full_path: bool = True,
    depth: int = 1, 
    EXT_LIST: Union[List[str], str] = []
) -> List[str]:
    """
    Collect file paths under the given input_path with filtering options.

    Args:
        input_path (str): The root directory to search.
        full_path (bool): 
            - True: return absolute paths.
            - False: return relative paths (relative to input_path).
        depth (int): Maximum depth of subdirectories to explore. Must be >= 1.
        EXT_LIST (List[str]): List of file extensions to filter by (case-insensitive). 
                              Example: ['.jpg', 'png'].

    Returns:
        List[str]: List of file paths matching the criteria.
    """
    if not isinstance(EXT_LIST, list):
        EXT_LIST = [EXT_LIST]
        
    # Normalize extension list
    EXT_LIST = [
        ext.lower() if ext.startswith(".") else "." + ext.lower()
        for ext in EXT_LIST
    ]

    if depth < 1:
        depth = 1

    file_paths = []
    input_path = os.path.abspath(input_path)

    for dirpath, _, file_names in os.walk(input_path):
        # Count how deep we are relative to the root
        relative_depth = dirpath[len(input_path):].count(os.sep)
        if relative_depth < depth:
            for file_name in file_names:
                ext = os.path.splitext(file_name)[1].lower()
                if not EXT_LIST or ext in EXT_LIST:
                    full_path = os.path.join(dirpath, file_name)
                    if return_full_path:
                        file_paths.append(full_path)
                    else:
                        file_paths.append(os.path.relpath(full_path, input_path))

    return file_paths


def pick_selected_file_paths(
    input_folder: str,
    depth: int,
    EXT_LIST: Union[List[str], str],
    relative_selected_paths: List[str] = [],
) -> List[str]:

    all_file_paths = get_file_paths(
        input_path=input_folder, depth=depth, EXT_LIST=EXT_LIST
    )

    if relative_selected_paths:
        selected_file_paths = []
        for relative_selected_path in relative_selected_paths:
            if os.path.isdir(os.path.join(input_folder, relative_selected_path)):
                # If it's a directory, add all files under it
                dir_full_path = os.path.join(input_folder, relative_selected_path)
                dir_file_paths = get_file_paths(
                    input_path=dir_full_path,
                    depth=depth,
                    EXT_LIST=EXT_LIST,
                )
                selected_file_paths.extend(dir_file_paths)
            else:
                # If it's a file, add it directly
                full_path = os.path.join(input_folder, relative_selected_path)
                if os.path.exists(full_path):
                    selected_file_paths.append(full_path)
                else:
                    logging.warning(f"Selected path does not exist: {full_path}")

        selected_file_paths = list(set(selected_file_paths))  # Remove duplicates
        
        logging.info(
            f"File Paths Captured Using Provided List: {len(selected_file_paths)}"
        )
    else:
        selected_file_paths = all_file_paths
        logging.info(
            f"File Paths Captured Automatically: {len(selected_file_paths)}"
        )
        
    return selected_file_paths

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


def move_file(src_path: str, dest_path: str, mode: str = "cut"):
    """
    Move a file from src_path to dest_path.
    
    Args:
        src_path (str): Path to the source file
        dest_path (str): Path to the destination file
        mode (str): "cut" to move the file, "copy" to copy it. Default is "cut".
    """
    import shutil

    try:
        if mode == "cut":
            os.rename(src_path, dest_path)
            logging.info(f"Moved file from {src_path} to {dest_path}")
        elif mode == "copy":
            shutil.copy2(src_path, dest_path)
            logging.info(f"Copied file from {src_path} to {dest_path}")
    except Exception as e:
        logging.error(f"Error moving file from {src_path} to {dest_path}: {e}")
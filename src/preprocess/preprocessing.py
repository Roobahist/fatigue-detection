import json
import logging
import os
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd

from Zh.src.utils.common import save_obj
from Zh.src.utils.file_folder import (
    copy_directory_structure,
    create_folders,
    get_file_paths,
)
from Zh.src.utils.serialized_parallel import process
from Zh.src.utils.video import get_video_info

logging.basicConfig(
    level=logging.INFO,  # or DEBUG for more verbosity
    format="[%(asctime)s][%(levelname)s] %(message)s",
)


# e7
def fix_columns(
    data: pd.DataFrame, mapping: Optional[dict] = None
) -> Tuple[pd.DataFrame, List[str]]:
    # Ensure the 'file_name' column exists
    if "file_name" not in data.columns:
        raise ValueError("The data must contain a 'file_name' column.")

    # Add the 'frame' column
    data["frame"] = data.groupby("file_name").cumcount()

    # Split 'file_name' to extract 'fold', 'subject_id', 'target', and 'type'
    split_columns = data["file_name"].str.split("_", expand=True)
    data["fold"] = split_columns[0].astype(int)
    data["subject_id"] = split_columns[1].astype(int).astype(str)
    data["target"] = split_columns[2]
    data["type"] = split_columns[4].str.replace(".csv", "", regex=False)

    # Handle the 'target' column
    if mapping:
        unique_targets = data["target"].unique()
        if not set(unique_targets).issubset(set(mapping.keys())):
            raise ValueError("Mapping does not include all unique target values.")
        data["target"] = data["target"].map(mapping).astype(float)
    else:
        data["target"] = data["target"].astype(float)

    # Reorder columns
    columns_order = ["file_name", "type", "frame", "fold", "subject_id", "target"]
    remaining_columns = [col for col in data.columns if col not in columns_order]
    data = data[columns_order + remaining_columns]
    columns_to_drop = [
        "file_name",
        "type",
        "frame",
        "fold",
        "subject_id",
    ]

    return data, columns_to_drop


def _remove_background_from_video_one_frame(
    frame,
):
    # Initialize MediaPipe Selfie Segmentation
    mp_selfie_segmentation = mp.solutions.selfie_segmentation
    segmentation = mp_selfie_segmentation.SelfieSegmentation(model_selection=0)

    # Convert the BGR image to RGB (MediaPipe expects RGB)
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the image and get the segmentation mask
    results = segmentation.process(image_rgb)

    # The segmentation mask is a float array (0.0 to 1.0)
    # where values close to 1.0 indicate the person.
    # Create a binary mask (True for person, False for background)
    # Adjust the threshold (e.g., 0.1) if the segmentation is too aggressive/lenient
    condition = np.stack((results.segmentation_mask,) * 3, axis=-1) > 0.1

    # Create a background:
    # For simplicity, we'll use a solid black background.
    bg_image = np.zeros(frame.shape, dtype=np.uint8)  # Solid black

    # Apply the mask to blend the foreground (person) with the new background
    output_frame = np.where(condition, frame, bg_image)

    return output_frame


def _remove_background_from_video_one_file(
    input_video_path: str, output_folder_path: str, suffix: str = "_bgremoved"
):
    """
    Processes a video to remove the background of the main subject
    using MediaPipe Selfie Segmentation and saves the output to a specified folder.

    The output video will have the same name and extension as the input video,
    with the given suffix added before the file extension.
    The FPS and dimensions of the video will remain unchanged.

    Args:
        input_video_path (str): The full path to the input video file.
        output_folder_path (str): The path to the directory where the processed
                                  video will be saved.
        suffix (str, optional): A suffix to add to the output video filename
                                before the extension. Defaults to "_bg_removed".
    """

    # Input Validation
    if not os.path.exists(input_video_path):
        print(f"Error: Input video path not found: {input_video_path}")
        return

    if not os.path.isfile(input_video_path):
        print(f"Error: Input path is not a file: {input_video_path}")
        return

    if not os.path.exists(output_folder_path):
        print(f"Output directory does not exist. Creating: {output_folder_path}")
        os.makedirs(output_folder_path)
    elif not os.path.isdir(output_folder_path):
        print(f"Error: Output path is not a directory: {output_folder_path}")
        return

    # Open Video Capture
    cap = cv2.VideoCapture(input_video_path)

    if not cap.isOpened():
        print(f"Error: Could not open video file: {input_video_path}")
        return

    # Get Video Properties
    fps, width, height, _, _ = get_video_info(cap)

    # Prepare Output Video Path
    base_name = os.path.basename(input_video_path)
    file_name_without_ext, file_extension = os.path.splitext(base_name)
    output_file_name = f"{file_name_without_ext}{suffix}{file_extension}"
    output_video_path = os.path.join(output_folder_path, output_file_name)

    # Initialize Video Writer
    fourcc_out = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_video_path, fourcc_out, fps, (width, height))

    if not out.isOpened():
        print(f"Error: Could not create video writer for: {output_video_path}")
        cap.release()
        return

    print(f"Processing video: {input_video_path}")
    print(f"Saving output to: {output_video_path}")
    print(f"Dimensions: {width}x{height}, FPS: {fps}")

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break  # End of video
        output_frame = _remove_background_from_video_one_frame(frame)

        # Write the processed frame to the output video
        out.write(output_frame)
        frame_count += 1

    # Release resources
    cap.release()
    out.release()

    print(f"Processing complete. Total frames processed: {frame_count}")
    print(f"Output video saved to: {output_video_path}")


def remove_background_from_video(
    input_folder_path: str,
    output_folder_path: str,
    suffix: str = "_bgremoved",
    parallel: Optional[bool] = False,
    max_workers: Optional[int] = 10,
    maxtasksperchild: Optional[int] = 1,
    use_tqdm: Optional[bool] = False,
):

    create_folders(output_folder_path)

    args_list = [
        {
            "input_video_path": input_video_path,
            "output_folder_path": output_folder_path,
            "suffix": suffix,
        }
        for input_video_path in get_file_paths(
            input_path=input_folder_path, depth=5, EXT_LIST=["mov", "mp4"]
        )
    ]

    print(f"Number of files to process : {len(args_list)}")

    process(
        worker_function=_remove_background_from_video_one_file,
        handler_function=None,
        worker_params_list=args_list,
        parallel=parallel,
        max_workers=max_workers,
        maxtasksperchild=maxtasksperchild,
        use_tqdm=use_tqdm,
    )


def _fit_normalizer_one_file(
    train_val_test_file_path, output_directory_path, chunk_size, feature_columns
):

    import pandas as pd
    from sklearn.preprocessing import StandardScaler

    index = train_val_test_file_path["index"]
    directory_path = os.path.join(output_directory_path, str(index))
    create_folders(directory_path)

    output_file_path = os.path.join(directory_path, str(index) + "_normalizer.pkl")

    scaler = StandardScaler()
    for file_path in train_val_test_file_path["paths"]["train"]:
        print(f"{index} - {file_path}")
        for chunk in pd.read_csv(file_path, chunksize=chunk_size):
            chunk = chunk[feature_columns]
            scaler.partial_fit(chunk)
    return {"obj": scaler, "output_path": output_file_path, "file_type": "pickle"}


def fit_normalizer(
    train_val_test_file_paths: List[Dict],
    output_directory_path: str,
    chunk_size: 1000,
    feature_columns: List[str],
    parallel: bool = True,
    max_workers: int = 10,
    maxtasksperchild: int = 1,
    use_tqdm: bool = True,
):

    create_folders(output_directory_path)

    args_list = [
        {
            "train_val_test_file_path": train_val_test_file_path,
            "output_directory_path": output_directory_path,
            "chunk_size": chunk_size,
            "feature_columns": feature_columns,
        }
        for train_val_test_file_path in train_val_test_file_paths
    ]

    process(
        worker_function=_fit_normalizer_one_file,
        handler_function=save_obj,
        worker_params_list=args_list,
        parallel=parallel,
        max_workers=max_workers,
        maxtasksperchild=maxtasksperchild,
        use_tqdm=use_tqdm,
    )


def _apply_normalizer_one_file(
    train_val_test_file_path: str, output_directory_path: str
):

    import pickle

    from Zh.src.utils.file_folder import create_folders

    index = train_val_test_file_path["index"]
    all_splits_paths = (
        train_val_test_file_path["paths"]["test"]
        + train_val_test_file_path["paths"]["val"]
        + train_val_test_file_path["paths"]["train"]
    )

    origin_directory_path = os.path.commonpath(all_splits_paths)
    origin_directory_subfolders_path = list(
        set([os.path.dirname(file_path) for file_path in all_splits_paths])
    )
    output_directory_subfolders_path = [
        folder_path.replace(origin_directory_path, output_directory_path)
        for folder_path in origin_directory_subfolders_path
    ]

    create_folders(output_directory_subfolders_path)

    with open(
        os.path.join(output_directory_path, f"{index}_normalizer.pkl"), "rb"
    ) as file:
        scaler = pickle.load(file)

    for file_path in all_splits_paths:
        print(f"{index} - {file_path}")

        data = pd.read_csv(file_path)
        scaled = pd.DataFrame(
            scaler.transform(data[scaler.feature_names_in_]),
            columns=scaler.feature_names_in_,
        )
        rest = data[data.columns.difference(scaler.feature_names_in_, sort=False)]
        data = pd.concat([rest, scaled], axis=1)
        data.to_csv(
            file_path.replace(origin_directory_path, output_directory_path), index=False
        )


def apply_normalizer(
    input_directory_path: str,
    parallel: bool = True,
    max_workers: int = 10,
    maxtasksperchild: int = 1,
    use_tqdm: bool = True,
):

    with open(os.path.join(input_directory_path, "split_info.json"), "r") as file:
        train_val_test_file_paths = json.load(file)

    args_list = [
        {
            "train_val_test_file_path": train_val_test_file_path,
            "output_directory_path": os.path.join(
                input_directory_path, str(train_val_test_file_path["index"])
            ),
        }
        for train_val_test_file_path in train_val_test_file_paths
    ][10:]

    process(
        worker_function=_apply_normalizer_one_file,
        worker_params_list=args_list,
        parallel=parallel,
        max_workers=max_workers,
        maxtasksperchild=maxtasksperchild,
        use_tqdm=use_tqdm,
    )


def convert_csv_to_npy(source_dir, target_dir):
    """
    Traverses all files in the source directory and its subdirectories.
    If a file is a CSV, it reads it using pandas and saves it as a NPY file
    in the corresponding subdirectory in the target folder.

    Args:
        source_dir (str): Path to the source directory
        target_dir (str): Path to the target directory where NPY files will be saved

    Returns:
        tuple: (int, int) - Count of processed files and count of errors
    """
    copy_directory_structure(source_dir=source_dir, target_dir=target_dir)

    processed_count = 0
    error_count = 0
    columns = []

    # Walk through all directories and files in source_dir
    for root, _, files in os.walk(source_dir):
        # Calculate relative path and target subdir to maintain directory structure
        relative_path = os.path.relpath(root, source_dir)
        target_subdir = os.path.join(target_dir, relative_path)

        # Process each file in current directory
        for file in files:
            # Check if file is a CSV
            if file.lower().endswith(".csv"):
                if file == "video_info.csv":
                    continue

                source_file = os.path.join(root, file)

                # Generate target NPY filename (replace .csv with .npy)
                target_file = os.path.join(
                    target_subdir, os.path.splitext(file)[0] + ".npy"
                )

                if os.path.exists(target_file):

                    # TODO : problem with this implementation is that the column
                    #        names are not matched with already processed files.

                    logging.info(f"File Exists: {target_file}")
                    continue

                try:
                    # Read CSV using pandas
                    logging.info(f"Processing: {source_file}")
                    df = pd.read_csv(source_file)
                    current_columns = list(df.select_dtypes(include="number").columns)

                    if columns == []:
                        columns = list(df.select_dtypes(include="number").columns)
                        save_obj(
                            obj=columns,
                            output_path=os.path.join(target_dir, "columns.json"),
                            file_type="json",
                        )

                    assert (
                        current_columns == columns
                    ), f"Column names dont match {columns} - {current_columns}"
                    df = df[columns]
                    # Convert to numpy array and save as NPY
                    np.save(target_file, df.to_numpy())

                    logging.info(f"Saved as: {target_file}")
                    processed_count += 1

                except Exception as e:
                    logging.info(f"Error processing {source_file}: {str(e)}")
                    error_count += 1

    logging.info(f"Completed: {processed_count} files processed, {error_count} errors")
    return processed_count, error_count


def copy_video_split_info(source_dir, target_dir):
    """
    Traverses all files in the source directory and its subdirectories.
    If a file is a CSV, it reads it using pandas and saves it as a NPY file
    in the corresponding subdirectory in the target folder.

    Args:
        source_dir (str): Path to the source directory
        target_dir (str): Path to the target directory where NPY files will be saved

    Returns:
        tuple: (int, int) - Count of processed files and count of errors
    """
    # Check if source directory exists

    copy_directory_structure(source_dir=source_dir, target_dir=target_dir)

    processed_count = 0
    error_count = 0

    # Walk through all directories and files in source_dir
    for root, _, files in os.walk(source_dir):
        # Calculate relative path and target subdir to maintain directory structure
        relative_path = os.path.relpath(root, source_dir)
        target_subdir = os.path.join(target_dir, relative_path)

        # Process each file in current directory
        for file in files:
            source_file = os.path.join(root, file)
            target_file = os.path.join(target_subdir, file)

            if file == "split_info.json":
                try:
                    with open(source_file, "r") as json_file:
                        j = json.load(json_file)

                    for fold in j:
                        fold["input_paths"] = fold["output_paths"].copy()
                        splits = ["train", "test", "val"]
                        for split in splits:
                            fold["output_paths"][split] = [
                                path.replace(source_dir, target_dir).replace(
                                    ".csv", ".npy"
                                )
                                for path in fold["output_paths"][split]
                            ]

                    save_obj(j, target_file, "json")
                    print(f"Saved as: {target_file}")
                    processed_count += 1

                except Exception as e:
                    print(f"Error processing {source_file}: {str(e)}")
                    error_count += 1

            if file == "video_info.csv":
                try:
                    # Read CSV using pandas
                    print(f"Processing: {source_file}")

                    df = pd.read_csv(source_file)
                    df["path"] = df["path"].apply(
                        lambda x: x.replace(source_dir, target_dir).replace(
                            ".csv", ".npy"
                        )
                    )

                    df.to_csv(target_file, index=False)
                    print(f"Saved as: {target_file}")
                    processed_count += 1

                except Exception as e:
                    print(f"Error processing {source_file}: {str(e)}")
                    error_count += 1

    print(f"Completed: {processed_count} files processed, {error_count} errors")


if __name__ == "__main__":
    logging.info("Starting ...")
    # from Zh.src.utils.split_files import _get_train_val_test_file_paths

    # input_path = "/work/long_lab/Mojtaba/data/UTA/1.3.2-augmented_embedded_preprocessed"
    # output_directory_path = "/work/long_lab/Mojtaba/data/UTA/1.3.3-augmented_embedded_preprocessed_normalized"

    # train_val_test_file_paths = _get_train_val_test_file_paths(
    #     input_path = input_path,
    #     output_path=output_directory_path,
    #     primary_column_name = "fold",
    #     secondary_column_name = None
    # )

    # save_obj(obj=train_val_test_file_paths,
    #          output_path=os.path.join(output_directory_path, 'split_info_1.json'),
    #          file_type='json')

    # fit_normalizer(train_val_test_file_paths = train_val_test_file_paths,
    #                 output_directory_path = output_directory_path,
    #                 chunk_size = 1000,
    #                 feature_columns = [f"f_{i}" for i in range(128)],
    #                 parallel = True,
    #                 max_workers = 10,
    #                 maxtasksperchild = 1,
    #                 use_tqdm = True)

    # apply_normalizer(input_directory_path = output_directory_path,
    #                  parallel = True,
    #                  max_workers = 10,
    #                  maxtasksperchild = 1,
    #                  use_tqdm = True)

    # convert_csv_to_npy(
    #     source_dir="/work/long_lab/Mojtaba/data/UTA/1.3.3-augmented_embedded_preprocessed_normalized",
    #     target_dir="/work/long_lab/Mojtaba/data/UTA/1.3.4-augmented_embedded_preprocessed_normalized_npy",
    # )

    # TODO check and see if all the output dataframes have the same shape as inputs
    pass

import argparse
import logging
import os
from typing import List, Optional

import numpy as np
import pandas as pd

from Zh.src.utils.common import load_npz
from Zh.src.utils.file_folder import create_folders, pick_selected_file_paths
from Zh.src.utils.normalization import FileStats, stats_to_mean_std

logging.basicConfig(
    level=logging.INFO,  # or DEBUG for more verbosity
    format="[%(asctime)s][%(levelname)s] %(message)s",
)


def aggregate_normalization_info(
    input_folder: str,
    output_path: str,
    relative_selected_paths: Optional[List[str]] = None,
    depth: int = 1,
    file_indices_spec: str = "",
):

    create_folders(os.path.dirname(output_path))

    file_index_to_path = pick_selected_file_paths(
        input_folder=input_folder,
        depth=depth,
        EXT_LIST=[".npz"],
        relative_selected_paths=relative_selected_paths,
        file_indices_spec=file_indices_spec,
        sort_and_dedupe=True,
    )

    df_list = []

    for file_index, file_path in file_index_to_path.items():
        df = {}

        file_stats = load_npz(file_path)
        file_stats = FileStats(
            n=file_stats["n"],
            mean=file_stats["mean"],
            M2=file_stats["M2"],
        )

        # mean, std = stats_to_mean_std(file_stats)

        df["file_index"] = file_index
        df["subject_id"] = int(os.path.basename(file_path).split("__")[0])
        df["modality"] = os.path.basename(file_path).split("__")[1]
        df["n"] = file_stats.n
        df["mean"] = file_stats.mean
        df["M2"] = file_stats.M2
        # df["mean"] = mean
        # df["std"] = std

        df_list.append(df)

    df = pd.DataFrame(df_list)
    df.to_csv(output_path, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create a DataFrame from .npz files in a directory."
    )

    parser.add_argument(
        "input_folder",
        type=str,
        help="Path to the input folder containing .npz files.",
    )

    parser.add_argument(
        "output_path",
        type=str,
        help="Path to save the output DataFrame as a .csv file.",
    )
    parser.add_argument(
        "relative_selected_paths",
        nargs="*",
        help="List of relative file path to process. If empty, all files in the input folder will be processed.",
    )
    parser.add_argument(
        "--depth",
        type=int,
        default=1,
        help="Number of directories to traverse from the input folder.",
    )
    parser.add_argument(
        "--file_indices_spec",
        type=str,
        default="",
        help="Comma-separated indices/ranges, e.g. '0-9,20,35-'",
    )

    args = parser.parse_args()

    logging.info("Starting to aggregate normalization info...")

    aggregate_normalization_info(
        input_folder=args.input_folder,
        output_path=args.output_path,
        relative_selected_paths=args.relative_selected_paths,
        depth=args.depth,
        file_indices_spec=args.file_indices_spec,
    )

    logging.info("Finished aggregating normalization info.")

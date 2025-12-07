import argparse
import logging
import os
from collections import defaultdict
from typing import Iterable, Optional

from Zh.src.utils.common import load_npz, save_npz
from Zh.src.utils.file_folder import *
from Zh.src.utils.normalization import *
from Zh.src.utils.serialized_parallel import process

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s][%(levelname)s] %(message)s",
)


def get_subject_id_from_video_id(video_id: str, dataset_name: str) -> int:
    if dataset_name == "UTA":
        return int(os.path.basename(video_id).split("_")[1])
    elif dataset_name == "DROZY":
        return int(os.path.basename(video_id).split("-")[0])
    else:
        raise ValueError(f"Unknown dataset for subject ID extraction: {dataset_name}")


def process_single_video(
    output_file_path: str,
    file_paths: Iterable[str],
    subject_id: Optional[str] = None,
):
    logging.info("=" * 50)
    logging.info(f"Processing file {subject_id}: {len(file_paths)}")
    logging.info(file_paths)

    file_stats = []
    for file_path in file_paths:
        data = load_npz(file_path)
        file_stats.append(
            FileStats(
                n=data["n"],
                mean=data["mean"],
                M2=data["M2"],
            )
        )

    aggregated_stats = aggregate_stats_for_files(file_stats)

    save_npz(
        path=output_file_path,
        arrays={
            "n": aggregated_stats.n,
            "mean": aggregated_stats.mean,
            "M2": aggregated_stats.M2,
        },
    )

    logging.info(f"Done processing file {subject_id}: saved to {output_file_path}")


def process_multiple_videos(
    input_folder: str,
    output_folder: str,
    relative_selected_paths: Iterable[str],
    dataset_name: str = "DROZY",
    file_name_suffix: str = "",
    parallel: bool = False,
    max_workers: Optional[int] = None,
    maxtasksperchild: Optional[int] = None,
    use_tqdm: bool = True,
):

    create_folders(output_folder)

    file_index_to_path = pick_selected_file_paths(
        input_folder=input_folder,
        depth=3,
        EXT_LIST=[".npz"],
        relative_selected_paths=relative_selected_paths,
        file_indices_spec="",
        sort_and_dedupe=True,
    )

    subjectwise_split_file_index_to_path = defaultdict(list)

    for _, file_path in file_index_to_path.items():
        subject_id = get_subject_id_from_video_id(
            video_id=file_path,
            dataset_name=dataset_name,
        )
        subjectwise_split_file_index_to_path[subject_id].append(file_path)

    args_list = [
        {
            "output_file_path": os.path.join(
                output_folder,
                f"{subject_id}__{file_name_suffix}__normalization_info.npz",
            ),
            "file_paths": file_paths,
            "subject_id": subject_id,
        }
        for subject_id, file_paths in subjectwise_split_file_index_to_path.items()
    ]
    logging.info(f"Processing {len(args_list)} subjects")

    process(
        worker_function=process_single_video,
        worker_params_list=args_list,
        handler_function=None,
        parallel=parallel,
        max_workers=max_workers,
        maxtasksperchild=maxtasksperchild,
        use_tqdm=use_tqdm,
        auto_slice=False,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Find normalization info for dataset")

    parser.add_argument(
        "input_folder",
        type=str,
        help="Path to the input folder containing video files.",
    )
    parser.add_argument(
        "output_folder",
        type=str,
        help="Path to the output folder to save processed videos.",
    )
    parser.add_argument(
        "relative_selected_paths",
        nargs="*",
        help="List of relative file path to process. If empty, all files in the input folder will be processed.",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="DROZY",
        choices=["DROZY", "UTA"],
        help="Name of the dataset.",
    )
    parser.add_argument(
        "--file_name_suffix",
        type=str,
        default="",
        help="Suffix of the file name to process.",
    )
    parser.add_argument(
        "--parallel",
        action="store_true",
        help="Process videos in parallel.",
    )
    parser.add_argument(
        "--max_workers",
        type=int,
        default=None,
        help="Maximum number of workers for parallel processing.",
    )
    parser.add_argument(
        "--maxtasksperchild",
        type=int,
        default=None,
        help="Maximum tasks per child process.",
    )
    parser.add_argument(
        "--no_tqdm",
        action="store_false",
        dest="use_tqdm",
        default=True,
        help="Disable tqdm progress bars.",
    )
    args = parser.parse_args()

    logging.info("Starting ...")

    process_multiple_videos(
        input_folder=args.input_folder,
        output_folder=args.output_folder,
        relative_selected_paths=args.relative_selected_paths,
        dataset_name=args.dataset_name,
        file_name_suffix=args.file_name_suffix,
        parallel=args.parallel,
        max_workers=args.max_workers,
        maxtasksperchild=args.maxtasksperchild,
        use_tqdm=args.use_tqdm,
    )

    logging.info("Done!")

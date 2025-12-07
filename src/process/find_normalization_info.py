from typing import Iterable, Optional

from Zh.src.utils.file_folder import *
from Zh.src.utils.memmap_handler import MemmapHandler
from Zh.src.utils.serialized_parallel import process
from Zh.src.utils.normalization import *
from Zh.src.utils.common import save_npz, update_file_name


def process_single_video(
    input_folder: str,
    output_folder: str,
    file_path: str,
    batch_size: int = 10000,
    file_index: Optional[int] = 0,
):
    logging.info(f"Processing file {file_index}: {file_path}")

    mh_input = MemmapHandler(
        path=file_path,
        mode="r",
    )

    file_stats = compute_file_stats_from_batches(
        mh_input.iter_batches(batch_size=batch_size, as_numpy=True)
    )

    output_file_path = update_file_name(
        file_path = file_path,
        input_folder = input_folder,
        output_folder = output_folder,
        suffix = "_normalization_info",
        extension = ".npz",
    )

    save_npz(
        path=output_file_path,
        arrays={
            "n" : file_stats.n,
            "mean" : file_stats.mean,
            "M2" : file_stats.M2,
        },
    )


def process_multiple_videos(
    input_folder: str,
    output_folder: str,
    relative_selected_paths: Iterable[str],
    file_indices_spec: str = "",
    batch_size: int = 10000,
    parallel: bool = False,
    max_workers: Optional[int] = None,
    maxtasksperchild: Optional[int] = None,
    use_tqdm: bool = True,
):
    if not output_folder or output_folder == input_folder:
        output_folder = input_folder
    else:
        copy_directory_structure(input_folder, output_folder)

    file_index_to_path = pick_selected_file_paths(
        input_folder=input_folder,
        depth=3,
        EXT_LIST=[".dat"],
        relative_selected_paths=relative_selected_paths,
        file_indices_spec=file_indices_spec,
        sort_and_dedupe=True,
    )

    args_list = [
        {
            "input_folder": input_folder,
            "output_folder": output_folder,
            "file_path": file_path,
            "batch_size": batch_size,
            "file_index": file_index,
        }
        for file_index, file_path in file_index_to_path.items()
    ]

    process(
        worker_function=process_single_video,
        worker_params_list=args_list,
        handler_function=None,
        parallel=parallel,
        max_workers=max_workers,
        maxtasksperchild=maxtasksperchild,
        use_tqdm=use_tqdm,
        auto_slice=True,
    )


if __name__ == "__main__":
    import argparse
    import logging

    logging.basicConfig(
        level=logging.INFO,  # or DEBUG for more verbosity
        format="[%(asctime)s][%(levelname)s] %(message)s",
    )

    parser = argparse.ArgumentParser(description="Find normalization info for dataset")

    parser.add_argument(
        "input_folder",
        type=str,
        help="Path to the input folder containing video files.",
    )
    parser.add_argument(
        "relative_selected_paths",
        nargs="*",
        help="List of relative file path to process. If empty, all files in the input folder will be processed.",
    )
    parser.add_argument(
        "--output_folder",
        type=str,
        default="",
        help="Path to the output folder to save processed videos, if not specified, input_folder is used.",
    )
    parser.add_argument(
        "--file_indices_spec",
        type=str,
        default="",
        help="Comma-separated indices/ranges, e.g. '0-9,20,35-'",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1000,
        help="Batch size for processing.",
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
        file_indices_spec=args.file_indices_spec,
        batch_size=args.batch_size,
        parallel=args.parallel,
        max_workers=args.max_workers,
        maxtasksperchild=args.maxtasksperchild,
        use_tqdm=args.use_tqdm,
    )

    logging.info("Done!")

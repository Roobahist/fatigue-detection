from typing import List, Optional, Tuple, Union

import cv2


def get_video_info(
    cap: Optional[cv2.VideoCapture] = None, file_path: Optional[str] = None
) -> Tuple[int, int, int, int]:
    """
    Retrieves video information from a cv2.VideoCapture object.

    Args:
        cap: The cv2.VideoCapture object.
        file_path: The path to the video file.

        Raises:
        ValueError: If both `cap` and `file_path` are provided, or if neither is provided.

    Returns:
        A tuple containing (fps, width, height, total_frames) as integers.
    """

    if (cap is None and not file_path) or (cap is not None and file_path):
        raise ValueError(
            "Provide either `cap` or `file_path`, but not both or neither."
        )
    if cap is None:
        cap = cv2.VideoCapture(file_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if file_path is not None:
        cap.release()

    return fps, width, height, total_frames


def get_video_info_from_csv(
    file_paths: Union[str,List[str]],
    output_directory_path: Union[str,None] = None,
    output_file_prefix: Union[str,None] = "video_info",
):

    import os

    import pandas as pd

    from Zh.src.utils.common import find_new_file_name, get_csv_length

    if isinstance(file_paths, str):
        file_paths = [file_paths]

    data = pd.DataFrame()
    data["path"] = file_paths
    data["total_frames"] = data["path"].apply(
        lambda x: get_csv_length(file_path=x, skip_header=True)
    )

    if output_directory_path and output_file_prefix:
        file_name = find_new_file_name(
            output_directory_path=output_directory_path,
            file_name_prefix=output_file_prefix,
            file_name_extension="csv",
        )
        data.to_csv(os.path.join(output_directory_path, file_name), index=False)

    return data


# if __name__ == "__main__":
#     from Zh.src.utils.file_folder import get_file_paths

#     input_path = "/work/long_lab/Mojtaba/data/UTA/1.3.2-augmented_embedded_preprocessed"
#     file_paths = get_file_paths(folder_path=input_path, depth=3, EXT_LIST=[".csv"])
#     _ = get_video_info_from_csv(file_paths = file_paths, output_directory_path = input_path, output_file_prefix = "video_info")
import logging
import os

from Zh.src.preprocess.preprocessing import preprocess_data
from Zh.src.utils.file_folder import create_folders, get_file_paths
from Zh.src.utils.load_data import load_data

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    logger = logging.getLogger(__name__)

    input_path = "/work/long_lab/Mojtaba/data/UTA/1.3.1-augmented_embedded"
    output_path = "/work/long_lab/Mojtaba/data/UTA/1.3.2-augmented_embedded_preprocessed"
    mapping = {"10": 8.5, "5": 6.5, "0": 2}

    logger.info("Starting preprocessing of augmented data.")
    logger.info(f"Mapping used for preprocessing: {mapping}")

    file_paths = get_file_paths(folder_path=input_path, depth=3, EXT_LIST=[".csv"])

    input_directory_folders = list(set([os.path.dirname(i) for i in file_paths]))
    folders_to_create = [
        i.replace(input_path, output_path) for i in input_directory_folders
    ]
    create_folders(folders_to_create)

    for file_path in file_paths:
        unpreprocessed_data = load_data(input_path=file_path, file_names=[])
        preprocessed_data, columns_to_drop = preprocess_data(
            data=unpreprocessed_data, mapping=mapping
        )

        output_file_path = file_path.replace(input_path, output_path)
        root, ext = os.path.splitext(output_file_path)
        output_file_path = root + "_preprocessed" + ext

        preprocessed_data.to_csv(output_file_path, index=False)

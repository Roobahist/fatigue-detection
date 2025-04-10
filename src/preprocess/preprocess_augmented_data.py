import os

from Zh.src.preprocess.preprocessing import preprocess_data
from Zh.src.utils.file_folder import create_folders
from Zh.src.utils.load_data import load_data
import logging

if __name__ == "__main__":
    path_to_data = "/work/long_lab/Mojtaba/data/UTA/1.3.2-augmented_embedded"
    output_path = "/work/long_lab/Mojtaba/data/UTA/1.3.3-preprocessed"
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    logger = logging.getLogger(__name__)

    logger.info("Starting preprocessing of augmented data.")

    mapping = {"10": 8.5, "5": 6.5, "0": 2}
    logger.info(f"Mapping used for preprocessing: {mapping}")

    unpreprocessed_data = load_data(path_to_data=path_to_data, file_names=[])
    preprocessed_data, columns_to_drop = preprocess_data(
        data=unpreprocessed_data, mapping=mapping
    )

    create_folders(to_create_folder_paths=[output_path])

    output_file_name = "facenet_preprocessed_alternativemapping.csv"
    preprocessed_data.to_csv(os.path.join(output_path, output_file_name), index=False)

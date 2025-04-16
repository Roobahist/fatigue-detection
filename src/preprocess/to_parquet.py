from Zh.src.utils.load_data import csv_to_parquet

if __name__ == "__main__":
    file_path = "/work/long_lab/Mojtaba/data/UTA/1.3.3-preprocessed/facenet_preprocessed_alternativemapping.csv"
    _ = csv_to_parquet(file_path = file_path)
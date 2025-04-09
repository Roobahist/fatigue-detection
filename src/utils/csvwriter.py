import os
import csv

class CSVWriter:
    def __init__(self, file_path):
        self.file_path = file_path
        self.ensure_newline()
        self.file = open(file_path, 'a+', newline='')
        self.writer = csv.writer(self.file)

    def ensure_newline(self):
        if os.path.exists(self.file_path):
            with open(self.file_path, 'rb+') as f:
                f.seek(0, os.SEEK_END)
                if f.tell() > 0:
                    f.seek(-1, os.SEEK_END)
                    if f.read(1) != b'\n':
                        f.write(b'\n')

    def write(self, row):
        self.writer.writerow(row)
        self.file.flush()

    def get_current_line_number(self):
        self.file.seek(0)
        return sum(1 for _ in self.file)
    
    def close(self):
        self.file.close()


# CSVW = CSVWriter(
#     file_path="/work/long_lab/Mojtaba/data/UTA/1.3.1-augmentated_embedded/embedding/Facenet/cropped/1_06_5_30fps_cropped.csv"
# )
# print(CSVW.get_current_line_number())
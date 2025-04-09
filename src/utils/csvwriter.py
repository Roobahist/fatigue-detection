class CSVWriter:
    def __init__(self, filename):
        import csv
        self.file = open(filename, 'a+', newline='')
        self.writer = csv.writer(self.file)

    def write(self, row):
        self.writer.writerow(row)
        self.file.flush()

    def current_line(self):
        self.file.seek(0)
        return sum(1 for _ in self.file)
    
    def close(self):
        self.file.close()


# CSVW = CSVWriter(
#     filename="/work/long_lab/Mojtaba/data/UTA/1.3.1-augmentated_embedded/embedding/Facenet/cropped/1_06_5_30fps_cropped.csv"
# )
# print(CSVW.current_line())
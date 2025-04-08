class CSVWriter:
    def __init__(self, filename):
        import csv
        self.file = open(filename, 'a', newline='')
        self.writer = csv.writer(self.file)

    def write(self, row):
        self.writer.writerow(row)
        self.file.flush()

    def close(self):
        self.file.close()

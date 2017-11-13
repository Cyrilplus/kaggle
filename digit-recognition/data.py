import csv
import numpy as np

def train():
    def reader():
        with open('./dataset/train.csv', 'r') as f:
            csv_reader = csv.reader(f)
            next(csv_reader)
            for row in csv_reader:
                row = map(lambda v: int(v), row)
                label = row[0]
                img = np.array(row[1:], dtype=np.float32) / 255.0
                yield (img, label)
    return reader

def test():
    def reader():
        with open('./dataset/test.csv', 'r') as f:
            csv_reader = csv.reader(f)
            next(csv_reader)
            for row in csv_reader:
                img = np.array(map(lambda v: int(v), row), dtype=np.float32) / 255.0
                yield (img, )
    return reader


def save_result2csv(filename, labels):
    with open(filename, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['ImageId', 'Label'])
        indexes = np.arange(1, (len(labels) + 1))
        rows = np.column_stack((indexes, labels))
        writer.writerows(rows)

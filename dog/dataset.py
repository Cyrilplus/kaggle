import os
import csv
import numpy as np
from PIL import Image


def _init():
    index2label = []
    img_label = {}
    with open('./dataset/labels.csv', 'r') as f:
        csv_reader = csv.reader(f)
        next(csv_reader)
        label_set = set()
        for row in csv_reader:
            img, label = row
            label_set.add(label)
            img_label[img] = label
        # label encoding by array index to label
        index2label = list(label_set)
        index2label.sort() # sorted by charactor
        label2index = {l: i for i, l in enumerate(index2label)}
        img_label = {img: label2index[label] for img, label in img_label.items()}
    return img_label, index2label
        
_img_label, _index2label = _init()

def train():
    def reader():
        for img in _img_label.keys():
            print img
            img_data = Image.open('./dataset/train/' + img + '.jpg').convert('RGB')
            yield np.array(img_data).astype(np.float32) / 255.0, _img_label[img]
    return reader
        

if __name__ == '__main__':
    reader = train()()
    img, label = next(reader)
    print img, label
    print _index2label[label]
    print _index2label



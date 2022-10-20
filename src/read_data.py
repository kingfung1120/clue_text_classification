import json
import os
import pandas as pd
import numpy as np
from datetime import datetime

from sklearn.model_selection import train_test_split
import torch
from transformers import AutoTokenizer

DATA_PATH = 'iflytek_public'


def read_file(file_name):
    li = []
    with open(os.path.join(DATA_PATH, file_name), 'rb') as f:
        for line in f.readlines():
            li.append(json.loads(line))
    return li


def get_labels_df():
    labels = read_file('labels.json')
    labels_df = pd.DataFrame.from_records(labels, index='label')
    return labels_df


def get_all_data_df():
    train = read_file('train.json')
    test = read_file('dev.json')
    train_df = pd.DataFrame.from_records(train)
    test_df = pd.DataFrame.from_records(test)

    model_train_df, valid_df = train_test_split(train_df, test_size=0.3, random_state=123)

    return model_train_df, valid_df, test_df


TOKENIZER = AutoTokenizer.from_pretrained('bert-base-chinese')


class Dataset(torch.utils.data.Dataset):

    def __init__(self, df, max_length=512, num_class=119):

        self.labels = [int(label) for label in df['label']]
        self.texts = [TOKENIZER(text, 
                               padding='max_length', max_length = max_length, truncation=True,
                                return_tensors="pt") for text in df['sentence']]

    def classes(self):
        return self.labels

    def __len__(self):
        return len(self.labels)

    def get_batch_labels(self, idx):
        return np.array(self.labels[idx])

    def get_batch_texts(self, idx):
        return self.texts[idx]

    def __getitem__(self, idx):

        batch_texts = self.get_batch_texts(idx)
        batch_y = self.get_batch_labels(idx)

        return batch_texts, batch_y



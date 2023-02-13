import os
import numpy as np

import torch
from torchtext.datasets import IMDB
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.dataset import random_split
from torchtext.data.functional import to_map_style_dataset

import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import time
def make_dataset(args):
    
    tokenizer = get_tokenizer('basic_english')
    def yield_tokens(data_iter):
        for _, text in data_iter:
            yield tokenizer(text)

    train_iter = IMDB(split='train')
    vocab = build_vocab_from_iterator(yield_tokens(train_iter), specials=["<unk>"])
    vocab.set_default_index(vocab["<unk>"])
    text_pipeline = lambda x: vocab(tokenizer(x))
    label_pipeline = lambda x: 0 if x=='neg' else 1

    train_iter = IMDB(split='train')
    train_data_list = list(iter(train_iter))
    train_data, train_labels = [],[]
    
    max_len = 256
    for _label,_text in train_data_list:
        processed_text = torch.tensor(text_pipeline(_text), dtype=torch.int64)
        processed_text = F.pad(input=processed_text, pad=(0,max_len-len(processed_text)), mode='constant', value=0)
        train_data.append(processed_text)
        train_labels.append(label_pipeline(_label))

    test_iter = IMDB(split='test')
    test_data_list = list(iter(test_iter))
    val_data,val_labels = [],[]

    max_len = 256
    for _label,_text in test_data_list:
        processed_text = torch.tensor(text_pipeline(_text), dtype=torch.int64)
        processed_text = F.pad(input=processed_text, pad=(0,max_len-len(processed_text)), mode='constant', value=0)
        val_data.append(processed_text)
        val_labels.append(label_pipeline(_label))
    return train_data, train_labels, val_data, val_labels

def get_dataset(args):
    train_data, train_labels, val_data, val_labels = make_dataset(args)
    trainset = MiniImagenet84(args, train_data, train_labels, train = "True")
    testset = MiniImagenet84(args, val_data, val_labels, train = "False")
    return trainset, testset

class MiniImagenet84(Dataset):
    def __init__(self, args, data, labels, train = None,sample_indexes=None):
        # self.root = os.path.expanduser(args.train_root)
        self.train = train

        self.data = data
        self.labels = labels
        self.train_data = self.data
        self.train_labels = self.labels
        self.indexes = np.arange(len(data))

        self.args = args
        # self.num_classes = self.args.num_classes
        if sample_indexes is not None:
            self.train_data = self.train_data[sample_indexes]
            self.train_labels = np.array(self.train_labels)[sample_indexes]

        self.train_samples_idx = []
        self.train_probs = np.ones(len(self.labels))*(-1)
        self.avg_probs = np.ones(len(self.labels))*(-1)
        self.times_seen = np.ones(len(self.labels))*1e-6


    def __getitem__(self, index):

        text, labels = self.data[index], self.labels[index]
        return text, labels, index
        # if self.train == "True":
        #     return text, labels, index
        # elif self.train == "False":
        #     return text, labels
        

    def __len__(self):
        return len(self.data)

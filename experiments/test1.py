# import torch
# from torchtext.legacy import data, datasets, vocab
# import time
# from torch.utils.data import DataLoader, Dataset
# import numpy as np

# def d(tensor=None):
#     """
#     Returns a device string either for the best available device,
#     or for the device corresponding to the argument
#     :param tensor:
#     :return:
#     """
#     if tensor is None:
#         return 'cuda' if torch.cuda.is_available() else 'cpu'
#     return 'cuda' if tensor.is_cuda else 'cpu'


# class IMDB_text_train(Dataset):
#     def __init__(self, arg, data, labels, split='train',sample_indexes=None):
#         # self.root = os.path.expanduser(arg.train_root)

#         self.train_data = data
#         self.train_labels = labels

#         # self.arg = arg

#         if sample_indexes is not None:
#             self.train_data = self.train_data[sample_indexes]
#             self.train_labels = np.array(self.train_labels)[sample_indexes]

#         if split == 'train':
#             self.train_samples_idx = []
#             self.train_probs = np.ones(len(labels))*(-1)
#             self.avg_probs = np.ones(len(labels))*(-1)
#             self.times_seen = np.ones(len(labels))*1e-6


#     def __getitem__(self, index):

#         text, labels = self.data[index], self.labels[index]
#         if self.train == "True":
#             return text, labels, index
#         elif self.train == "False":
#             return text, labels

#     def __len__(self):
#         return len(self.data)

# def get_datsets():
#     TEXT = data.Field(lower=True, include_lengths=True, batch_first=True)
#     LABEL = data.Field(sequential=False)
#     train, test = datasets.IMDB.splits(TEXT, LABEL)

#     TEXT.build_vocab(train, max_size=50000 - 2) # - 2 to make space for <unk> and <pad>
#     LABEL.build_vocab(train)

#     train,test = data.BucketIterator.splits((train, test), batch_size=4)
#     print(train.__dict__)

#     train_data, train_label = [],[]
#     for each in train:
#         train_label.append(each.label)
#         train_data.append(each.text)

#     test_data, test_label = [],[]
#     for each in test:
#         test_label.append(each.label)
#         test_data.append(each.text)
#     train_data_1 = IMDB_text_train('arg',train_data,train_label,split='train')
#     test_data_1 = IMDB_text_train('arg',test_data,test_label,split='test')
#     return train_data_1,test_data_1



import os
import numpy as np

import torch
import torchtext.datasets
from torchtext.datasets import IMDB
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.dataset import random_split
from torchtext.data.functional import to_map_style_dataset

import torch.nn.functional as F

import torchvision as tv

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
        if self.train == "True":
            return text, labels, index
        elif self.train == "False":
            return text, labels

    def __len__(self):
        return len(self.data)


# class IMDB_test(Dataset):
#     def __init__(self, args, split='test', sample_indexes = None):
#         super().__init__()
#         test_data_list = list(iter(test_iter))
#         test_text_list = []
#         test_label_list = []
#         for _label,_text in test_data_list:
#             processed_text = torch.tensor(text_pipeline(_text), dtype=torch.int64)
#             test_text_list.append(processed_text)
#             test_label_list.append(label_pipeline(_label))
#         self.test_text_list = test_text_list
#         self.test_label_list = test_label_list

#     def __len__(self):
#         return len(self.test_text_list)

# class IMDBdata(Dataset):
#     def __init__(self, args, split='train', sample_indexes = None):
#         # super().__init__()
#         self.args = args

#         # if sample_indexes is not None:
#         #     self.train_data = self.train_data[sample_indexes]
#         #     self.train_labels = np.array(self.train_labels)[sample_indexes]

#         # self.num_classes = self.args.num_classes

        
#         self.vocab = vocab
#         self.train_data = text_list

#         self.labels = np.asarray(label_list, dtype=np.long)

#         self.train_samples_idx = []
#         self.train_probs = np.ones(len(self.labels))*(-1)
#         self.avg_probs = np.ones(len(self.labels))*(-1)
#         self.times_seen = np.ones(len(self.labels))*1e-6


#     def __getitem__(self, index):
#         text, labels = self.data[index], self.labels[index]
#         img = Image.fromarray(img)

#         if self.transform is not None:
#             img = self.transform(img)

#         if self.target_transform is not None:
#             labels = self.target_transform(labels)

#         return text, labels, index
    
#     def __len__(self):
#         return len(self.train_data)

from torchtext.legacy import data, datasets, vocab
# from torchtext.legacy.data import Dataset
from torch.utils.data import DataLoader, Dataset
import numpy as np
import torch
from former.util import d

TEXT = data.Field(lower=True, include_lengths=True, batch_first=True)
LABEL = data.Field(sequential=False)

class IMDB_textdata(Dataset):
    def __init__(self, args, data, train = None,sample_indexes=None):
        self.train = train

        self.data = data
        self.train_data = self.data
        self.args = args
        self.fields = {'text': TEXT, 'label': LABEL}
        self.num_classes = 2
        if sample_indexes is not None:
            self.train_data = self.train_data[sample_indexes]
            # self.train_labels = np.array(self.train_labels)[sample_indexes]
        self.train_samples_idx = []
        self.train_probs = np.ones(len(self.data))*(-1)
        self.avg_probs = np.ones(len(self.data))*(-1)
        self.times_seen = np.ones(len(self.data))*1e-6


    def __getitem__(self, index):

        text_label = self.data[index]
        if self.train == "True":
            return text_label
        elif self.train == "False":
            return text_label

    def __len__(self):
        return len(self.data)


def get_dataset(args):

    train, test = datasets.IMDB.splits(TEXT, LABEL)
    vocab_size = 50000
    TEXT.build_vocab(train, max_size=vocab_size - 2)
    LABEL.build_vocab(train)
    print(train[2].__dict__)
    trainset = IMDB_textdata(args, train, train = "True")
    testset = IMDB_textdata(args, test, train = "False")
    print('!!', trainset[2].__dict__)
    # train_iter, test_iter = data.BucketIterator.splits((trainset, testset), batch_size=4, \
    #                                         device=d())
    # print(len(train_iter))

    train_iter = torch.utils.data.DataLoader(trainset, batch_size=4, \
                                                shuffle=True, \
                                                pin_memory=True, \
                                                drop_last = True)

    return train_iter

args = ''
train_x = get_dataset(args)
for batch in train_x:
    print(batch.text[0])
    break
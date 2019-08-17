from _context import former
from former import util

from util import d, here

import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F

from torchtext import data, datasets, vocab

import numpy as np

from argparse import ArgumentParser
from torch.utils.tensorboard import SummaryWriter

import random, tqdm, sys, math, gzip

# Used for converting between nats and bits
LOG2E = math.log2(math.e)
TEXT = data.Field(lower=True, include_lengths=True, batch_first=True)
LABEL = data.Field(sequential=False)
NUM_CLS = 2

def go(arg):
    """
    Creates and trains a basic transformer for the IMDB sentiment classification task.
    """

    # load the IMDB data
    if arg.final:
        train, test = datasets.IMDB.splits(TEXT, LABEL)

        TEXT.build_vocab(train, max_size=arg.vocab_size - 2)
        LABEL.build_vocab(train)

        train_iter, test_iter = data.BucketIterator.splits((train, test), batch_size=arg.batch_size, device=util.d())
    else:
        tdata, _ = datasets.IMDB.splits(TEXT, LABEL)
        train, test = tdata.split(split_ratio=0.8)

        TEXT.build_vocab(train, max_size=arg.vocab_size - 2) # - 2 to make space for <unk> and <pad>
        LABEL.build_vocab(train)

        train_iter, test_iter = data.BucketIterator.splits((train, test), batch_size=arg.batch_size, device=util.d())

    print(f'- nr. of training examples {len(train_iter)}')
    print(f'- nr. of {"test" if arg.final else "validation"} examples {len(test_iter)}')

    mx = max([input.text[0].size(1) for input in train_iter])
    mx = mx * 2

    print(f'- maximum sequence length: {mx}')

    # create the model
    model = former.CTransformer(emb=arg.embedding_size, heads=arg.num_heads, depth=arg.depth, seq_length=mx, num_tokens=arg.vocab_size, num_classes=NUM_CLS)
    if torch.cuda.is_available():
        model.cuda()
    opt = torch.optim.Adam(lr=arg.lr, params=model.parameters())

    # training loop
    for e in range(arg.num_epochs):

        print(f'\n epoch {e}')
        for batch in tqdm.tqdm(train_iter):
            opt.zero_grad()

            input = batch.text[0]
            label = batch.label - 1

            out = model(input)
            loss = F.nll_loss(out, label)

            loss.backward()
            opt.step()

        with torch.no_grad():
            tot, cor= 0.0, 0.0

            for batch in test_iter:
                input = batch.text[0]

                label = batch.label - 1

                out = model(input).argmax()

                tot += float(input.size(0))
                cor += float((label == out).sum().item())

            acc = cor / tot
            print(f'-- {"validation" if arg.final else "test"} accuracy {acc:.3}')

if __name__ == "__main__":

    parser = ArgumentParser()

    parser.add_argument("-e", "--num-epochs",
                        dest="num_epochs",
                        help="Number of epochs.",
                        default=80, type=int)

    parser.add_argument("-b", "--batch-size",
                        dest="batch_size",
                        help="The batch size.",
                        default=64, type=int)

    parser.add_argument("-l", "--learn-rate",
                        dest="lr",
                        help="Learning rate",
                        default=0.0001, type=float)

    parser.add_argument("-T", "--tb_dir", dest="tb_dir",
                        help="Tensorboard logging directory",
                        default='./runs')

    parser.add_argument("-f", "--final", dest="final",
                        help="Whether to run on the real test set (if not included, the validation set is used).",
                        action="store_true")

    parser.add_argument("-E", "--embedding", dest="embedding_size",
                        help="Size of the character embeddings.",
                        default=128, type=int)

    parser.add_argument("-V", "--vocab-size", dest="vocab_size",
                        help="Number of words in the vocabulary.",
                        default=50_000, type=int)


    parser.add_argument("-H", "--heads", dest="num_heads",
                        help="Number of attention heads.",
                        default=8, type=int)

    parser.add_argument("-d", "--depth", dest="depth",
                        help="Depth of the network (nr of self-attention layers)",
                        default=4, type=int)

    parser.add_argument("-r", "--random-seed",
                        dest="seed",
                        help="RNG seed. Negative for random",
                        default=1, type=int)

    options = parser.parse_args()

    print('OPTIONS ', options)

    go(options)

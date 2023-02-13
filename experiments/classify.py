import former
from former import util
from former.util import d, here

import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F

# from torchtext import data, datasets, vocab
from torchtext.legacy import data, datasets, vocab
import pickle

import numpy as np

from argparse import ArgumentParser
from torch.utils.tensorboard import SummaryWriter

import random, tqdm, sys, math, gzip

from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
import time
from get_data import *
from utils_train import *

import os
from copy import deepcopy
from datetime import datetime
# Used for converting between nats and bits
LOG2E = math.log2(math.e)
TEXT = data.Field(lower=True, include_lengths=True, batch_first=True)
LABEL = data.Field(sequential=False)
NUM_CLS = 2



def go(arg):
    """
    Creates and trains a basic transformer for the IMDB sentiment classification task.
    """
    tbw = SummaryWriter(log_dir=arg.tb_dir) # Tensorboard logging

    trainset, testset = get_dataset(arg)
    train_iter = torch.utils.data.DataLoader(trainset, batch_size=arg.batch_size, \
                                                shuffle=True, \
                                                pin_memory=True, \
                                                drop_last = True)
    test_iter = torch.utils.data.DataLoader(testset, batch_size=arg.batch_size, shuffle=True,pin_memory=True)
    
    print(f'- nr. of training examples {len(train_iter)}')
    print(f'- nr. of {"test" if arg.final else "validation"} examples {len(test_iter)}')

    if arg.max_length < 0:
        mx = max([input.text[0].size(1) for input in train_iter])
        mx = mx * 2
        print(f'- maximum sequence length: {mx}')
    else:
        mx = arg.max_length

    # create the model
    model = former.CTransformer(emb=arg.embedding_size, heads=arg.num_heads, depth=arg.depth, seq_length=mx, num_tokens=arg.vocab_size, num_classes=NUM_CLS, max_pool=arg.max_pool)
    if torch.cuda.is_available():
        model.cuda()

    model = nn.DataParallel(model, device_ids=[0, 1, 2])
    # opt = torch.optim.Adam(lr=arg.lr, params=model.parameters())
    opt = torch.optim.SGD(lr=arg.lr,params=model.parameters())
    sch = torch.optim.lr_scheduler.LambdaLR(opt, lambda i: min(i / (arg.lr_warmup / arg.batch_size), 1.0))

    # training loop
    seen = 0
    for e in range(arg.num_epochs):
        train_iter = prepare_loader(arg, train_iter, e)

        loss_per_epoch, _, top1_train_ac = train_CrossEntropy(arg, model, device, \
                                                        train_iter, opt, sch,e)

        print('######  testing')
        loss_per_epoch, acc_val_per_epoch_i = testing(arg, model, device, test_iter)

        print(f'\n epoch {e}')
        
if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("-m","--method",
                        dest="method",
                        help="type of sgd",
                        default='unif-SGD', type=str)
    
    parser.add_argument('--c_sgd_warmup', 
                        help="Number of ecpochs with random sampling for p-SGD andd c-SGD",
                        default=0, type=int)

    parser.add_argument('--budget',
                        help='Percentage of buget to use',
                        default=1.0,type=float)

    parser.add_argument("-e", "--num-epochs",
                        dest="num_epochs",
                        help="Number of epochs.",
                        default=30, type=int)

    parser.add_argument("-b", "--batch-size",
                        dest="batch_size",
                        help="The batch size.",
                        default=4, type=int)

    parser.add_argument("-l", "--learn-rate",
                        dest="lr",
                        help="Learning rate",
                        default=0.1, type=float)

    parser.add_argument("-T", "--tb_dir", dest="tb_dir",
                        help="Tensorboard logging directory",
                        default='./runs')

    parser.add_argument("-f", "--final", dest="final",
                        help="Whether to run on the real test set (if not included, the validation set is used).",
                        action="store_true")

    parser.add_argument("--max-pool", dest="max_pool",
                        help="Use max pooling in the final classification layer.",
                        action="store_true")

    parser.add_argument("-E", "--embedding", dest="embedding_size",
                        help="Size of the character embeddings.",
                        default=256, type=int)

    parser.add_argument("-V", "--vocab-size", dest="vocab_size",
                        help="Number of words in the vocabulary.",
                        default=101000, type=int)

    parser.add_argument("-M", "--max", dest="max_length",
                        help="Max sequence length. Longer sequences are clipped (-1 for no limit).",
                        default=512, type=int)

    parser.add_argument("-H", "--heads", dest="num_heads",
                        help="Number of attention heads.",
                        default=8, type=int)

    parser.add_argument("-d", "--depth", dest="depth",
                        help="Depth of the network (nr. of self-attention layers)",
                        default=6, type=int)

    parser.add_argument("-r", "--random-seed",
                        dest="seed",
                        help="RNG seed. Negative for random",
                        default=1, type=int)

    parser.add_argument("--lr-warmup",
                        dest="lr_warmup",
                        help="Learning rate warmup.",
                        default=10_000, type=int)

    parser.add_argument("--gradient-clipping",
                        dest="gradient_clipping",
                        help="Gradient clipping.",
                        default=1.0, type=float)
    
    parser.add_argument("--momentum",
                        dest="momentum",
                        help="momentum for SGD",
                        default=0.9, type=float)

    options = parser.parse_args()


    print('OPTIONS ', options)

    model = go(options)

    time = datetime.now()
    print('saving model ..')
    with open(f'{time}-model.pkl', 'wb') as f:
        pickle.dump(model, f)
    print('model saved')

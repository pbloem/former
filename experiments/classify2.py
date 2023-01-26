import former
from former import util
from former.util import d, here

import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F

# from torchtext import data, datasets, vocab
from torchtext.legacy import data, datasets, vocab

import numpy as np

from argparse import ArgumentParser
from torch.utils.tensorboard import SummaryWriter

import random, tqdm, sys, math, gzip,time

# import torchtext.datasets
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator

from test1 import *
from utils_train import *

# Used for converting between nats and bits
LOG2E = math.log2(math.e)
TEXT = data.Field(lower=True, include_lengths=True, batch_first=True)
LABEL = data.Field(sequential=False)
NUM_CLS = 2

def make_dataset(arg):
    
    tokenizer = get_tokenizer('basic_english')
    def yield_tokens(data_iter):
        for _, text in data_iter:
            yield tokenizer(text)

    train_iter = IMDB(split='train')
    vocab = build_vocab_from_iterator(yield_tokens(train_iter), specials=["<unk>"],max_tokens=arg.vocab_size-1)
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

def go(arg):
    """
    Creates and trains a basic transformer for the IMDB sentiment classification task.
    """
    tbw = SummaryWriter(log_dir=arg.tb_dir) # Tensorboard logging

    # load the IMDB data
    # if arg.final:
    #     train, test = datasets.IMDB.splits(TEXT, LABEL)
    #     TEXT.build_vocab(train, max_size=arg.vocab_size - 2)
    #     LABEL.build_vocab(train)

    #     train_iter, test_iter = data.BucketIterator.splits((train, test), batch_size=arg.batch_size, device=util.d())
    # else:
    #     tdata, _ = datasets.IMDB.splits(TEXT, LABEL)
    #     train, test = tdata.split(split_ratio=0.8)

    #     TEXT.build_vocab(train, max_size=arg.vocab_size - 2) # - 2 to make space for <unk> and <pad>
    #     LABEL.build_vocab(train)

    #     train_iter, test_iter = data.BucketIterator.splits((train, test), batch_size=arg.batch_size, device=util.d())

    trainset, testset = get_dataset(arg)
    train_iter = torch.utils.data.DataLoader(trainset, batch_size=arg.batch_size, \
                                                shuffle=True, \
                                                pin_memory=True, \
                                                drop_last = True)
    test_iter = torch.utils.data.DataLoader(testset, batch_size=arg.batch_size, shuffle=False,pin_memory=True)

    
    print(f'- nr. of training examples {len(train_iter)}')
    print(f'- nr. of {"test" if arg.final else "validation"} examples {len(test_iter)}')

    if arg.max_length < 0:
        mx = max([input.text[0].size(1) for input in train_iter])
        mx = mx * 2
        print(f'- maximum sequence length: {mx}')
    else:
        mx = arg.max_length

    # create the model
    # model = former.CTransformer(emb=arg.embedding_size, heads=arg.num_heads, depth=arg.depth, seq_length=mx, num_tokens=arg.vocab_size, num_classes=NUM_CLS, max_pool=arg.max_pool)
    NUM_CLS = 2
    model = former.CTransformer(emb=256, heads=8, depth=6, seq_length=512, num_tokens=1000000, num_classes=NUM_CLS)
    if torch.cuda.is_available():
        model.cuda()

    opt = torch.optim.Adam(lr=arg.lr, params=model.parameters())
    sch = torch.optim.lr_scheduler.LambdaLR(opt, lambda i: min(i / (arg.lr_warmup / arg.batch_size), 1.0))

    # training loop
    seen = 0
    for e in range(arg.num_epochs):
        print("**-start-**")
        # print(train_iter.dataset.fields.text)
        # time.sleep(3)

        # print(train_iter)
        train_iter = prepare_loader(arg, train_iter, e)

        # # exit()
        loss_per_epoch, _, top1_train_ac = train_CrossEntropy(arg, model, device, \
                                                        train_iter, opt, e)

        # print('######  testing')
        # loss_per_epoch, acc_val_per_epoch_i = testing(arg, model, device, test_iter)

        print(f'\n epoch {e}')
        # model.train(True)

        # for batch in tqdm.tqdm(train_iter):

        #     opt.zero_grad()
        #     # print(len(batch))
        #     print(batch)
        #     # input = batch.text[0]
        #     # label = batch.label - 1
        #     input = batch[0]
        #     label = batch[1]
        #     if input.size(1) > mx:
        #         input = input[:, :mx]

        #     input,label = input.to(device),label.to(device)
        #     # print(input.shape, label.shape)
        #     # time.sleep(10)
        #     out = model(input)
        #     print('out', out)
        #     loss = F.nll_loss(out, label)

        #     loss.backward()

        #     # clip gradients
        #     # - If the total gradient vector has a length > 1, we clip it back down to 1.
        #     if arg.gradient_clipping > 0.0:
        #         nn.utils.clip_grad_norm_(model.parameters(), arg.gradient_clipping)

        #     opt.step()
        #     sch.step()

        #     seen += input.size(0)
        #     tbw.add_scalar('classification/train-loss', float(loss.item()), seen)

        with torch.no_grad():

            model.train(False)
            tot, cor= 0.0, 0.0

            for batch in test_iter:

                input = batch[0]
                label = batch[1]
                input,label = input.to(device), label.to(device)
                if input.size(1) > mx:
                    input = input[:, :mx]
                out = model(input).argmax(dim=1)

                tot += float(input.size(0))
                cor += float((label == out).sum().item())

            acc = cor / tot
            print(f'-- {"test" if arg.final else "validation"} accuracy {acc:.3}')
            print('loss_per_epoch: ',loss_per_epoch)
            # tbw.add_scalar('classification/test-loss', float(loss.item()), e)


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("-m","--method",
                        dest="method",
                        help="type of sgd",
                        default='unif-sgd', type=str)
    
    parser.add_argument('--c_sgd_warmup', 
                        help="Number of ecpochs with random sampling for p-SGD andd c-SGD",
                        default=0, type=int)

    parser.add_argument('--budget',
                        help='Percentage of buget to use',
                        default=1.0,type=float)

    parser.add_argument("-e", "--num-epochs",
                        dest="num_epochs",
                        help="Number of epochs.",
                        default=20, type=int)

    parser.add_argument("-b", "--batch-size",
                        dest="batch_size",
                        help="The batch size.",
                        default=4, type=int)

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

    parser.add_argument("--max-pool", dest="max_pool",
                        help="Use max pooling in the final classification layer.",
                        action="store_true")

    parser.add_argument("-E", "--embedding", dest="embedding_size",
                        help="Size of the character embeddings.",
                        default=128, type=int)

    parser.add_argument("-V", "--vocab-size", dest="vocab_size",
                        help="Number of words in the vocabulary.",
                        default=50_000, type=int)

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

    options = parser.parse_args()

    print('OPTIONS ', options)

    go(options)

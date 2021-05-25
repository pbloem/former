import unittest
import torch, gzip

import transformers as trf

from former import util
from former.util import slice_diag, compute_compression, estimate_compression

from collections.abc import Sequence

from torch.utils.tensorboard import SummaryWriter

import fire

def test_gpt2(batch_size=16, subset=(None, None), name='distilgpt2', tb_dir='./runs/', context=None, skip=0):
    """
    Test the compute_compression function by checking the performance of GPT-2
    :return:
    """

    tbw = SummaryWriter(log_dir=tb_dir)

    tokenizer = trf.GPT2Tokenizer.from_pretrained(name)
    model = trf.GPT2LMHeadModel.from_pretrained(name)

    with gzip.open(util.here() + '/data/enwik8.gz') as file:
        text = str(file.read())

    fr, to = subset
    fr, to = 0 if fr is None else fr, len(text) if to is None else to
    text = text[fr:to]

    numchars = len(text)

    encoded_input = tokenizer(text, return_tensors='pt')['input_ids'][0]

    if torch.cuda.is_available():
        model.cuda()

    context = model.config.n_ctx if context is None else context

    bits, numcharsr = compute_compression(model, data=encoded_input, context=context, batch_size=batch_size, verbose=True, tbw=tbw, tok=tokenizer, skip=skip)

    print('total bits: ', bits)
    print('bits per byte (1): ', bits/numcharsr)
    print('bits per byte (2): ', bits/numchars)

if __name__ == '__main__':

    fire.Fire(test_gpt2)
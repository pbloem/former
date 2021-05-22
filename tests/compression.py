import unittest
import torch, gzip

import transformers as trf

from former import util
from former.util import slice_diag, compute_compression, estimate_compression

from collections.abc import Sequence

import fire

def test_gpt2(batch_size=16, subset=(None, None), name='distilgpt2', samples=-1):
    """
    Test the compute_compression function by checking the performance of GPT-2
    :return:
    """

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

    if samples < 0:
        bits = compute_compression(model, data=encoded_input, context=model.config.n_ctx, batch_size=batch_size, verbose=True)
    else:
        bits = estimate_compression(model, data=encoded_input, context=model.config.n_ctx, batch_size=batch_size,
                                   verbose=True, nsamples=samples)

    print('total bits: ', bits)
    div = numchars if samples < 0 else samples
    print('bits per byte: ', bits/div)

if __name__ == '__main__':

    fire.Fire(test_gpt2)
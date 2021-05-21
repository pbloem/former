import unittest
import torch, gzip

import transformers as trf

from former import util
from former.util import slice_diag, compute_compression

import fire

def test_gpt2(batch_size=16, subset=None, name='distilgpt2'):
    """
    Test the compute_compression function by checking the performance of GPT-2
    :return:
    """

    tokenizer = trf.GPT2Tokenizer.from_pretrained(name)
    model = trf.GPT2LMHeadModel.from_pretrained(name)

    with gzip.open(util.here() + '/data/enwik8.gz') as file:
        text = str(file.read())

    if subset is not None:
        text = text[:subset]

    numchars = len(text)

    encoded_input = tokenizer(text, return_tensors='pt')['input_ids'][0]

    if torch.cuda.is_available():
        model.cuda()

    bits = compute_compression(model, data=encoded_input, context=model.config.n_ctx, batch_size=batch_size, verbose=True)

    print('total bits: ', bits)
    print('bits per byte: ', bits/numchars)

if __name__ == '__main__':

    fire.Fire(test_gpt2)
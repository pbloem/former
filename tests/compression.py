import unittest
import torch, gzip

import transformers as trf

from former import util
from former.util import slice_diag, compute_compression

import fire

def test_gpt2(batch_size=16, subset=None):
    """
    Test the compute_compression function by checking the performance of GPT-2
    :return:
    """
    name = 'distilgpt2'

    tokenizer = trf.GPT2Tokenizer.from_pretrained(name)
    model = trf.GPT2LMHeadModel.from_pretrained(name)

    with gzip.open(util.here() + '/data/enwik8.gz') as file:
        text = str(file.read())

    numchars = len(text)
    print(f'Lodaed enwik8. {type(text)=}, {numchars} characters in total. Chunk: "{text[150_000:150_100]}"')

    if subset is not None:
        text = text[:subset]

    encoded_input = tokenizer(text, return_tensors='pt')['input_ids'][0]

    if torch.cuda.is_available():
        model.cuda()

    bits = compute_compression(model, data=encoded_input, context=model.config.n_ctx, batch_size=32, verbose=True)

    print('total bits: ', bits)
    print('bits per byte: ', bits/numchars)

if __name__ == '__main__':

    fire.Fire(test_gpt2)
import unittest
import torch, gzip

import transformers as trf

from former import util
from former.util import slice_diag, compute_compression

class MyTestCase(unittest.TestCase):

    def test_slice_diagonal(self):

        m = torch.randint(high=20, size=(16, 24, 3, 5, 9))

        print(m[0, 0, 0])
        print(slice_diag(m[0, 0, 0], l=5))

    def test_gpt2(self):
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

        print(f'{type(text)=}, {numchars} characters in total. Chunk: "{text[50_000:50_100]}"')
        encoded_input = tokenizer(text, return_tensors='pt')['input_ids'][0]

        if torch.cuda.is_available():
            model.cuda()

        bits = compute_compression(model, data=encoded_input, context=model.config.n_ctx, batch_size=32, verbose=True)

        print('total bits: ', bits)
        print('bits per byte: ', bits/numchars)

if __name__ == '__main__':
    unittest.main()

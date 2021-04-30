import unittest

import torch
from torch import nn

class GPT2Attention(nn.Module):
    """
    GPT2 implementation of multi-head self-attention (simplified version of the Huggingface implementation).

    :param input:
    :param heads:
    :return:
    """
    def __init__(self, emb, nheads):
        super().__init__()

        self.nheads = nheads
        self.emb = emb

        self.c_attn = nn.Conv1D(3 * emb, emb)
        # -- (out_channels, in_channels): this is just a linear layer.

        self.c_proj = nn.Conv1D(emb, emb)

    def _attn(self, q, k, v):

        w = torch.matmul(q, k) # raw attention weights

        w = w / (float(v.size(-1)) ** 0.5) # scaled attention weights

        w = nn.Softmax(dim=-1)(w) # normalized attention weights

        return torch.matmul(w, v)

    def merge_heads(self, x):

        x = x.permute(0, 2, 1, 3).contiguous()

        new_x_shape = x.size()[:-2] + (x.size(-2) * x.size(-1),)

        return x.view(*new_x_shape)

    def split_heads(self, x, key=False):

        new_x_shape = x.size()[:-1] + (self.n_head, x.size(-1) // self.n_head)

        x = x.view(*new_x_shape)

        if key:
            return x.permute(0, 2, 3, 1)  # (batch, head, head_features, seq_length)
        else:
            return x.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)

    def forward(
        self,
        input_sequence
    ):
        query, key, value = self.c_attn(input_sequence).split(self.split_size, dim=2)

        query = self.split_heads(query)
        key = self.split_heads(key, k=True)
        value = self.split_heads(value)

        a = self._attn(query, key, value)

        a = self.merge_heads(a)
        a = self.c_proj(a)

        return a

class SATestCase(unittest.TestCase):

    def test_sa(self):

        self.assertEqual(True, False)


if __name__ == '__main__':
    unittest.main()

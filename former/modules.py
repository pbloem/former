from .util import mask_, d, slice_diag

import torch
from torch import nn
import torch.nn.functional as F

import random, math, sys

class SelfAttention(nn.Module):
    """
    Canonical implementation of multi-head self attention.
    """

    def __init__(self, emb, heads=8, mask=False, kqnorm=False):
        """

        :param emb:
        :param heads:
        :param mask:
        :param kqnorm:
        """

        super().__init__()

        assert emb % heads == 0, f'Embedding dimension ({emb}) should be divisible by nr. of heads ({heads})'

        self.emb = emb
        self.heads = heads
        self.mask = mask

        s = emb // heads
        # - We will break the embedding into `heads` chunks and feed each to a different attention head

        self.tokeys    = nn.Linear(emb, emb, bias=False)
        self.toqueries = nn.Linear(emb, emb, bias=False)
        self.tovalues  = nn.Linear(emb, emb, bias=False)

        self.unifyheads = nn.Linear(emb, emb)

        self.kqnorm = kqnorm
        if kqnorm:
            self.kln = nn.LayerNorm([s])
            self.qln = nn.LayerNorm([s])

    def forward(self, x):

        b, t, e = x.size()
        h = self.heads
        assert e == self.emb, f'Input embedding dim ({e}) should match layer embedding dim ({self.emb})'

        s = e // h

        keys    = self.tokeys(x)
        queries = self.toqueries(x)
        values  = self.tovalues(x)

        keys    = keys.view(b, t, h, s)
        queries = queries.view(b, t, h, s)
        values  = values.view(b, t, h, s)

        if self.kqnorm:
            keys = self.kln(keys)
            queries = self.qln(queries)

        # -- We first compute the k/q/v's on the whole embedding vectors, and then split into the different heads.
        #    See the following video for an explanation: https://youtu.be/KmAISyVvE1Y

        # Compute scaled dot-product self-attention

        # - fold heads into the batch dimension
        keys = keys.transpose(1, 2).contiguous().view(b * h, t, s)
        queries = queries.transpose(1, 2).contiguous().view(b * h, t, s)
        values = values.transpose(1, 2).contiguous().view(b * h, t, s)

        queries = queries / (e ** (1/4))
        keys    = keys / (e ** (1/4))
        # - Instead of dividing the dot products by sqrt(e), we scale the keys and values.
        #   This should be more memory efficient

        # - get dot product of queries and keys, and scale
        dot = torch.bmm(queries, keys.transpose(1, 2))

        assert dot.size() == (b*h, t, t)

        if self.mask: # mask out the upper half of the dot matrix, excluding the diagonal
            mask_(dot, maskval=float('-inf'), mask_diagonal=False)

        dot = F.softmax(dot, dim=2)
        # - dot now has row-wise self-attention probabilities

        # apply the self attention to the values
        out = torch.bmm(dot, values).view(b, h, t, s)

        # swap h, t back, unify heads
        out = out.transpose(1, 2).contiguous().view(b, t, s * h)

        return self.unifyheads(out)

class SelfAttentionAlt(nn.Module):
    """
    Alternative implementation of self-attention. Should contain fewer parameters, may be faster?
    """

    def __init__(self, emb, heads=8, mask=False):
        """

        :param emb:
        :param heads:
        :param mask:
        """

        super().__init__()

        assert emb % heads == 0, f'Embedding dimension ({emb}) should be divisible by nr. of heads ({heads})'

        self.emb = emb
        self.heads = heads
        self.mask = mask

        s = emb // heads
        # - We will break the embedding into `heads` chunks and feed each to a different attention head

        self.downproj    = nn.Linear(emb, emb, bias=False)
        # -- Single e x e priojection applied before splitting

        # The key, query and value projections of the different heads
        self.tokeys    = nn.Parameter(torch.empty(heads, s, s))
        self.toqueries = nn.Parameter(torch.empty(heads, s, s))
        self.tovalues  = nn.Parameter(torch.empty(heads, s, s))

        self.unifyheads = nn.Linear(emb, emb)

    def forward(self, x):

        b, t, e = x.size()
        h = self.heads
        assert e == self.emb, f'Input embedding dim ({e}) should match layer embedding dim ({self.emb})'

        s = e // h

        x = self.downproj(x).view(b, t, h, s)

        keys    = torch.einsum('bthk, hik -> bthi', x, self.tokeys)
        queries = torch.einsum('bthk, hik -> bthi', x, self.toqueries)
        values  = torch.einsum('bthk, hik -> bthi', x, self.tovalues)

        keys    = keys.view(b, t, h, s)
        queries = queries.view(b, t, h, s)
        values  = values.view(b, t, h, s)

        # Compute scaled dot-product self-attention

        # - fold heads into the batch dimension
        keys = keys.transpose(1, 2).contiguous().view(b * h, t, s)
        queries = queries.transpose(1, 2).contiguous().view(b * h, t, s)
        values = values.transpose(1, 2).contiguous().view(b * h, t, s)

        queries = queries / (e ** (1/4))
        keys    = keys / (e ** (1/4))
        # - Instead of dividing the dot products by sqrt(e), we scale the keys and values.
        #   This should be more memory efficient

        # - get dot product of queries and keys, and scale
        dot = torch.bmm(queries, keys.transpose(1, 2))

        assert dot.size() == (b*h, t, t)

        if self.mask: # mask out the upper half of the dot matrix, excluding the diagonal
            mask_(dot, maskval=float('-inf'), mask_diagonal=False)

        dot = F.softmax(dot, dim=2)
        # - dot now has row-wise self-attention probabilities

        # apply the self attention to the values
        out = torch.bmm(dot, values).view(b, h, t, s)

        # swap h, t back, unify heads
        out = out.transpose(1, 2).contiguous().view(b, t, s * h)

        return self.unifyheads(out)

class SelfAttentionNarrow(nn.Module):
    """
    A self attention with a reduced parameter space (experimental).

    * Uses _the same_ key/query/value transformation on each head, but applied to a different slice of the embedding vector.
    * Dispenses with the linear layer after merging the heads.

    """

    def __init__(self, emb, heads=8, mask=False):
        """
        :param emb:
        :param heads:
        :param mask:
        """

        super().__init__()

        assert emb % heads == 0, f'Embedding dimension ({emb}) should be divisible by nr. of heads ({heads})'

        self.emb = emb
        self.heads = heads
        self.mask = mask

        s = emb // heads
        # - We will break the embedding into `heads` chunks and feed each to a different attention head

        self.tokeys    = nn.Linear(s, s, bias=False)
        self.toqueries = nn.Linear(s, s, bias=False)
        self.tovalues  = nn.Linear(s, s, bias=False)

    def forward(self, x):

        b, t, e = x.size()
        h = self.heads
        assert e == self.emb, f'Input embedding dim ({e}) should match layer embedding dim ({self.emb})'

        s = e // h
        x = x.view(b, t, h, s)

        keys    = self.tokeys(x)
        queries = self.toqueries(x)
        values  = self.tovalues(x)

        # -- We first compute the k/q/v's on the whole embedding vectors, and then split into the different heads.
        #    See the following video for an explanation: https://youtu.be/KmAISyVvE1Y

        # Compute scaled dot-product self-attention

        # - fold heads into the batch dimension
        keys = keys.transpose(1, 2).contiguous().view(b * h, t, s)
        queries = queries.transpose(1, 2).contiguous().view(b * h, t, s)
        values = values.transpose(1, 2).contiguous().view(b * h, t, s)

        queries = queries / (e ** (1/4))
        keys    = keys / (e ** (1/4))
        # - Instead of dividing the dot products by sqrt(e), we scale the keys and values.
        #   This should be more memory efficient

        # - get dot product of queries and keys, and scale
        dot = torch.bmm(queries, keys.transpose(1, 2))

        assert dot.size() == (b*h, t, t)

        if self.mask: # mask out the upper half of the dot matrix, excluding the diagonal
            mask_(dot, maskval=float('-inf'), mask_diagonal=False)

        dot = F.softmax(dot, dim=2)
        # - dot now has row-wise self-attention probabilities

        # apply the self attention to the values
        out = torch.bmm(dot, values).view(b, h, t, s)

        # swap h, t back, unify heads
        out = out.transpose(1, 2).contiguous().view(b, t, h * s)

        return out

class Conv1D(nn.Module):
    """
    1D-convolutional layer as defined by Radford et al. for OpenAI GPT (and also used in GPT-2).

    Basically works like a linear layer but the weights are transposed.

    from:

    Args:
        nf (:obj:`int`): The number of output features.
        nx (:obj:`int`): The number of input features.

    NB: Note the illogical argument order.
    """

    def __init__(self, nf, nx, he=True):
        super().__init__()

        self.nf = nf

        w = torch.empty(nx, nf)
        b = torch.zeros(nf)

        if not he:
            nn.init.normal_(w, std=0.02) # default initialization, seems to be optimized for specific size
        else:
            # Default initialization for nn.Linear
            nn.init.kaiming_uniform_(w, a=math.sqrt(5))
            # -- This assumes a leaky relu activation which isn't what's used downstream, but it's what's used in the other
            #    SA implementations

            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(w)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(b, -bound, bound)

        self.weight = nn.Parameter(w)
        self.bias = nn.Parameter(b)

    def forward(self, x):

        size_out = x.size()[:-1] + (self.nf,) # dimensions of the output tensor

        x = x.view(-1, x.size(-1))
        # -- The weights are applied to the last dimension, all others are collapsed into a single match dimension

        x = torch.addmm(self.bias, x, self.weight)

        x = x.view(*size_out) # restore the original dimensions

        return x

class SelfAttentionGPT2(nn.Module):
    """
    This is the self-attention operation as implemented in the Huggingface port of GPT2. The code has been
    simplified to remove several features not used here but otherwise it should do exactly the same as GPT2 when run with
    normal parameters.

    It is very similar to the default SelfAttention below, with the exception of the way it's initialized and some
    small speed improvements in the custom implementation of the linear layer (the Conv1D defined above).

    We include this primarily for comparison with our own canonical implementation to check for performance differences.
    """
    def __init__(self, emb, heads, mask=False):
        super().__init__()

        self.nheads = heads
        self.emb = emb
        self.mask = mask

        #self.c_attn = Conv1D(3 * emb, emb)
        # -- (out_channels, in_channels):
        #    This is a very slight modification of a linear layer

        self.c_attn = nn.Linear(emb, 3*emb)

        #self.c_proj = Conv1D(emb, emb)
        self.c_proj = nn.Linear(emb, emb)

    def _attn(self, q, k, v):

        dot = torch.matmul(q, k) # raw attention weights

        dot = dot / (float(v.size(-1)) ** 0.5) # scaled attention weights

        if self.mask: # Apply the attention mask
            mask_(dot, maskval=float('-inf'), mask_diagonal=False)
        # -- This is implemented differently in the Huggingface version, but the effect should be the same.

        dot = nn.Softmax(dim=-1)(dot) # normalized attention weights

        return torch.matmul(dot, v) # attention over values

    def merge_heads(self, x):

        x = x.permute(0, 2, 1, 3).contiguous()

        new_x_shape = x.size()[:-2] + (x.size(-2) * x.size(-1),)

        return x.view(*new_x_shape)

    def split_heads(self, x, is_key=False):

        new_x_shape = x.size()[:-1] + (self.nheads, x.size(-1) // self.nheads)

        x = x.view(*new_x_shape)

        if is_key:
            return x.permute(0, 2, 3, 1)  # (batch, head, head_features, seq_length)
        else:
            return x.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)

    def forward(self, input_sequence):

        b, t, e = input_sequence.size()

        query, key, value = self.c_attn(input_sequence).split(e, dim=2)

        query = self.split_heads(query)
        key = self.split_heads(key, is_key=True)
        value = self.split_heads(value)

        a = self._attn(query, key, value)

        a = self.merge_heads(a)
        a = self.c_proj(a)

        return a

class SelfAttentionWide(nn.Module):
    """
    A self-attention with a larger number of parameters than the standard one.

    Uses a full-size embedding vector for each head.
    """

    def __init__(self, emb, heads=8, mask=False):
        """

        :param emb:
        :param heads:
        :param mask:
        """

        super().__init__()

        self.emb = emb
        self.heads = heads
        self.mask = mask

        self.tokeys = nn.Linear(emb, emb * heads, bias=False)
        self.toqueries = nn.Linear(emb, emb * heads, bias=False)
        self.tovalues = nn.Linear(emb, emb * heads, bias=False)

        self.unifyheads = nn.Linear(heads * emb, emb)

    def forward(self, x):

        b, t, e = x.size()
        h = self.heads
        assert e == self.emb, f'Input embedding dim ({e}) should match layer embedding dim ({self.emb})'

        keys    = self.tokeys(x)   .view(b, t, h, e)
        queries = self.toqueries(x).view(b, t, h, e)
        values  = self.tovalues(x) .view(b, t, h, e)

        # compute scaled dot-product self-attention

        # - fold heads into the batch dimension
        keys = keys.transpose(1, 2).contiguous().view(b * h, t, e)
        queries = queries.transpose(1, 2).contiguous().view(b * h, t, e)
        values = values.transpose(1, 2).contiguous().view(b * h, t, e)

        queries = queries / (e ** (1/4))
        keys    = keys / (e ** (1/4))
        # - Instead of dividing the dot products by sqrt(e), we scale the keys and values.
        #   This should be more memory efficient

        # - get dot product of queries and keys, and scale
        dot = torch.bmm(queries, keys.transpose(1, 2))

        assert dot.size() == (b*h, t, t)

        if self.mask: # mask out the upper half of the dot matrix, excluding the diagonal
            mask_(dot, maskval=float('-inf'), mask_diagonal=False)

        dot = F.softmax(dot, dim=2)
        # - dot now has row-wise self-attention probabilities

        # apply the self attention to the values
        out = torch.bmm(dot, values).view(b, h, t, e)

        # swap h, t back, unify heads
        out = out.transpose(1, 2).contiguous().view(b, t, h * e)

        return self.unifyheads(out)


class SelfAttentionRelative(nn.Module):
    """
    Implementation of self-attention with relative position embeddings.

    Inspired by the Transformer-XL relative positions. Not guaranteed to be exactly the same. See
      https://youtu.be/oUhGZMCTHtI
    for an explanation.

    """

    def __init__(self, emb, pos_embedding, heads=8, mask=False, ):
        """

        :param emb:
        :param heads:
        :param mask:
        """

        super().__init__()

        assert emb % heads == 0, f'Embedding dimension ({emb}) should be divisible by nr. of heads ({heads})'

        self.emb = emb
        self.heads = heads
        self.mask = mask

        self.pos = pos_embedding # embedding layer

        e, s, h = emb, emb // heads, heads
        # - We will break the embedding into `heads` chunks and feed each to a different attention head

        self.tokeys    = nn.Linear(emb, emb, bias=False)
        self.tokeys_pos    = nn.Linear(emb, emb, bias=False)
        self.toqueries = nn.Linear(emb, emb, bias=False)
        self.tovalues  = nn.Linear(emb, emb, bias=False)

        self.unifyheads = nn.Linear(emb, emb)

        self.parma, self.parmb = nn.Parameter(torch.randn(1, h, 1, s)), nn.Parameter(torch.randn(1, h, 1, s))

    def forward(self, x):


        b, t, e = x.size()
        h = self.heads
        assert e == self.emb, f'Input embedding dim ({e}) should match layer embedding dim ({self.emb})'

        s = e // h

        keys     = self.tokeys(x)
        queries  = self.toqueries(x)
        values   = self.tovalues(x)

        positions = self.pos(torch.arange(2*t-1, device=d(x)))[None, :].expand(b, 2*t-1, e)
        keys_pos = self.tokeys_pos(positions)

        assert keys_pos.size() == (b, 2*t-1, e)

        keys     = keys.view(b, t, h, s)
        keys_pos = keys_pos.view(b, 2*t-1, h, s)
        queries  = queries.view(b, t, h, s)
        values   = values.view(b, t, h, s)

        # -- We first compute the k/q/v's on the whole embedding vectors, and then split into the different heads.
        #    See the following video for an explanation: https://youtu.be/KmAISyVvE1Y

        # Compute scaled dot-product self-attention

        # - fold heads into the batch dimension
        keys = keys.transpose(1, 2).contiguous().view(b * h, t, s)
        keys_pos = keys_pos.transpose(1, 2).contiguous().view(b * h, 2*t-1, s)
        queries = queries.transpose(1, 2).contiguous().view(b * h, t, s)
        values = values.transpose(1, 2).contiguous().view(b * h, t, s)

        # expand a and b in batch dimension, and fold in heads
        parma = self.parma.expand(b, h, t, s).contiguous().view(b*h, t, s)
        parmb = self.parmb.expand(b, h, t, s).contiguous().view(b*h, t, s)

        # The matrix of raw attention weights (`dot`) is the sum of four different matrix products

        dot_tt = torch.einsum('bis, bjs -> bij',  queries, keys)     # -- basic self attention: token with token
        assert dot_tt.size()== (b*h, t, t), f'{dot_tt.size()}'

        dot_tp = torch.einsum('bis, bjs -> bij', queries, keys_pos) # -- token with position
        dot_tp = slice_diag(dot_tp, l=t)
        assert dot_tp.size() == (b*h, t, t), f'{dot_tp.size()}'

        dot_pt = torch.einsum('bis, bjs -> bij', parma, keys)  # -- position with token
        assert dot_pt.size() == (b*h, t, t), f'{dot_pt.size()}'

        dot_pp =  torch.einsum('bis, bjs -> bij', parmb, keys_pos)  # -- pos with pos
        dot_pp = slice_diag(dot_pp, l=t)
        assert dot_pp.size() == (b*h, t, t), f'{dot_pp.size()}'

        dot = dot_tt + dot_tp + dot_pt + dot_pp

        assert dot.size() == (b*h, t, t)

        if self.mask: # mask out the upper half of the dot matrix, excluding the diagonal
            mask_(dot, maskval=float('-inf'), mask_diagonal=False)

        dot = F.softmax(dot, dim=2)
        # - dot now has row-wise self-attention probabilities

        # apply the self attention to the values
        out = torch.bmm(dot, values).view(b, h, t, s)

        # swap h, t back, unify heads
        out = out.transpose(1, 2).contiguous().view(b, t, s * h)

        return self.unifyheads(out)


class TransformerBlock(nn.Module):
    """
    A straightforward transformer block.
    """

    def __init__(self, emb, heads, mask, seq_length, ff_hidden_mult=4, dropout=0.0, attention_type='default',
                 pos_embedding=None, sa_kwargs={}):
        super().__init__()

        if attention_type == 'default':
            self.attention = SelfAttention(emb, heads=heads, mask=mask, **sa_kwargs)
        elif attention_type == 'alt':
            self.attention = SelfAttentionAlt(emb, heads=heads, mask=mask)
        elif attention_type == 'wide':
            self.attention = SelfAttentionWide(emb, heads=heads, mask=mask)
        elif attention_type == 'gpt2':
            self.attention = SelfAttentionGPT2(emb, heads=heads, mask=mask)
        elif attention_type == 'narrow':
            self.attention = SelfAttentionNarrow(emb, heads=heads, mask=mask)
        elif attention_type == 'relative':
            assert pos_embedding is not None
            self.attention = SelfAttentionRelative(emb, heads=heads, mask=mask, pos_embedding=pos_embedding)
        else:
            raise Exception(f'Self-attention type {type} not recognized.')

        self.mask = mask

        self.norm1 = nn.LayerNorm(emb)
        self.norm2 = nn.LayerNorm(emb)

        self.ff = nn.Sequential(

            nn.Linear(emb, ff_hidden_mult * emb),
            nn.ReLU(),
            nn.Linear(ff_hidden_mult * emb, emb)
        )

        self.do = nn.Dropout(dropout)

    def forward(self, x):

        attended = self.attention(x)

        x = self.norm1(attended + x)

        x = self.do(x)

        fedforward = self.ff(x)

        x = self.norm2(fedforward + x)

        x = self.do(x)

        return x
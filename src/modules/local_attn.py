

import torch
import torch.nn as nn
import torch.nn.functional as F
import math, copy, time
from torch.autograd import Variable
import numpy as np

class Encoder(nn.Module):
    def __init__(self, num_layers, h, d_model, window_size, dropout):
        super(Encoder, self).__init__()
        self.enc_layers = nn.ModuleList([EncoderLayer(h, d_model, window_size, dropout)
                                          for _ in range(num_layers)])
    def forward(self, x, mask):
        for layer in self.enc_layers:
            x = layer(x, mask)
        return x

class EncoderLayer(nn.Module):
    def __init__(self, h, d_model, window_size=16, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.multihead_attn = MultiHeadedAttention(h, d_model, window_size, dropout)
        self.layernorm = LayerNorm(d_model, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        out = self.multihead_attn(x,x,x,mask)
        out = self.dropout(out)
        out = self.layernorm(x + out)
        return out


class LayerNorm(nn.Module):
    "Construct a layernorm module (See citation for details)."
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(features))
        self.beta = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.beta * (x - mean) / (std + self.eps) + self.gamma


def mask_local_mask(size, window_size=16):
    tmp = torch.ones(size, size).long()
    mask = torch.triu(tmp, diagonal=int(window_size/2)) | (1 - torch.triu(tmp, diagonal=-int(window_size/2-1)))
    del tmp
    return (1 - mask).unsqueeze(0)

def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

def attention(query, key, value, window_size=16, mask=None, dropout=None):
    "Compute 'Scaled Dot Product Attention'"
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)

    ids = scores.topk(k=window_size, dim=-1)[1]
    dy_mask = torch.zeros(scores.shape, device=ids.device).scatter(-1, ids, 1).to(scores.device)

    #print("mask: ", mask[0, 0, 0, :])
    #print("dy_mask:", dy_mask[0 ,0 ,0, :])
    if mask is not None:
        comnine_mask = dy_mask.type_as(mask) & mask
        scores = scores.masked_fill(comnine_mask == 0, -1e9)

    #if mask is not None:
    #    scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim = -1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn


class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, window_size=16, dropout=0.1):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.window_size = window_size
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        """

        :param query:
        :param key:
        :param value:
        :param mask:
        :return:
        """
        "Implements Figure 2"
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)
        
        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]   # [batch, head, len, d_k]

        # 2) Apply attention on all the projected vectors in batch.
        x, self.attn = attention(query, key, value, window_size=self.window_size, mask=mask,
                                 dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)


if __name__ == "__main__":
    local_mask = mask_local_mask(19, 16)
    print(local_mask.shape)



import torch
import torch.nn as nn
import torch.nn.functional as F
import math, copy, time
from torch.autograd import Variable
import numpy as np


def subsequent_mask(size):
    "Mask out subsequent positions."
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0

def make_trg_mask(tgt, pad):
    "Create a mask to hide padding and future words."
    tgt_mask = (tgt != pad).unsqueeze(-2)
    tgt_mask = tgt_mask & Variable(
        subsequent_mask(tgt.size(-1)).type_as(tgt_mask.data))
    return tgt_mask

def mask_local_mask(size, window_size=16):
    tmp = torch.ones(size, size).long()
    mask = torch.triu(tmp, diagonal=int(window_size/2)) | (1 - torch.triu(tmp, diagonal=-int(window_size/2-1)))
    del tmp
    return (1 - mask).unsqueeze(0)

def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

def attention(query, key, value, mask=None, dropout=None):
    "Compute 'Scaled Dot Product Attention'"
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)

    ids = scores.topk(k=16, dim=-1)[1]
    dy_mask = torch.zeros(scores.shape, device=ids.device).scatter(-1, ids, 1).to(scores.device)

    #print("mask: ", mask[0, 0, 0, :])
    #print("dy_mask:", dy_mask[0 ,0 ,0, :])
    if mask is not None:
        comnine_mask = dy_mask.type_as(mask) & mask
        #print("comnine_mask: ", comnine_mask[0, 0, 0, :])
        #exit()
        scores = scores.masked_fill(comnine_mask == 0, -1e9)

    #if mask is not None:
    #    scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim = -1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn


class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
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
        x, self.attn = attention(query, key, value, mask=mask,
                                 dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)


if __name__ == "__main__":
    local_mask = mask_local_mask(19, 16)
    print(local_mask.shape)


import torch
import torch.nn as nn
import torch.nn.functional as F
import math, copy, time
from torch.autograd import Variable
import numpy as np
# from .tcn import TemporalConvNet
import random
from collections import defaultdict


def mask_local_mask(size, local_ws=16):
    tmp = torch.ones(size, size).long()
    mask = torch.triu(tmp, diagonal=int(local_ws/2)) | (1 - torch.triu(tmp, diagonal=-int(local_ws/2-1)))
    return (1 - mask).unsqueeze(0)


def TemporalAttention(x, feat_dim=512, window_size=12):
    """
    :param x: [batch, t, 512]
    :return:
    """

    scores = torch.matmul(x, x.transpose(-2, -1)) / math.sqrt(feat_dim)
    ids = scores.topk(k=window_size, dim=-1)[1]

    dynamic_mask = torch.zeros(scores.shape, device=ids.device).scatter(-1, ids, 1).contiguous().long()
    local_mask = mask_local_mask(size=x.size(1), local_ws=window_size).long().to(x.device)

    mask = dynamic_mask & local_mask
    mask = mask.type_as(x).to(x.device)

    scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim=-1)
    soft_avg_x = torch.matmul(p_attn, x)

    return soft_avg_x


class TemporalAttention2(nn.Module):
    def __init__(self, feat_dim=512, window_size=12):
        super(TemporalAttention2, self).__init__()
        self.feat_dim = feat_dim
        self.window_size = window_size
        self.relu = nn.ReLU()
        self.enc_conv1 = nn.Conv2d(in_channels=512, out_channels=512,
                                   kernel_size=(5, 5), padding=(2, 0), stride=1)
        self.enc_bn1 = nn.BatchNorm2d(512, affine=True)
        self.enc_pool1 = nn.MaxPool2d(kernel_size=(2,2), stride=(2,2))

        self.enc_conv2 = nn.Conv2d(in_channels=512, out_channels=512,
                                   kernel_size=(5, 3), padding=(2, 0), stride=1)
        self.enc_bn2 = nn.BatchNorm2d(512, affine=True)
        self.enc_pool2 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.enc = nn.Sequential(self.enc_conv1, self.enc_bn1, self.relu, self.enc_pool1,
                                 self.enc_conv2, self.enc_bn2, self.relu, self.enc_pool2)

    def forward(self, x):
        """
        :param x:  [batch, t, 512]
        :return:  [batch, t, window_size, 512]
        """
        scores = torch.matmul(x, x.transpose(-2, -1)) / math.sqrt(self.feat_dim)  # [batch, t, t]
        local_mask = mask_local_mask(size=x.size(1), local_ws=2 * self.window_size).to(x.device) # [batch, t, t]

        scores = scores.masked_fill(local_mask == 0, -1e9)
        ids = scores.topk(k=self.window_size, dim=-1)[1].sort(-1)[0]  # [batch, t, k]
        # print(ids)
        # print(ids.shape)

        out = []
        for i in range(x.size(0)):  # batch
            batch_t = []
            for j in range(x.size(1)):  # t
                t = x[i].index_select(0, ids[i, j, :]).unsqueeze(0)  # [1, k, 512]
                batch_t.append(t)
            batch_t = torch.cat(batch_t, dim=0).unsqueeze(0)   # [1, t, k, 512]
            out.append(batch_t)
        out = torch.cat(out, dim=0) # [bs, t, k, 512]
        out = out.permute(0, 3, 1, 2).contiguous()
        out = self.enc(out).squeeze()

        return out


class TemporalAttention3(nn.Module):
    def __init__(self, feat_dim=512, window_size=12, dropout=0.2):
        super(TemporalAttention3, self).__init__()
        self.feat_dim = feat_dim
        self.window_size = window_size
        self.relu = nn.ReLU()
        # self.k_tcn = TemporalConvNet(feat_dim, [feat_dim, feat_dim], kernel_size=3, dropout=dropout)
        self.rnn = nn.GRU(feat_dim, feat_dim)
        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        """

        :param x:  [batch, t, 512]
        :return:  [batch, t, window_size, 512]
        """
        scores = torch.matmul(x, x.transpose(-2, -1)) / math.sqrt(self.feat_dim)  # [batch, t, t]
        local_mask = mask_local_mask(size=x.size(1), local_ws=2 * self.window_size).to(x.device) # [batch, t, t]

        scores = scores.masked_fill(local_mask == 0, -1e9)
        ids = scores.topk(k=self.window_size, dim=-1)[1].sort(-1)[0].detach_()  # require_grad=False, [batch, t, k]

        # print(ids)

        feature = []
        for i in range(x.size(0)):  # batch
            batch_t = []
            for j in range(x.size(1)):  # t
                t = x[i].index_select(0, ids[i, j, :]).unsqueeze(0)  # [1, k, 512]
                batch_t.append(t)
            batch_t = torch.cat(batch_t, dim=0).unsqueeze(0)   # [1, t, k, 512]
            feature.append(batch_t)
        feature = torch.cat(feature, dim=0) # [bs, t, k, 512]
        feature = feature.reshape(-1, self.window_size, self.feat_dim)  # [bs*t, k, 512]

        # tcn
        # feature = feature.permute(0, 2, 1).contiguous()  # [bs*t, 512, k]
        # out = self.k_tcn(feature) # [bs*t, 512, k]

        # rnn
        feature = feature.permute(1,0,2).contiguous()  # [k, bs*t, 512]
        _, out = self.rnn(feature)  # [1, bs*t, 512]
        # print(out.shape)

        out = out.squeeze().reshape(x.size(0), x.size(1), -1)  # [batch, t, 512]
        del feature

        # residual connection
        out += x
        return out


class TemporalAttention4(nn.Module):
    def __init__(self, feat_dim=512, window_size=12, dropout=0.2):
        super(TemporalAttention4, self).__init__()
        self.feat_dim = feat_dim
        self.window_size = window_size
        self.relu = nn.ReLU()
        # self.k_tcn = TemporalConvNet(feat_dim, [feat_dim, feat_dim], kernel_size=3, dropout=dropout)
        self.rnn = nn.GRU(feat_dim, feat_dim)
        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        """

        :param x:  [batch, t, 512]
        :return:  [batch, t, window_size, 512]
        """

        scores = torch.matmul(x, x.transpose(-2, -1)) / math.sqrt(self.feat_dim)  # [batch, t, t]
        local_mask = mask_local_mask(size=x.size(1), local_ws=2 * self.window_size).to(x.device) # [batch, t, t]

        scores = scores.masked_fill(local_mask == 0, -1e9)
        ids = scores.topk(k=self.window_size, dim=-1)[1].sort(-1)[0].detach_()  # require_grad=False, [batch, t, k]

        # temporal_ids = random.sample(range(x.size(1)), int(x.size(1) / 4))
        # temporal_ids.sort()

        if self.training:
            temporal_ids = random.sample(range(x.size(1)), int(x.size(1) / 4))
            temporal_ids.sort()
            # print("train: ", temporal_ids)
        else:
            temporal_ids = [int(id) for id in np.linspace(0, x.size(1)-1, int(x.size(1)/4))]
            temporal_ids.sort()
            # print("test: ", temporal_ids)


        feature = []
        for i in range(x.size(0)):  # batch
            batch_t = []
            for j in temporal_ids:  # t
                t = x[i].index_select(0, ids[i, j, :]).unsqueeze(0)  # [1, k, 512]
                batch_t.append(t)
            batch_t = torch.cat(batch_t, dim=0).unsqueeze(0)   # [1, t/4, k, 512]
            feature.append(batch_t)
        feature = torch.cat(feature, dim=0) # [bs, t/4, k, 512]
        feature = feature.reshape(-1, self.window_size, self.feat_dim)  # [bs*t/4, k, 512]

        # rnn
        feature = feature.permute(1,0,2).contiguous()  # [k, bs*t/4, 512]
        _, out = self.rnn(feature)  # [1, bs*t/4, 512]


        out = out.squeeze().reshape(x.size(0), len(temporal_ids), -1)  # [batch, t/4, 512]
        del feature

        # residual connection
        # temporal_ids_2 = [int(id) for id in np.linspace(0, x.size(1)-1, len(temporal_ids))]
        # temporal_ids_2.sort()
        # sampled_x = x[:, temporal_ids_2, :]
        #
        # out += sampled_x
        return out


class TemporalAttention5(nn.Module):
    def __init__(self, feat_dim=512, window_size=12, dropout=0.2):
        super(TemporalAttention5, self).__init__()
        self.feat_dim = feat_dim
        self.window_size = window_size
        self.relu = nn.ReLU()
        # self.k_tcn = TemporalConvNet(feat_dim, [feat_dim, feat_dim], kernel_size=3, dropout=dropout)
        self.rnn = nn.GRU(feat_dim, feat_dim)
        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        """

        :param x:  [batch, t, 512]
        :return:  [batch, t, window_size, 512]
        """

        scores = torch.matmul(x, x.transpose(-2, -1)) / math.sqrt(self.feat_dim)  # [batch, t, t]
        local_mask = mask_local_mask(size=x.size(1), local_ws=2 * self.window_size).to(x.device) # [batch, t, t]

        scores = scores.masked_fill(local_mask == 0, -1e9)
        ids = scores.topk(k=self.window_size, dim=-1)[1].sort(-1)[0].detach_()  # require_grad=False, [batch, t, k]

        features = []
        feature_len = []
        # print("origin size: ", x.size(1))
        for i in range(x.size(0)):  # batch
            batch_ids = ids[i, :, :]  # [t, k]
            batch_t = []
            id_map = defaultdict(int)

            for j in range(x.size(1)):  #  t
                key = (torch.mean(batch_ids[j].float()).cpu().numpy().tolist(),
                       torch.std(batch_ids[j].float()).cpu().numpy().tolist())
                if key not in id_map:  # batch_ids[j]: [k]
                    id_map[key] = j
                    t = x[i].index_select(0, ids[i, j, :]).unsqueeze(0) # [1, k, 512]
                    batch_t.append(t)      # <= t
            feature_len.append(len(batch_t))
            batch_t = torch.cat(batch_t, dim=0).unsqueeze(0)  # [1, <=t, k, 512]
            features.append(batch_t)              # the length of batch_t is different.
        max_len = max(feature_len)
        new_feature = torch.zeros(x.size(0), max_len, self.window_size, self.feat_dim).to(x.device)  # [batch, max_len, k, 512]
        for i in range(x.size(0)):
            new_feature[i, :feature_len[i], :, :] = features[i]
        # print("after fusion, size: ", max_len)


        # rnn
        new_feature = new_feature.reshape(-1, self.window_size, self.feat_dim)  # [bs*t/4, k, 512]
        new_feature = new_feature.permute(1,0,2).contiguous()  # [k, bs*max_len, 512]
        _, out = self.rnn(new_feature)  # [1, bs*t/4, 512]


        out = out.squeeze().reshape(x.size(0), max_len, -1)  # [batch, max_len, 512]
        del features

        # residual connection
        # temporal_ids_2 = [int(id) for id in np.linspace(0, x.size(1)-1, max_len)]
        # temporal_ids_2.sort()
        # sampled_x = x[:, temporal_ids_2, :]
        #
        # out += sampled_x
        return out, torch.LongTensor(feature_len).to(x.device)


if __name__ == "__main__":
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    x = torch.randn(2, 152, 512).cuda()
    model = TemporalAttention5(512, 12).cuda()
    out, leng = model(x)
    print(out.shape, leng)
    # a = x.index_select(0, torch.LongTensor([1,2]))
    # print(a.shape)
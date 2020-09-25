"""
Implementation of "Fully Convolutional Networks for Continuous Sign Language Recognition"
"""

import torch
import torch.nn as nn
# from .local_attn import Encoder, mask_local_mask, LayerNorm
from src.modules.tcna import TemporalAttention3

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


class BasicBlock(nn.Module):
    def __init__(self, inplanes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.norm_layer = nn.BatchNorm2d
        self.conv1 = conv3x3(inplanes, planes, stride=1)
        self.bn1 = self.norm_layer(planes, affine=True)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)  # downsample
        self.conv2 = conv3x3(planes, planes, stride=1)
        self.bn2 = self.norm_layer(planes, affine=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        return x


class MainStream(nn.Module):
    def __init__(self, vocab_size):
        super(MainStream, self).__init__()

        # cnn
        # first layer: channel 3 -> 32
        self.conv = conv3x3(3, 32)
        self.bn = nn.BatchNorm2d(32, affine=True)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(2, 2)

        # 4 basic blocks
        channels = [32, 64, 128, 256, 512]
        layers = []
        for num_layer in range(len(channels) - 1):
            layers.append(BasicBlock(channels[num_layer], channels[num_layer + 1]))
        self.layers = nn.Sequential(*layers)

        self.avgpool = nn.AdaptiveAvgPool2d(output_size=1)
        # self-attention

        # encoder G1, two F5-S1-P2-M2
        self.tcna = TemporalAttention3(feat_dim=512, window_size=16, dropout=0.2) # [bs ,512, t/4]
        self.enc1_conv1 = nn.Conv1d(in_channels=512,
                                    out_channels=512,
                                    kernel_size=5,
                                    stride=1,
                                    padding=2)
        self.enc1_bn1 = nn.BatchNorm1d(512, affine=True)
        self.enc1_pool1 = nn.MaxPool1d(kernel_size=2, stride=2)

        self.enc1_conv2 = nn.Conv1d(in_channels=512,
                                    out_channels=512,
                                    kernel_size=5,
                                    stride=1,
                                    padding=2)
        self.enc1_bn2 = nn.BatchNorm1d(512, affine=True)
        self.enc1_pool2 = nn.MaxPool1d(kernel_size=2, stride=2)
        # self.enc1 = nn.Sequential(self.enc1_conv1, self.enc1_bn1, self.relu, self.enc1_pool1,
        #                          self.enc1_conv2, self.enc1_bn2, self.relu, self.enc1_pool2)

        # encoder G2, one F3-S1-P1
        self.enc2_conv = nn.Conv1d(in_channels=512,
                                   out_channels=1024,
                                   kernel_size=3,
                                   stride=1,
                                   padding=1)
        self.enc2_bn = nn.BatchNorm1d(1024, affine=True)
        # self.enc2 = nn.Sequential(self.enc2_conv, self.enc2_bn, self.relu)

        self.fc = nn.Linear(1024, vocab_size)

        self.init()

    def init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.Linear)):
                # Slightly different from the TF version which uses truncated_normal for initialization
                # cf https://github.com/pytorch/pytorch/pull/5617
                m.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                m.bias.data.zero_()


    def forward(self, video, len_video=None):
        """
        x: [batch, num_f, 3, h, w]
        """
        # print("input: ", video.size())
        bs, num_f, c, h, w = video.size()

        x = video.reshape(-1, c, h, w)

        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.layers(x)
        x = self.avgpool(x).squeeze_()  # [bs*t, 512]

        x = x.reshape(bs, -1, 512)     # [bs, t ,512]
        x = self.tcna(x)   # [bs, t ,512]
        x = x.permute(0, 2, 1)  # [bs, 512, t]

        # enc1
        x = self.enc1_conv1(x)  # [bs, 512, t/2]
        x = self.enc1_bn1(x)
        x = self.relu(x)
        x = self.enc1_pool1(x)  # [bs, 512, t/2]

        x = self.enc1_conv2(x)  # [bs, 512, t/2]
        x = self.enc1_bn2(x)
        x = self.relu(x)
        x = self.enc1_pool2(x)  # [bs, 512, t/4]
        # enc2
        x = self.enc2_conv(x)
        x = self.enc2_bn(x)
        out = self.relu(x)

        out = out.permute(0, 2, 1)
        logits = self.fc(out)  # [batch, t/4, vocab_size]
        len_video = torch.Tensor(bs * [logits.size(1)]).long().to(logits.device)
        return logits, len_video


if __name__ == "__main__":
    x = torch.randn(1, 300, 3, 112, 112).cuda()
    model = MainStream(1233).cuda()
    for i in range(500):
        out = model(x)
        print(out[0].shape, out[1])

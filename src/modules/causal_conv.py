import torch
import torch.nn as nn


class CasualConv1D(nn.Conv1d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1,
                 padding=0, groups=1, bias=True):
        """Inherited from Conv1D. Costum padding on the left of inputs, and keep the dimension same.
        """
        super(CasualConv1D, self).__init__(in_channels, out_channels, kernel_size, stride=stride,
                                           padding=padding, dilation=dilation, groups=groups, bias=bias)

        custom_pad = (self.kernel_size[0] - 1) * self.dilation[0]
        self.pad = nn.ConstantPad1d((custom_pad, 0), 0)

    def forward(self, x):
        """
        :param x: [batch, channel, t]
        :return:  [batch, channel, t]
        """
        x = self.pad(x)  # [batch, channle, t+pad]
        return super(CasualConv1D, self).forward(x)


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation=1, padding=0, dropout=0.2):
        """temporal convolution block.
        Note: the layer normalization layer is TODO.

        :param padding: assert to be zero.
        """
        super(TemporalBlock, self).__init__()
        self.conv1 = CasualConv1D(n_inputs, n_outputs, kernel_size, stride, dilation)
        # TODO? the mean and variance is computed only on the last dimension.
        self.norm_layer1 = nn.BatchNorm1d(n_outputs, affine=True)
        self.dropout1 = nn.Dropout(dropout)
        self.relu1 = nn.ReLU()

        self.conv2 = CasualConv1D(n_outputs, n_outputs, kernel_size, stride, dilation)
        # TODO?
        self.norm_layer2 = nn.BatchNorm1d(n_outputs, affine=True)
        self.dropout2 = nn.Dropout(dropout)
        self.relu2 = nn.ReLU()

        self.net = nn.Sequential(self.conv1, self.norm_layer1, self.relu1, self.dropout1,
                                 self.conv2, self.norm_layer2, self.relu2, self.dropout2)

        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.LayerNorm):
                m.bias.data.zero_()
                m.weight.data.fill_(1.0)

    def forward(self, x):
        """
        :param x: [bs, channel, t]
        :return: [bs, channel ,t]
        """
        res = x
        x = self.conv1(x)   # [bs, channel, t]
        x = self.norm_layer1(x)  # [bs, channel, t]
        x = self.dropout1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.norm_layer2(x)
        x = self.dropout2(x)
        x = self.relu2(x)
        if self.downsample is not None:
            res = self.downsample(res)
        return self.relu(x + res)


if __name__ == "__main__":
    x = torch.zeros(2, 512, 20)
    model = TemporalBlock(512, 512, kernel_size=3, stride=1, dilation=2)
    print(model(x).shape)

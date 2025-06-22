
import torch
import torch.nn as nn
import torch.nn.functional as F

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class ConvBlock(nn.Module):
    def __init__(self, in_c, out_c, kernel_size, stride, padding):
        super().__init__()
        self.conv = nn.Conv2d(in_c, out_c, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_c)
        self.prelu = nn.PReLU()
    def forward(self, x):
        return self.prelu(self.bn(self.conv(x)))

class MiniFASNetV2(nn.Module):
    def __init__(self, num_classes=3, img_channel=3):
        super().__init__()
        self.conv1 = ConvBlock(img_channel, 32, 3, 2, 1)
        self.conv2_dw = ConvBlock(32, 32, 3, 1, 1)

        self.conv_23 = nn.Sequential(
            ConvBlock(32, 103, 1, 1, 0),
            ConvBlock(103, 103, 3, 2, 1),
            ConvBlock(103, 64, 1, 1, 0),
        )

        self.conv_3 = nn.Sequential(
            self._make_block(64, 13),
            self._make_block(64, 26),
            self._make_block(64, 13),
            self._make_block(64, 52),
        )

        self.conv_34 = nn.Sequential(
            ConvBlock(64, 231, 1, 1, 0),
            ConvBlock(231, 231, 3, 2, 1),
            ConvBlock(231, 128, 1, 1, 0),
        )

        self.conv_4 = nn.Sequential(
            self._make_block(128, 154),
            self._make_block(128, 52),
            self._make_block(128, 26),
            self._make_block(128, 52),
            self._make_block(128, 26),
            self._make_block(128, 26),
        )

        self.conv_45 = nn.Sequential(
            ConvBlock(128, 308, 1, 1, 0),
            ConvBlock(308, 308, 3, 2, 1),
            ConvBlock(308, 128, 1, 1, 0),
        )

        self.conv_5 = nn.Sequential(
            self._make_block(128, 26),
            self._make_block(128, 26),
        )

        self.conv_6_sep = ConvBlock(128, 512, 1, 1, 0)
        self.conv_6_dw = nn.Conv2d(512, 512, 7, groups=512, bias=False)

        self.flatten = Flatten()
        self.linear = nn.Linear(512, 128, bias=False)
        self.bn = nn.BatchNorm1d(128)
        self.dropout = nn.Dropout(p=0.2)
        self.prob = nn.Linear(128, num_classes)

    def _make_block(self, in_c, out_c):
        return nn.Sequential(
            ConvBlock(in_c, out_c, 1, 1, 0),
            ConvBlock(out_c, out_c, 3, 1, 1),
            ConvBlock(out_c, in_c, 1, 1, 0),
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2_dw(x)
        x = self.conv_23(x)
        x = self.conv_3(x)
        x = self.conv_34(x)
        x = self.conv_4(x)
        x = self.conv_45(x)
        x = self.conv_5(x)
        x = self.conv_6_sep(x)
        x = self.conv_6_dw(x)
        x = self.flatten(x)
        x = self.linear(x)
        x = self.bn(x)
        x = self.dropout(x)
        x = self.prob(x)
        return x

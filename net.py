import torch
from torch import nn
from torch.nn import functional as F, Conv2d
from PHFP import *
from AA_DSMamba2 import *
from CASE import *


class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction_ratio=4):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        hidden_channels = max(in_channels // reduction_ratio, 4)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, hidden_channels, bias=False),
            nn.ReLU(),
            nn.Linear(hidden_channels, 64, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.shape
        y = self.avg_pool(x).view(b, c)
        weights = self.fc(y).view(b, 64, 1, 1)
        selected = x[:, :64, :, :] * weights
        return selected


class Conv_Block(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Conv_Block, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 3, 1, 1, padding_mode="reflect", bias=False),
            nn.BatchNorm2d(out_channel),
            nn.Dropout2d(0.3),
            nn.LeakyReLU(),
            nn.Conv2d(out_channel, out_channel, 3, 1, 1, padding_mode="reflect", bias=False),
            nn.BatchNorm2d(out_channel),
            nn.Dropout2d(0.3),
            nn.LeakyReLU(),
        )

    def forward(self, x):
        return self.layer(x)


class Downsample(nn.Module):
    def __init__(self, channel):
        super(Downsample, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(channel, channel, 3, 2, 1, padding_mode="reflect", bias=False),
            nn.BatchNorm2d(channel),
            nn.LeakyReLU(),
        )

    def forward(self, x):
        return self.layer(x)


class Upsample(nn.Module):
    def __init__(self, channel):
        super(Upsample, self).__init__()
        self.layer = nn.Conv2d(channel, channel // 2, 1, 1)

    def forward(self, x, feature_map):
        up = F.interpolate(x, scale_factor=2, mode="nearest")
        out = self.layer(up)
        return torch.cat((out, feature_map), dim=1)


class SFD_Mamba2Net(nn.Module):
    """Clean segmentation backbone.

    This version removes all dynamic mapping / TBDM / shallow style adapter paths.
    The network keeps the original segmentation pathway and exposes O3/O4 features
    for O3/O4 supervised source-target feature alignment.
    """

    def __init__(self):
        super(SFD_Mamba2Net, self).__init__()
        self.c1 = Conv_Block(1, 64)
        self.CASE = CASE(scales=[1, 2, 3], beta=0.5, c=15)
        self.channel_attn = ChannelAttention(in_channels=64 + 1)

        self.d1 = Downsample(64)
        self.c2 = Conv_Block(64, 128)
        self.d2 = Downsample(128)
        self.c3 = Conv_Block(128, 256)
        self.d3 = Downsample(256)
        self.c4 = Conv_Block(256, 512)
        self.d4 = Downsample(512)
        self.c5 = Conv_Block(512, 1024)
        self.mamba2 = AA_DSMamba2(1024, 1024, 32)

        self.u1 = Upsample(1024)
        self.PHFP1 = PHFP(1024, 1024)
        self.c6 = Conv_Block(1024, 512)

        self.u2 = Upsample(512)
        self.PHFP2 = PHFP(512, 512)
        self.c7 = Conv_Block(512, 256)

        self.u3 = Upsample(256)
        self.PHFP3 = PHFP(256, 256)
        self.c8 = Conv_Block(256, 128)

        self.u4 = Upsample(128)
        self.PHFP4 = PHFP(128, 128)
        self.c9 = Conv_Block(128, 64)

        self.out = Conv2d(64, 1, 3, 1, 1)
        self.TH = nn.Sigmoid()

    def forward(self, x, return_feature=False):
        R1 = self.c1(x)
        case_features = self.CASE(x)
        R1 = torch.cat([R1, case_features], dim=1)
        R1 = self.channel_attn(R1)

        R2 = self.c2(self.d1(R1))
        R3 = self.c3(self.d2(R2))
        R4 = self.c4(self.d3(R3))
        R5 = self.c5(self.d4(R4))
        R5 = self.mamba2(R5)

        O1 = self.u1(R5, R4)
        O1 = self.PHFP1(O1)
        O1 = self.c6(O1)

        O2 = self.u2(O1, R3)
        O2 = self.PHFP2(O2)
        O2 = self.c7(O2)

        O3 = self.u3(O2, R2)
        O3 = self.PHFP3(O3)
        O3 = self.c8(O3)

        O4 = self.u4(O3, R1)
        O4 = self.PHFP4(O4)
        O4 = self.c9(O4)

        output = self.out(O4)

        if return_feature:
            weak_global = F.adaptive_avg_pool2d(O4, (1, 1)).flatten(1)
            strong_global = F.adaptive_avg_pool2d(R5, (1, 1)).flatten(1)
            return output, {
                "weak_global": weak_global,
                "strong_global": strong_global,
                "weak_local": O4,
                "R2": R2,
                "O1": O1,
                "O2": O2,
                "O3": O3,
                "O4": O4,
                "R5": R5,
            }
        return output


if __name__ == "__main__":
    x = torch.randn(2, 1, 512, 512)
    net = SFD_Mamba2Net()
    y, feat = net(x, return_feature=True)
    print(y.shape)
    print(feat.keys())

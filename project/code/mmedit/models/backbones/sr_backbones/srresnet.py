import torch
import torch.nn as nn
from mmedit.models.registry import BACKBONES
from mmedit.utils import get_root_logger
import torch.nn.functional as F
from torch.nn import init
from collections import OrderedDict


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        out += residual
        out = self.relu(out)
        return out


class ResidualBlockAttention(nn.Module):
    def __init__(self, channels):
        super(ResidualBlockAttention, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
        self.attention = CoordAtt(channels, channels)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        out += residual
        out = self.attention(out) + out
        out = self.relu(out)
        return out


@BACKBONES.register_module()
class SRResNet(nn.Module):
    def __init__(self, in_channel=3, out_channel=3, attention_block=False, mutil_kernel=False, num_channels=3,
                 num_residual_blocks=8, scaling_factor=4):
        super(SRResNet, self).__init__()
        self.mutil_kernel = mutil_kernel
        if attention_block:
            num_residual_blocks = num_residual_blocks * 2

        if mutil_kernel:
            self.first = MutilKernel(in_channel, 64, act=True)
        else:
            self.first = nn.Sequential(
                nn.Conv2d(in_channel, 64, kernel_size=9, stride=1, padding=4),
                nn.ReLU(inplace=True),
            )

        # 添加残差块
        if attention_block:
            self.residual_blocks = nn.Sequential(
                *[ResidualBlockAttention(64) for _ in range(num_residual_blocks)]
            )
        else:
            self.residual_blocks = nn.Sequential(
                *[ResidualBlock(64) for _ in range(num_residual_blocks)]
            )

        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)

        channel_dict = {
            '2': 64,
            '3': 144,
            '4': 256,
            '8': 1024,
        }
        # 上采样模块
        self.upsample = nn.Sequential(
            nn.Conv2d(64, channel_dict[str(scaling_factor)], kernel_size=3, stride=1, padding=1),
            nn.PixelShuffle(upscale_factor=scaling_factor),
            nn.ReLU(inplace=True)
        )
        if self.mutil_kernel:
            self.conv3 = MutilKernel(16, out_channel, act=True)
        else:
            self.conv3 = nn.Conv2d(16, out_channel, kernel_size=9, stride=1, padding=4)

    def forward(self, x):
        out = self.first(x)
        residual = out
        out = self.residual_blocks(out)
        out = self.conv2(out)
        out += residual
        out = self.upsample(out)
        out = self.conv3(out)
        return out

    def init_weights(self, pretrained=None, strict=True):
        pass


class CALayer(nn.Module):
    def __init__(self, channel, reduction=2):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y


class MutilKernel(nn.Module):
    def __init__(self, in_channel, channel=512, act=False, kernels=[7, 9, 11], reduction=1, group=1, L=32):
        super().__init__()
        self.first = nn.Conv2d(in_channel, channel, kernel_size=3, padding=1, stride=1)
        self.d = reduction
        self.convs = nn.ModuleList([])
        for k in kernels:
            self.convs.append(
                nn.Sequential(OrderedDict([
                    ('conv', nn.Conv2d(channel, channel, kernel_size=k, padding=k // 2, groups=group)),
                ]))
            )
        self.fc = nn.Linear(channel, self.d)
        self.fcs = nn.ModuleList([])
        for i in range(len(kernels)):
            self.fcs.append(nn.Linear(self.d, channel))
        self.softmax = nn.Softmax(dim=0)
        self.act = act
        if self.act:
            self.relu = nn.ReLU()

    def forward(self, x):
        x = self.first(x)
        bs, c, _, _ = x.size()
        conv_outs = []
        for conv in self.convs:
            conv_outs.append(conv(x))
        feats = torch.stack(conv_outs, 0)  # k,bs,channel,h,w
        U = sum(conv_outs)  # bs,c,h,w
        S = U.mean(-1).mean(-1)  # bs,c
        Z = self.fc(S)  # bs,d

        weights = []
        for fc in self.fcs:
            weight = fc(Z)
            weights.append(weight.view(bs, c, 1, 1))  # bs,channel
        attention_weughts = torch.stack(weights, 0)  # k,bs,channel,1,1
        attention_weughts = self.softmax(attention_weughts)  # k,bs,channel,1,1

        V = (attention_weughts * feats).sum(0)
        if self.act:
            V = self.relu(V)
        return V


class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6


class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)


# class SpatialAndContextAttention(nn.Module):
#     def __init__(self, inp, oup, reduction=2):
#         super(SpatialAndContextAttention, self).__init__()
#         self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
#         self.pool_w = nn.AdaptiveAvgPool2d((1, None))
#
#         mip = inp // reduction
#
#         self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
#         self.bn1 = nn.InstanceNorm2d(mip)
#         self.act = h_swish()
#
#         self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
#         self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
#
#     def forward(self, x):
#         n, c, h, w = x.size()
#         x_h = self.pool_h(x)
#         x_w = self.pool_w(x).permute(0, 1, 3, 2)
#
#         y = torch.cat([x_h, x_w], dim=2)
#         y = self.conv1(y)
#         y = self.bn1(y)
#         y = self.act(y)
#
#         x_h, x_w = torch.split(y, [h, w], dim=2)
#         x_w = x_w.permute(0, 1, 3, 2)
#
#         a_h = self.conv_h(x_h).sigmoid()
#         a_w = self.conv_w(x_w).sigmoid()
#         return a_w * a_h


class SpatialAndContextAttention(nn.Module):
    def __init__(self, in_channel, reduction=16):
        super(SpatialAndContextAttention, self).__init__()
        assert in_channel >= reduction and in_channel % reduction == 0
        self.reduction = reduction
        self.cardinality = 4
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Sequential(
            nn.Linear(in_channel, in_channel // self.reduction, bias=False),
            nn.ReLU(inplace=True)
        )
        self.fc2 = nn.Sequential(
            nn.Linear(in_channel, in_channel // self.reduction, bias=False),
            nn.ReLU(inplace=True)
        )
        self.fc3 = nn.Sequential(
            nn.Linear(in_channel, in_channel // self.reduction, bias=False),
            nn.ReLU(inplace=True)
        )
        self.fc4 = nn.Sequential(
            nn.Linear(in_channel, in_channel // self.reduction, bias=False),
            nn.ReLU(inplace=True)
        )

        self.fc = nn.Sequential(
            nn.Linear(in_channel // self.reduction * self.cardinality, in_channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y1 = self.fc1(y)
        y2 = self.fc2(y)
        y3 = self.fc3(y)
        y4 = self.fc4(y)
        y_concate = torch.cat([y1, y2, y3, y4], dim=1)
        y_ex_dim = self.fc(y_concate).view(b, c, 1, 1)

        return x * y_ex_dim.expand_as(x)



class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6


class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)


class CoordAtt(nn.Module):
    def __init__(self, inp, oup, reduction=2):
        super(CoordAtt, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mip = max(8, inp // reduction)

        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.act = h_swish()

        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        n, c, h, w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)

        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.act(y)

        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()
        return a_w * a_h
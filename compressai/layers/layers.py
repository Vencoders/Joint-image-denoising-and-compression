# Copyright (c) 2021-2022, InterDigital Communications, Inc
# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted (subject to the limitations in the disclaimer
# below) provided that the following conditions are met:

# * Redistributions of source code must retain the above copyright notice,
#   this list of conditions and the following disclaimer.
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
# * Neither the name of InterDigital Communications, Inc nor the names of its
#   contributors may be used to endorse or promote products derived from this
#   software without specific prior written permission.

# NO EXPRESS OR IMPLIED LICENSES TO ANY PARTY'S PATENT RIGHTS ARE GRANTED BY
# THIS LICENSE. THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND
# CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT
# NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
# PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
# OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
# WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
# OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
# ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

from typing import Any

import torch
import torch.nn as nn
from compressai.fastonn import SelfONN

from torch import Tensor
from compressai.layers.gdn import GDN

from fastonn import *
__all__ = [
    "AttentionBlock",
    "AttentionBlockSelfONN",
    "MaskedConv2d",
    "ResidualBlock",
    "ResidualBlockUpsample",
    "ResidualBlockWithStride",
    "conv3x3",
    "SelfONNconv3x3",
    "subpel_conv3x3",
    "conv1x1",
    "SelfONNconv1x1",
    "CBAMLayer",
]


class MaskedConv2d(nn.Conv2d):
 #
    def __init__(self, *args: Any, mask_type: str = "A", **kwargs: Any):
        super().__init__(*args, **kwargs)

        if mask_type not in ("A", "B"):
            raise ValueError(f'Invalid "mask_type" value "{mask_type}"')

        self.register_buffer("mask", torch.ones_like(self.weight.data))
        _, _, h, w = self.mask.size()
        self.mask[:, :, h // 2, w // 2 + (mask_type == "B") :] = 0
        self.mask[:, :, h // 2 + 1 :] = 0

    def forward(self, x: Tensor) -> Tensor:
        # TODO(begaintj): weight assigment is not supported by torchscript
        self.weight.data *= self.mask
        return super().forward(x)

def conv3x3(in_ch: int, out_ch: int, stride: int = 1) -> nn.Module:
    """3x3 convolution with padding."""
    return nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1)

def SelfONNconv3x3(in_ch: int, out_ch: int, stride: int = 1,q=3):

    return SelfONN2d(in_ch, out_ch, kernel_size=3 ,stride=stride,padding=1,q=3)

def conv_down(in_chn, out_chn, bias=False):
    layer = nn.Conv2d(in_chn, out_chn, kernel_size=4, stride=2, padding=1, bias=bias)
    return layer

def subpel_conv3x3(in_ch: int, out_ch: int, r: int = 1) -> nn.Sequential:
    """3x3 sub-pixel convolution for up-sampling.上采样"""
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch * r ** 2, kernel_size=3, padding=1), nn.PixelShuffle(r)
    )

def conv1x1(in_ch: int, out_ch: int, stride: int = 1) -> nn.Module:
    """1x1 convolution."""
    return nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride)

def SelfONNconv1x1(in_ch: int, out_ch: int, stride: int=1, q=3):

    return SelfONN2d(in_ch,out_ch,kernel_size=1,stride=stride,q=3)

class ResidualBlockWithStride(nn.Module):
    """Residual block with a stride on the first convolution.
    Args:
        in_ch (int): number of input channels
        out_ch (int): number of output channels
        stride (int): stride value (default: 2)
    """

    def __init__(self, in_ch: int, out_ch: int, stride: int = 2):
        super().__init__()
        self.conv1 = conv3x3(in_ch, out_ch, stride=stride)
        self.leaky_relu = nn.LeakyReLU(inplace=True)
        self.conv2 = conv3x3(out_ch, out_ch)
        self.gdn = GDN(out_ch)
        if stride != 1 or in_ch != out_ch:
            self.skip = conv1x1(in_ch, out_ch, stride=stride)
        else:
            self.skip = None

    def forward(self, x: Tensor) -> Tensor:
        identity = x
        out = self.conv1(x)
        out = self.leaky_relu(out)
        out = self.conv2(out)
        out = self.gdn(out)

        if self.skip is not None:
            identity = self.skip(x)

        out += identity
        return out


class ResidualBlockUpsample(nn.Module):
    """Residual block with sub-pixel upsampling on the last convolution.

    Args:
        in_ch (int): number of input channels
        out_ch (int): number of output channels
        upsample (int): upsampling factor (default: 2)
    """

    def __init__(self, in_ch: int, out_ch: int, upsample: int = 2):
        super().__init__()
        self.subpel_conv = subpel_conv3x3(in_ch, out_ch, upsample)
        self.leaky_relu = nn.LeakyReLU(inplace=True)
        self.conv = conv3x3(out_ch, out_ch)
        self.igdn = GDN(out_ch, inverse=True)
        self.upsample = subpel_conv3x3(in_ch, out_ch, upsample)

    def forward(self, x: Tensor) -> Tensor:
        identity = x
        out = self.subpel_conv(x)
        out = self.leaky_relu(out)
        out = self.conv(out)
        out = self.igdn(out)
        identity = self.upsample(x)
        out += identity
        return out

class ResidualBlock(nn.Module):
    """Simple residual block with two 3x3 convolutions.

    Args:
        in_ch (int): number of input channels
        out_ch (int): number of output channels
    """

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.conv1 = conv3x3(in_ch, out_ch, stride)
        self.leaky_relu = nn.LeakyReLU(inplace=True)
        self.conv2 = conv3x3(out_ch, out_ch)
        if in_ch != out_ch or stride > 1:
            self.skip = conv1x1(in_ch, out_ch, stride)
        else:
            self.skip = None

    def forward(self, x: Tensor) -> Tensor:
        identity = x
        out = self.conv1(x)
        out = self.leaky_relu(out)
        out = self.conv2(out)
        out = self.leaky_relu(out)

        if self.skip is not None:
            identity = self.skip(x)

        out = out + identity
        return out

class CBAMLayer(nn.Module):
    def __init__(self, channel=128, reduction=16, spatial_kernel=7):
        super(CBAMLayer, self).__init__()

        # channel attention 压缩H,W为1
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        # shared MLP
        self.mlp = nn.Sequential(
            # nn.Linear(channel, channel // reduction, bias=False)
            nn.Conv2d(channel, channel // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            # nn.Linear(channel // reduction, channel,bias=False)
            nn.Conv2d(channel // reduction, channel, 1, bias=False)
        )

        # spatial attention
        self.conv = nn.Conv2d(2, 1, kernel_size=spatial_kernel,
                              padding=spatial_kernel // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_out = self.mlp(self.max_pool(x))
        avg_out = self.mlp(self.avg_pool(x))
        channel_out = self.sigmoid(max_out + avg_out)
        x = channel_out * x

        max_out, _ = torch.max(x, dim=1, keepdim=True)
        avg_out = torch.mean(x, dim=1, keepdim=True)
        spatial_out = self.sigmoid(self.conv(torch.cat([max_out, avg_out], dim=1)))
        x = spatial_out * x
        return x



class ResidualUnit(nn.Module):
            """Simple residual unit."""
            def __init__(self, N: int):
                super().__init__()
                self.conv = nn.Sequential(
                    conv1x1(N, N // 2),
                    nn.ReLU(inplace=True),
                    conv3x3(N // 2, N // 2),
                    #ModulatedDeformConv2d(N // 2, N // 2, kernel_size=3),
                    nn.ReLU(inplace=True),
                    conv1x1(N // 2, N),
                )
                self.relu = nn.ReLU(inplace=True)
            def forward(self, x: Tensor) -> Tensor:
                identity = x
                out = self.conv(x)
                out += identity
                out = self.relu(out)
                ##print(out.shape)
                return out


class AttentionBlock(nn.Module):

    def __init__(self, N: int):
        super().__init__()
        self.conv0 = nn.ModuleList([
            nn.Conv2d(N, N // 4, 3, 1, 1, 1),
            nn.Conv2d(N, N // 4, 3, 1, 2, 2),
            nn.Conv2d(N, N // 4, 3, 1, 3, 3),
            nn.Conv2d(N, N // 4, 3, 1, 4, 4),
        ])

        self.conv_a = nn.Sequential(ResidualUnit(N), CBAMLayer(), ResidualUnit(N), CBAMLayer(), ResidualUnit(N))
        #self.conv_a = nn.Sequential(ResidualUnit(N), ResidualUnit(N), ResidualUnit(N))
        self.conv_b = nn.Sequential(
            ResidualUnit(N),
            ResidualUnit(N),
            ResidualUnit(N),
            conv1x1(N, N),
        )

    def forward(self, x: Tensor) -> Tensor:
        identity = x
        x = torch.cat([conv(x) for conv in self.conv0], dim=1)
        a = self.conv_a(x)
        #print(a.shape)
        b = self.conv_b(x)
        #print(b.shape)
        out = a * torch.sigmoid(b)
        out += identity
        return out

class AttentionBlockSelfONN(nn.Module):

    def __init__(self, N: int):
        super().__init__()

        class ResidualUnitSelfONN(nn.Module):
            """Simple residual unit."""
            def __init__(self):
                super().__init__()
                self.conv = nn.Sequential(
                    SelfONNconv1x1(N, N // 2,q=3),
                    nn.Tanh(),
                    SelfONNconv3x3(N // 2, N // 2,q=3),
                    nn.Tanh(),
                    SelfONNconv1x1(N // 2,N,q=3),
                )
                self.tanh = nn.Tanh()

            def forward(self, x: Tensor) -> Tensor:
                identity = x
                #print(x.shape)##[16,192,16,16]
                out = self.conv(x)
                #print(out.shape)##[16,192,14,14]
                out += identity
                out = self.tanh(out)
                #print(out.shape)
                return out

        self.conv0 = nn.ModuleList([
            nn.Conv2d(N, N // 4, 3, 1, 1, 1),
            nn.Conv2d(N, N // 4, 3, 1, 2, 2),
            nn.Conv2d(N, N // 4, 3, 1, 3, 3),
            nn.Conv2d(N, N // 4, 3, 1, 4, 4),
        ])

        self.conv_a = nn.Sequential(ResidualUnitSelfONN(), CBAMLayer(), ResidualUnitSelfONN(), CBAMLayer(),ResidualUnitSelfONN())
        self.conv_b = nn.Sequential(
            ResidualUnit(N),
            ResidualUnit(N),
            ResidualUnit(N),
            #ResidualUnitONN(),
            conv1x1(N, N),
        )

    def forward(self, x: Tensor) -> Tensor:
        identity = x
        x = torch.cat([conv(x) for conv in self.conv0], dim=1)
        a = self.conv_a(x)
        #print(a.shape)
        b = self.conv_b(x)
        #print(b.shape)
        out = a * torch.sigmoid(b)
        out += identity
        return out



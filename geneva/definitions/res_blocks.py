# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""
Blocks definitions used in the ResBlock architecture
used by https://arxiv.org/abs/1802.05637
"""

import torch
from torch import nn
import torch.nn.utils.spectral_norm as spectral_norm

from geneva.definitions.self_attention import SelfAttention
from geneva.definitions.activations import ACTIVATIONS
from geneva.definitions.conditional_batchnorm import ConditionalBatchNorm2d


class Conv1x1(nn.Conv2d):
    def __init__(self, ch_in, ch_out, ker=1, str=1, pad=0, bias=False):
        super().__init__(ch_in, ch_out,
                         kernel_size=ker, stride=str,
                         padding=pad, bias=bias)


class Conv3x3(nn.Conv2d):
    def __init__(self, ch_in, ch_out, ker=3, str=1, pad=1, bias=False):
        super().__init__(ch_in, ch_out,
                         kernel_size=ker, stride=str,
                         padding=pad, bias=bias)


class ResUpBlock(nn.Module):
    def __init__(self, channels_in, channels_out, embedding_dim,
                 conditional=True, self_attn=False,
                 use_spectral_norm=True, activation='relu'):
        super().__init__()
        self.conditional = conditional

        self.activation = ACTIVATIONS[activation]
        self.upsampler = nn.Upsample(scale_factor=2, mode='nearest')

        if conditional:
            self.bn1 = ConditionalBatchNorm2d(channels_in, embedding_dim)
            self.bn2 = ConditionalBatchNorm2d(channels_out, embedding_dim)
        else:
            self.bn1 = nn.BatchNorm2d(channels_in)
            self.bn2 = nn.BatchNorm2d(channels_out)

        self.conv1 = Conv3x3(channels_in, channels_out)
        self.conv2 = Conv3x3(channels_out, channels_out)
        self.conv_shortcut = Conv1x1(channels_in, channels_out)

        if use_spectral_norm:
            self.conv1 = spectral_norm(self.conv1)
            self.conv2 = spectral_norm(self.conv2)
            self.conv_shortcut = spectral_norm(self.conv_shortcut)

        self.apply_sa = self_attn
        if self.apply_sa:
            self.self_attention = SelfAttention(channels_out)

    def forward(self, x, y=None):
        x_residual = x

        x = self.bn1(x, y) if self.conditional else self.bn1(x)
        x = self.activation(x)
        x = self.upsampler(x)
        x = self.conv1(x)

        x = self.bn2(x, y) if self.conditional else self.bn2(x)
        x = self.activation(x)
        x = self.conv2(x)

        if self.apply_sa:
            x = self.self_attention(x)

        x_residual = self.upsampler(x_residual)
        x_residual = self.conv_shortcut(x_residual)

        return x + x_residual


class ResDownBlock(nn.Module):
    def __init__(self, channels_in, channels_out, downsample=True,
                 self_attn=False, first_block=False, activation='relu',
                 use_spectral_norm=True):
        super().__init__()
        self.first_block = first_block

        self.activation = ACTIVATIONS[activation]
        self.downsample = downsample
        self.downsampler = torch.nn.AvgPool2d(kernel_size=2)

        self.conv1 = Conv3x3(channels_in, channels_out)
        self.conv2 = Conv3x3(channels_out, channels_out)

        if use_spectral_norm:
            self.conv1 = spectral_norm(self.conv1)
            self.conv2 = spectral_norm(self.conv2)

        if downsample:
            self.conv_shortcut = Conv1x1(channels_in, channels_out)
            if use_spectral_norm:
                self.conv_shortcut = spectral_norm(self.conv_shortcut)

        self.apply_sa = self_attn
        if self.apply_sa:
            self.self_attention = SelfAttention(channels_out)

    def forward(self, x):
        x_residual = x

        if not self.first_block:
            x = self.activation(x)

        x = self.conv1(x)

        x = self.activation(x)
        x = self.conv2(x)

        if self.downsample:
            x = self.downsampler(x)

        if self.apply_sa:
            x = self.self_attention(x)

        if self.downsample:
            if self.first_block:
                x_residual = self.downsampler(x_residual)
                x_residual = self.conv_shortcut(x_residual)
            else:
                x_residual = self.conv_shortcut(x_residual)
                x_residual = self.downsampler(x_residual)

        return x + x_residual

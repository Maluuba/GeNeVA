# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""
An Implementation of self-attention module.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm


class SelfAttention(nn.Module):
    """Self Attention Module as in https://arxiv.org/pdf/1805.08318.pdf
    """
    def __init__(self, C):
        """
        Args:
            C: Number of channels in the input feature maps torch attend over.
        """
        super(SelfAttention, self).__init__()

        self.f_x = spectral_norm(
            nn.Conv2d(in_channels=C, out_channels=C // 8, kernel_size=1))
        self.g_x = spectral_norm(
            nn.Conv2d(in_channels=C, out_channels=C // 8, kernel_size=1))
        self.h_x = spectral_norm(
            nn.Conv2d(in_channels=C, out_channels=C, kernel_size=1))

        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        """Projects input x to three different features spaces using f(x),
        g(x) and h(x). Applying outer product to outputs of f(x) and g(x).
        A softmax is applied on top to get a attention map. The attention map
        is applied to the output of h(x) to get the final attended features.
        Args:
            x: input features maps. shape=(B,C,H,W).
        Returns:
            y: Attended features of shape=(B,C,H,W).

        """
        B, C, H, W = x.size()
        N = H * W

        f = self.f_x(x).view(B, C // 8, N)
        g = self.g_x(x).view(B, C // 8, N)

        s = torch.bmm(f.permute(0, 2, 1), g)  # f(x)^{T} * g(x)
        beta = F.softmax(s, dim=1)

        h = self.h_x(x).view(B, C, N)
        o = torch.bmm(h, beta).view(B, C, H, W)

        y = self.gamma * o + x

        return y

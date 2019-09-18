# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""
Conditional Batch Norm Layer following
https://arxiv.org/abs/1802.05637
"""
import torch.nn as nn


class ConditionalBatchNorm2d(nn.Module):
    def __init__(self, in_channels, conditon_dim):
        super().__init__()
        self.channels = in_channels
        self.bn = nn.BatchNorm2d(in_channels, affine=False)
        self.fc = nn.Linear(conditon_dim,
                            in_channels * 2)

        self.fc.weight.data[:, :in_channels] = 1
        self.fc.weight.data[:, in_channels:] = 0

    def forward(self, activations, condition):
        condition = self.fc(condition)
        gamma = condition[:, :self.channels]\
            .unsqueeze(2).unsqueeze(3)
        beta = condition[:, self.channels:]\
            .unsqueeze(2).unsqueeze(3)

        activations = self.bn(activations)
        return activations.mul(gamma).add(beta)

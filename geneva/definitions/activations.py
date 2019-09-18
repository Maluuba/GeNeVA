# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""Activation functions manager"""
import torch


ACTIVATIONS = {
    'relu': torch.nn.ReLU(),
    'leaky_relu': torch.nn.LeakyReLU(),
    'selu': torch.nn.SELU(),
}

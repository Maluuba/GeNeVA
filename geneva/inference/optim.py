# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""Optimizers Manager"""

import torch


def _adam(parameters, lr, beta_1=0.9, beta_2=0.999, weight_decay=0):
    return torch.optim.Adam(params=parameters,
                            lr=lr,
                            betas=(beta_1, beta_2),
                            weight_decay=weight_decay)


def _rmsprop(parameters, lr):
    return torch.optim.RMSprop(params=parameters,
                               lr=lr)


OPTIM = {
    'adam': _adam,
    'rmsprop': _rmsprop,
}

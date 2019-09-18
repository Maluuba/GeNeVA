# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""
Regualerization methods for GANs
"""

import torch
from torch import autograd


def gradient_penalty(discriminator_out, data_point):
    """Regularizing GAN training by penalizing
    gradients for real data only, such that it does
    not produce orthogonal gradients to the data
    manifold at equilibrium.

    Follows:
    Which Training Methods for GANs do actually Converge? eq.(9)
    Args:
        - discriminator_out: output logits from the
        discriminator
        - data_point: real data point (x_real).
    Returns:
        reg: regularization value.

    Code from https://github.com/LMescheder/GAN_stability/blob/c66a8a40b8d21ee222b1b7d7b5747e3c639b83ef/gan_training/train.py
    License:
        MIT License

        Copyright (c) 2018 Lars Mescheder

        Permission is hereby granted, free of charge, to any person obtaining a copy
        of this software and associated documentation files (the "Software"), to deal
        in the Software without restriction, including without limitation the rights
        to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
        copies of the Software, and to permit persons to whom the Software is
        furnished to do so, subject to the following conditions:

        The above copyright notice and this permission notice shall be included in all
        copies or substantial portions of the Software.

        THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
        IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
        FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
        AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
        LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
        OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
        SOFTWARE.
    """
    batch_size = data_point.size(0)
    grad_dout = autograd.grad(outputs=discriminator_out.sum(),
                              inputs=data_point,
                              create_graph=True,
                              retain_graph=True,
                              only_inputs=True)[0]

    grad_dout2 = grad_dout.pow(2)
    assert(grad_dout2.size() == data_point.size())
    reg = grad_dout2.view(batch_size, -1).sum(1)

    return reg


def kl_penalty(mu, logvar):
    """ -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2) """
    KLD_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
    KLD = torch.mean(KLD_element).mul_(-0.5)
    return KLD

# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""Objective/Loss functions Manager"""
import torch.nn.functional as F


class HingeAdversarial():
    """Hinge Adversarial Loss as used in
    https://arxiv.org/pdf/1802.05957.pdf"""
    @staticmethod
    def discriminator(real, fake, wrong=None, wrong_weight=0.):
        """Discriminator loss term
        Args:
            - real: D(x)
            - fake: D(G(z))
            - wrong: D(x^) (wrong image and text pairing) [optional]
        Returns:
            - loss: Tensor containing the loss mean
        """
        l_real = F.relu(1. - real).mean()
        l_fake = F.relu(1. + fake).mean()

        if wrong is None:
            return l_real + l_fake
        else:
            l_wrong = F.relu(1. + wrong).mean()
            return l_real + wrong_weight * l_wrong + (1. - wrong_weight) * l_fake

    @staticmethod
    def generator(fake):
        """Generator Loss Term
        Args:
            - fake: D(G(z))
        Returns:
            - loss: Tensor containing the loss mean
        """
        return -fake.mean()


class Adversarial():
    """Classical Adversarial loss"""
    @staticmethod
    def discriminator(real, fake):
        """Discriminator loss term
        Args:
            - real: D(x)
            - fake: D(G(z))
        Returns:
            - loss: Tensor containing the loss mean
        """
        return F.softplus(-real).mean() \
            + F.softplus(fake).mean()

    @staticmethod
    def generator(fake):
        """Generator Loss Term
        Args:
            - fake: D(G(z))
        Returns:
            - loss: Tensor containing the loss mean
        """
        return F.softplus(-fake).mean()


LOSSES = {
    'classical': Adversarial,
    'hinge': HingeAdversarial,
}

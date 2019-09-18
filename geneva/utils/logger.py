# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""Logging Manager"""
import os


class Logger():
    def __init__(self, path, exp_name):
        """Logger instance constructor
        Args:
            - path: file path to write logs to
            - exp_name: experiment title
        """
        path = os.path.join(path, exp_name)
        self.log_path = os.path.join(path, 'logs.txt')

        if not os.path.exists(path):
            os.mkdir(path)

    def write_config(self, config):
        """ Logs Config file"""
        print(config)
        with open(self.log_path, 'a') as f:
            f.write(config + '\n')

    def write(self, epoch, iteration, d_real, d_fake, d_loss, g_loss):
        """Logs the iteration details
            Args:
                - epoch: epoch number
                - iteration: Iteration number (overall)
                - d_real: values of D(x)
                - d_fake: values of D(G(z))
                - d_loss: Discriminator loss
                - g_loss: Generator loss
        """
        info = ('Epoch={}, Iteration={}, D(x)={:.2f}, D(G(z))={:.2f},'
                'Disc_loss={:.2f}, Gen_loss={:.2f}') \
            .format(epoch,
                    iteration,
                    d_real.mean(),
                    d_fake.mean(),
                    d_loss,
                    g_loss)

        with open(self.log_path, 'a') as f:
            f.write(info + '\n')
        print(info)

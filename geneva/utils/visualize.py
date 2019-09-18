# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""Visualization Manager"""
import numpy as np
from visdom import Visdom


class VisdomPlotter():
    """Plots to Visdom"""
    def __init__(self, env_name='TellDrawRepeat', server='http://localhost'):
        """Visdom instance constuctor
        Args:
            - env_name: name of the environment (str)
            - server: endpoint (url)
        """
        self.viz = Visdom(server=server)
        self.env = env_name
        self.plots = {}
        self.real_aggregate = []
        self.fake_aggregate = []
        self.sigma_aggregate = []

    def plot(self, var_name, split_name, x, y, xlabel='iteration'):
        """Plots a line
        Args:
            - var_name: plot name (str)
            - split_name: split  name (str)
            - x: x_axis values [e.g, epoch/iteration] (float)
            - y: y_axis values (float)
            - xlabel: label of the x_axis (str)
        """
        if var_name not in self.plots:
            self.plots[var_name] = self.viz.line(
                X=np.array([x, x]),
                Y=np.array([y, y]),
                env=self.env,
                opts=dict(
                    legend=[split_name],
                    title=var_name,
                    xlabel=xlabel,
                    ylabel=var_name))
        else:
            self.viz.line(
                X=np.array([x]),
                Y=np.array([y]),
                env=self.env,
                win=self.plots[var_name],
                name=split_name,
                update='append')

    def draw(self, var_name, images, nrow=8):
        """Shows a grid of images
        Args:
            - var_name: plot title (str)
            - images: np.array of images (np.array)
            - nrow: number of images per row (integer)
        """
        images = images[:16] * 127.5 + 127.5
        if var_name not in self.plots:
            self.plots[var_name] = self.viz.images(images, env=self.env, nrow=nrow)
        else:
            self.viz.images(images, env=self.env, win=self.plots[var_name], nrow=nrow)

    def histogram(self):
        """Draws a hsitogram of D(x) and D(G(z))
        Takes no arguments, it uses the values saved using
        self.track
        """
        self._histogram('D(x)',
                        np.array(self.real_aggregate))
        self._histogram('D(G(z))',
                        np.array(self.fake_aggregate))
        self.fake_aggregate = []
        self.real_aggregate = []

        if len(self.sigma_aggregate) > 0:
            self._histogram('Sigma Gate',
                            np.array(self.sigma_aggregate))
            self.sigma_aggregate = []

    def track(self, real, fake):
        """Aggregates values of D(x) and D(G(z))
        over all the iterations. Values will be used to draw
        a histogram using self.histogram()
        Args:
            - real: tensor of D(x) (torch.FloatTensor)
            - fake: tensor of D(G(z)) (torch.FloatTensor)
        """
        self.real_aggregate.extend(real)
        self.fake_aggregate.extend(fake)

    def track_sigma(self, sigma):
        """Aggregates the values of sigma,
        the gating vector of the genrertor
        """
        if sigma is None:
            return

        sigma = sigma.mean(dim=1).data.cpu().numpy()
        self.sigma_aggregate.extend(sigma)

    def _histogram(self, var_name, data):
        if var_name not in self.plots:
            self.plots[var_name] = self.viz.histogram(
                X=data,
                env=self.env,
                opts=dict(numbins=20,
                          title=var_name))
        else:
            self.viz.histogram(
                X=data,
                env=self.env,
                win=self.plots[var_name],
                opts=dict(numbins=20,
                          title=var_name))

    def write(self, text, var_name='dialog'):
        """Shows Text in HTML format
        Args:
            - text: list of sentences (list)
            -var_name: plost title (str)
        """
        text = [t.replace('<', '#') for t in text]
        text = [t.replace('>', '#') for t in text]
        text = str.join('<ol/ ><br/ ><ol>', text[:32])
        if var_name not in self.plots:
            self.plots[var_name] = self.viz.text(text)
        else:
            self.viz.text(text, env=self.env, win=self.plots[var_name])

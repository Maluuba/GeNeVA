# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""
Generator Factory for creating a generator instance
with specific properties.
"""
import torch
import torch.nn as nn

from geneva.definitions.activations import ACTIVATIONS
from geneva.definitions.conditioning_augmentor import ConditioningAugmentor
from geneva.definitions.res_blocks import Conv3x3, ResUpBlock


class GeneratorFactory():
    @staticmethod
    def create_instance(cfg):
        """Creates an instance of a generator that matches
        the given arguments
        Args:
              cfg: Experiment configurations. This method needs
              [model, arch, conditioning] flags.
        Returns:
             generator: generator instance that implements nn.Module
        """
        if cfg.gan_type == 'recurrent_gan':
            return GeneratorRecurrentGANRes(cfg)
        else:
            raise Exception('Model {} not available. Please select'
                            'one of {recurrent_gan}'
                            .format(cfg.model))


class GeneratorRecurrentGANRes(nn.Module):
    """
    Recurrent GAN generator for 128x128 image generation,
    conditioned on previous image and current instruction
    """
    def __init__(self, cfg):
        super().__init__()
        self.conditional = cfg.conditioning is not None

        self.z_dim = cfg.noise_dim + cfg.conditioning_dim
        condition_dim = cfg.hidden_dim

        self.fc1 = nn.Linear(self.z_dim, 1024 * 4 * 4)
        # state size. (1024) x 4 x 4
        self.resup1 = ResUpBlock(1024, 1024, condition_dim,
                                 conditional=self.conditional,
                                 use_spectral_norm=cfg.generator_sn,
                                 activation=cfg.activation)

        # state size. (1024) x 8 x 8
        self.resup2 = ResUpBlock(1024, 512, condition_dim,
                                 conditional=self.conditional,
                                 use_spectral_norm=cfg.generator_sn,
                                 activation=cfg.activation)

        extra_channels = 0
        if cfg.use_fg and cfg.gen_fusion == 'concat':
            extra_channels = 512

        # state size. (512) x 16 x 16
        self.resup3 = ResUpBlock(512 + extra_channels, 256, condition_dim,
                                 conditional=self.conditional,
                                 self_attn=cfg.self_attention,
                                 use_spectral_norm=cfg.generator_sn,
                                 activation=cfg.activation)

        # state size. (256) x 32 x 32
        self.resup4 = ResUpBlock(256, 128, condition_dim,
                                 conditional=self.conditional,
                                 use_spectral_norm=cfg.generator_sn,
                                 activation=cfg.activation)

        # state size. (128) x 64 x 64
        self.resup5 = ResUpBlock(128, 64, condition_dim,
                                 conditional=self.conditional,
                                 use_spectral_norm=cfg.generator_sn,
                                 activation=cfg.activation)

        # state size. (64) x 128 x 128
        self.bn = nn.BatchNorm2d(64)
        self.activation = ACTIVATIONS[cfg.activation]
        self.conv = Conv3x3(64, 3)
        # state size. (3) x 128 x 128
        self.tanh = nn.Tanh()

        if cfg.cond_kl_reg is not None:
            self.condition_projector = ConditioningAugmentor(
                condition_dim,
                cfg.conditioning_dim)
        else:
            self.condition_projector = nn.Linear(condition_dim, cfg.conditioning_dim)

        self.gate = nn.Sequential(
            nn.Linear(condition_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.Sigmoid()
        )
        self.cfg = cfg

    def forward(self, z, y, img_feats):
        mu, logvar = None, None
        cond_y = self.condition_projector(y)
        if self.cfg.cond_kl_reg is not None:
            cond_y, mu, logvar = cond_y

        z = torch.cat([z, cond_y], dim=1)

        x = self.fc1(z)
        x = x.view(-1, 1024, 4, 4)
        x = self.resup1(x, y)
        x = self.resup2(x, y)

        sigma = None
        if self.cfg.use_fg:
            if self.cfg.gen_fusion == 'gate':
                sigma = self.gate(y)
                sigma = sigma.unsqueeze(2).unsqueeze(3)
                x = x * sigma + img_feats * (1 - sigma)
            elif self.cfg.gen_fusion == 'concat':
                x = torch.cat([x, img_feats], dim=1)

        x = self.resup3(x, y)
        x = self.resup4(x, y)
        x = self.resup5(x, y)
        x = self.bn(x)
        x = self.activation(x)
        x = self.conv(x)
        x = self.tanh(x)

        return x, mu, logvar, sigma

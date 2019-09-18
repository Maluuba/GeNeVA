# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""Image encoder using ResBlocks"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from geneva.definitions.res_blocks import ResDownBlock


class ImageEncoder(nn.Module):
    def __init__(self, cfg):
        """Encodes Image to 16x16 features maps of depth 256
        , return the 16x16 features as well as the global sum
        pooled features(shape=512)"""
        super().__init__()
        self.encode_image = cfg.use_fg

        if self.encode_image:
            if cfg.img_encoder_type == 'res_blocks':
                self.image_encoder = nn.Sequential(
                    # 3 x 128 x 128
                    ResDownBlock(3, 64, downsample=True,
                                 use_spectral_norm=False),
                    # 64 x 64 x 64
                    nn.BatchNorm2d(64),
                    ResDownBlock(64, 128, downsample=True,
                                 use_spectral_norm=False),
                    # 128 x 32 x 32
                    nn.BatchNorm2d(128),
                    ResDownBlock(128, cfg.image_feat_dim,
                                 downsample=True,
                                 use_spectral_norm=False),
                    nn.BatchNorm2d(cfg.image_feat_dim),
                    # 256 x 16 x 16
                )
            elif cfg.img_encoder_type == 'conv':
                self.image_encoder = nn.Sequential(
                    nn.Conv2d(3, 64, 4, 2, 1, bias=False),
                    nn.ReLU(),
                    nn.BatchNorm2d(64),
                    nn.Conv2d(64, 128, 4, 2, 1, bias=False),
                    nn.ReLU(),
                    nn.BatchNorm2d(128),
                    nn.Conv2d(128, cfg.image_feat_dim, 4, 2, 1,
                              bias=False),
                    nn.BatchNorm2d(cfg.image_feat_dim),
                )

            self.object_detector = nn.Linear(cfg.image_feat_dim,
                                             cfg.num_objects)

        self.cfg = cfg

    def forward(self, img):
        if not self.encode_image:
            return None, None, None

        image_features = self.image_encoder(img)
        pooled_features = torch.sum(image_features, dim=(2, 3))

        object_detections = F.sigmoid(self.object_detector(pooled_features))
        return image_features, pooled_features, object_detections

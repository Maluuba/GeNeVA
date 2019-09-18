# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""
Encoder that fuses the textual and image features to create the condition
input to the RNN
"""
import torch.nn as nn


class ConditionEncoder(nn.Module):
    def __init__(self, cfg):
        super(ConditionEncoder, self).__init__()

        self.text_projection = nn.Linear(cfg.embedding_dim,
                                         cfg.projected_text_dim)
        self.bn = nn.BatchNorm1d(cfg.input_dim)
        self.cfg = cfg

    def forward(self, text_features, image_features, current_image_feat=None):
        text_features = self.text_projection(text_features)

        fused_features = text_features

        fused_features = self.bn(fused_features)

        return fused_features, current_image_feat

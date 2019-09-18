# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""
An inference time implementation for
recurrent GAN. Main difference is connecting
last time step generations to the next time
step. (Getting rid of Teacher Forcing)
"""
import os
from glob import glob

import torch
import torch.nn as nn
from torch.nn import DataParallel
import cv2

from geneva.models.networks.generator_factory import GeneratorFactory
from geneva.models.image_encoder import ImageEncoder
from geneva.models.sentence_encoder import SentenceEncoder
from geneva.models.condition_encoder import ConditionEncoder


class InferenceRecurrentGAN():
    def __init__(self, cfg):
        """A recurrent GAN model, each time step an generated image
        (x'_{t-1}) and the current question q_{t} are fed to the RNN
        to produce the conditioning vector for the GAN.
        The following equations describe this model:

            - c_{t} = RNN(h_{t-1}, q_{t}, x^{~}_{t-1})
            - x^{~}_{t} = G(z | c_{t})
        """
        super(InferenceRecurrentGAN, self).__init__()
        self.generator = DataParallel(
            GeneratorFactory.create_instance(cfg),
            device_ids=[0]).cuda()

        self.rnn = nn.DataParallel(
            nn.GRU(cfg.input_dim, cfg.hidden_dim,
                   batch_first=False),
            dim=1,
            device_ids=[0]).cuda()

        self.layer_norm = nn.DataParallel(nn.LayerNorm(cfg.hidden_dim),
                                          device_ids=[0]).cuda()

        self.image_encoder = DataParallel(ImageEncoder(cfg),
                                          device_ids=[0]).cuda()

        self.condition_encoder = DataParallel(ConditionEncoder(cfg),
                                              device_ids=[0]).cuda()

        self.sentence_encoder = nn.DataParallel(SentenceEncoder(cfg),
                                                device_ids=[0]).cuda()

        self.cfg = cfg
        self.results_path = cfg.results_path
        if not os.path.exists(cfg.results_path):
            os.mkdir(cfg.results_path)

    def predict(self, batch):
        with torch.no_grad():
            batch_size = len(batch['image'])
            max_seq_len = batch['image'].size(1)
            scene_id = batch['scene_id']

            # Initial inputs for the RNN set to zeros
            prev_image = torch.FloatTensor(batch['background']).unsqueeze(0) \
                .repeat(batch_size, 1, 1, 1)
            hidden = torch.zeros(1, batch_size, self.cfg.hidden_dim)
            generated_images = []
            gt_images = []

            for t in range(max_seq_len):
                turns_word_embedding = batch['turn_word_embedding'][:, t]
                turns_lengths = batch['turn_lengths'][:, t]

                image_feature_map, image_vec, object_detections = \
                    self.image_encoder(prev_image)

                turn_embedding = self.sentence_encoder(turns_word_embedding, turns_lengths)
                rnn_condition, _ = self.condition_encoder(turn_embedding,
                                                          image_vec)

                rnn_condition = rnn_condition.unsqueeze(0)
                output, hidden = self.rnn(rnn_condition,
                                          hidden)

                output = output.squeeze(0)
                output = self.layer_norm(output)

                generated_image = self._forward_generator(batch_size, output,
                                                          image_feature_map)

                if (not self.cfg.inference_save_last_only) or (self.cfg.inference_save_last_only and t == max_seq_len - 1):
                    generated_images.append(generated_image)
                    gt_images.append(batch['image'][:, t])
                prev_image = generated_image

        _save_predictions(generated_images, batch['turn'], scene_id, self.results_path, gt_images)

    def _forward_generator(self, batch_size, condition, image_feature_maps):
        noise = torch.FloatTensor(batch_size,
                                  self.cfg.noise_dim).normal_(0, 1).cuda()

        fake_images, _, _, _ = self.generator(noise, condition, image_feature_maps)

        return fake_images

    def load(self, pre_trained_path, iteration=None):
        snapshot = _read_weights(pre_trained_path, iteration)

        self.generator.load_state_dict(snapshot['generator_state_dict'])
        self.rnn.load_state_dict(snapshot['rnn_state_dict'])
        self.layer_norm.load_state_dict(snapshot['layer_norm_state_dict'])
        self.image_encoder.load_state_dict(snapshot['image_encoder_state_dict'])
        self.condition_encoder.load_state_dict(snapshot['condition_encoder_state_dict'])
        self.sentence_encoder.load_state_dict(snapshot['sentence_encoder_state_dict'])


def _save_predictions(images, text, scene_id, results_path, gt_images):
    for i, scene in enumerate(scene_id):
        if not os.path.exists(os.path.join(results_path, str(scene))):
            os.mkdir(os.path.join(results_path, str(scene)))
        if not os.path.exists(os.path.join(results_path, str(scene) + '_gt')):
            os.mkdir(os.path.join(results_path, str(scene) + '_gt'))
        for t in range(len(images)):
            if t >= len(text[i]):
                continue
            image = (images[t][i].data.cpu().numpy() + 1) * 128
            image = image.transpose(1, 2, 0)[..., ::-1]
            query = text[i][t]
            gt_image = (gt_images[t][i].data.cpu().numpy() + 1) * 128
            gt_image = gt_image.transpose(1, 2, 0)[..., ::-1]
            cv2.imwrite(os.path.join(results_path, str(scene), '{}_{}.png'.format(t, query)),
                        image)
            cv2.imwrite(os.path.join(results_path, str(scene) + '_gt', '{}_{}.png'.format(t, query)),
                        gt_image)


def _read_weights(pre_trained_path, iteration):
    if iteration is None:
        iteration = ''
    iteration = str(iteration)
    try:
        snapshot = torch.load(glob('{}/snapshot_{}*'.format(pre_trained_path, iteration))[0])
    except IndexError:
        snapshot = torch.load(pre_trained_path)
    return snapshot

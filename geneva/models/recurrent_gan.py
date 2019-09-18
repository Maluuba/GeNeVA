# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""
A recurrent GAN model that draws images
based on dialog/description turns in sequence
"""
import os
import gc

import torch
import torch.nn as nn
from torch.nn import DataParallel
import numpy as np

from geneva.models.networks.generator_factory import GeneratorFactory
from geneva.models.networks.discriminator_factory import DiscriminatorFactory
from geneva.criticism.losses import LOSSES
from geneva.models.image_encoder import ImageEncoder
from geneva.models.sentence_encoder import SentenceEncoder
from geneva.models.condition_encoder import ConditionEncoder
from geneva.inference.optim import OPTIM
from geneva.definitions.regularizers import gradient_penalty, kl_penalty
from geneva.utils.logger import Logger
from geneva.models import _recurrent_gan


class RecurrentGAN():
    def __init__(self, cfg):
        """A recurrent GAN model, each time step a generated image
        (x'_{t-1}) and the current question q_{t} are fed to the RNN
        to produce the conditioning vector for the GAN.
        The following equations describe this model:

            - c_{t} = RNN(h_{t-1}, q_{t}, x^{~}_{t-1})
            - x^{~}_{t} = G(z | c_{t})
        """
        super(RecurrentGAN, self).__init__()

        # region Models-Instantiation

        self.generator = DataParallel(
            GeneratorFactory.create_instance(cfg)).cuda()

        self.discriminator = DataParallel(
            DiscriminatorFactory.create_instance(cfg)).cuda()

        self.rnn = nn.DataParallel(nn.GRU(cfg.input_dim,
                                          cfg.hidden_dim,
                                          batch_first=False), dim=1).cuda()

        self.layer_norm = nn.DataParallel(nn.LayerNorm(cfg.hidden_dim)).cuda()

        self.image_encoder = DataParallel(ImageEncoder(cfg)).cuda()

        self.condition_encoder = DataParallel(ConditionEncoder(cfg)).cuda()

        self.sentence_encoder = nn.DataParallel(SentenceEncoder(cfg)).cuda()

        # endregion

        # region Optimizers

        self.generator_optimizer = OPTIM[cfg.generator_optimizer](
            self.generator.parameters(),
            cfg.generator_lr,
            cfg.generator_beta1,
            cfg.generator_beta2,
            cfg.generator_weight_decay)

        self.discriminator_optimizer = OPTIM[cfg.discriminator_optimizer](
            self.discriminator.parameters(),
            cfg.discriminator_lr,
            cfg.discriminator_beta1,
            cfg.discriminator_beta2,
            cfg.discriminator_weight_decay)

        self.rnn_optimizer = OPTIM[cfg.rnn_optimizer](
            self.rnn.parameters(),
            cfg.rnn_lr)

        self.sentence_encoder_optimizer = OPTIM[cfg.gru_optimizer](
            self.sentence_encoder.parameters(),
            cfg.gru_lr)

        self.use_image_encoder = cfg.use_fg
        feature_encoding_params = list(self.condition_encoder.parameters())
        if self.use_image_encoder:
            feature_encoding_params += list(self.image_encoder.parameters())

        self.feature_encoders_optimizer = OPTIM['adam'](
            feature_encoding_params,
            cfg.feature_encoder_lr
        )

        # endregion

        # region Criterion

        self.criterion = LOSSES[cfg.criterion]()
        self.aux_criterion = DataParallel(torch.nn.BCELoss()).cuda()

        # endregion

        self.cfg = cfg
        self.logger = Logger(cfg.log_path, cfg.exp_name)

    def train_batch(self, batch, epoch, iteration, visualizer, logger):
        """
        The training scheme follows the following:
            - Discriminator and Generator is updated every time step.
            - RNN, SentenceEncoder and ImageEncoder parameters are
            updated every sequence
        """
        batch_size = len(batch['image'])
        max_seq_len = batch['image'].size(1)

        prev_image = torch.FloatTensor(batch['background'])
        prev_image = prev_image.unsqueeze(0) \
            .repeat(batch_size, 1, 1, 1)
        disc_prev_image = prev_image

        # Initial inputs for the RNN set to zeros
        hidden = torch.zeros(1, batch_size, self.cfg.hidden_dim)
        prev_objects = torch.zeros(batch_size, self.cfg.num_objects)

        teller_images = []
        drawer_images = []
        added_entities = []

        for t in range(max_seq_len):
            image = batch['image'][:, t]
            turns_word_embedding = batch['turn_word_embedding'][:, t]
            turns_lengths = batch['turn_lengths'][:, t]
            objects = batch['objects'][:, t]
            seq_ended = t > (batch['dialog_length'] - 1)

            image_feature_map, image_vec, object_detections = \
                self.image_encoder(prev_image)
            _, current_image_feat, _ = self.image_encoder(image)

            turn_embedding = self.sentence_encoder(turns_word_embedding,
                                                   turns_lengths)
            rnn_condition, current_image_feat = \
                self.condition_encoder(turn_embedding,
                                       image_vec,
                                       current_image_feat)

            rnn_condition = rnn_condition.unsqueeze(0)
            output, hidden = self.rnn(rnn_condition,
                                      hidden)

            output = output.squeeze(0)
            output = self.layer_norm(output)

            fake_image, mu, logvar, sigma = self._forward_generator(batch_size,
                                                                    output.detach(),
                                                                    image_feature_map)

            visualizer.track_sigma(sigma)

            hamming = objects - prev_objects
            hamming = torch.clamp(hamming, min=0)

            d_loss, d_real, d_fake, aux_loss, discriminator_gradient = \
                self._optimize_discriminator(image,
                                             fake_image.detach(),
                                             disc_prev_image,
                                             output,
                                             seq_ended,
                                             hamming,
                                             self.cfg.gp_reg,
                                             self.cfg.aux_reg)

            g_loss, generator_gradient = \
                self._optimize_generator(fake_image,
                                         disc_prev_image.detach(),
                                         output.detach(),
                                         objects,
                                         self.cfg.aux_reg,
                                         seq_ended,
                                         mu,
                                         logvar)

            if self.cfg.teacher_forcing:
                prev_image = image
            else:
                prev_image = fake_image

            disc_prev_image = image
            prev_objects = objects

            if (t + 1) % 2 == 0:
                prev_image = prev_image.detach()

            rnn_grads = []
            gru_grads = []
            condition_encoder_grads = []
            img_encoder_grads = []

            if t == max_seq_len - 1:
                rnn_gradient, gru_gradient, condition_gradient,\
                    img_encoder_gradient = self._optimize_rnn()

                rnn_grads.append(rnn_gradient.data.cpu().numpy())
                gru_grads.append(gru_gradient.data.cpu().numpy())
                condition_encoder_grads.append(condition_gradient.data.cpu().numpy())

                if self.use_image_encoder:
                    img_encoder_grads.append(img_encoder_gradient.data.cpu().numpy())

                visualizer.track(d_real, d_fake)

            hamming = hamming.data.cpu().numpy()[0]
            teller_images.extend(image[:4].data.numpy())
            drawer_images.extend(fake_image[:4].data.cpu().numpy())
            entities = str.join(',', list(batch['entities'][hamming > 0]))
            added_entities.append(entities)

        if iteration % self.cfg.vis_rate == 0:
            visualizer.histogram()
            self._plot_losses(visualizer, g_loss, d_loss, aux_loss, iteration)
            rnn_gradient = np.array(rnn_grads).mean()
            gru_gradient = np.array(gru_grads).mean()
            condition_gradient = np.array(condition_encoder_grads).mean()
            img_encoder_gradient = np.array(img_encoder_grads).mean()
            rnn_grads, gru_grads = [], []
            condition_encoder_grads, img_encoder_grads = [], []
            self._plot_gradients(visualizer, rnn_gradient, generator_gradient,
                                 discriminator_gradient, gru_gradient, condition_gradient,
                                 img_encoder_gradient, iteration)
            self._draw_images(visualizer, teller_images, drawer_images, nrow=4)
            self.logger.write(epoch, iteration, d_real, d_fake, d_loss, g_loss)

            if isinstance(batch['turn'], list):
                batch['turn'] = np.array(batch['turn']).transpose()

            visualizer.write(batch['turn'][0])
            visualizer.write(added_entities, var_name='entities')
            teller_images = []
            drawer_images = []

        if iteration % self.cfg.save_rate == 0:
            path = os.path.join(self.cfg.log_path,
                                self.cfg.exp_name)

            self._save(fake_image[:4], path, epoch,
                       iteration)
            if not self.cfg.debug:
                self.save_model(path, epoch, iteration)

    def _forward_generator(self, batch_size, condition, image_feature_maps):
        noise = torch.FloatTensor(batch_size,
                                  self.cfg.noise_dim).normal_(0, 1).cuda()

        fake_images, mu, logvar, sigma = self.generator(noise, condition,
                                                        image_feature_maps)

        return fake_images, mu, logvar, sigma

    def _optimize_discriminator(self, real_images, fake_images, prev_image,
                                condition, mask, objects, gp_reg=0, aux_reg=0):
        """Discriminator is updated every step independent of batch_size
        RNN and the generator
        """
        wrong_images = torch.cat((real_images[1:],
                                  real_images[0:1]), dim=0)
        wrong_prev = torch.cat((prev_image[1:],
                                prev_image[0:1]), dim=0)

        self.discriminator.zero_grad()
        real_images.requires_grad_()

        d_real, aux_real, _ = self.discriminator(real_images, condition,
                                                 prev_image)
        d_fake, aux_fake, _ = self.discriminator(fake_images, condition,
                                                 prev_image)
        d_wrong, _, _ = self.discriminator(wrong_images, condition,
                                           wrong_prev)

        d_loss, aux_loss = self._discriminator_masked_loss(d_real,
                                                           d_fake,
                                                           d_wrong,
                                                           aux_real,
                                                           aux_fake, objects,
                                                           aux_reg, mask)

        d_loss.backward(retain_graph=True)
        if gp_reg:
            reg = gp_reg * self._masked_gradient_penalty(d_real, real_images,
                                                         mask)
            reg.backward(retain_graph=True)

        grad_norm = _recurrent_gan.get_grad_norm(self.discriminator.parameters())
        self.discriminator_optimizer.step()

        d_loss_scalar = d_loss.item()
        d_real_np = d_real.cpu().data.numpy()
        d_fake_np = d_fake.cpu().data.numpy()
        aux_loss_scalar = aux_loss.item() if isinstance(aux_loss, torch.Tensor) else aux_loss
        grad_norm_scalar = grad_norm.item()
        del d_loss
        del d_real
        del d_fake
        del aux_loss
        del grad_norm
        gc.collect()

        return d_loss_scalar, d_real_np, d_fake_np, aux_loss_scalar, grad_norm_scalar

    def _optimize_generator(self, fake_images, prev_image, condition, objects, aux_reg,
                            mask, mu, logvar):
        self.generator.zero_grad()
        d_fake, aux_fake, _ = self.discriminator(fake_images, condition,
                                                 prev_image)
        g_loss = self._generator_masked_loss(d_fake, aux_fake, objects,
                                             aux_reg, mu, logvar, mask)

        g_loss.backward(retain_graph=True)
        gen_grad_norm = _recurrent_gan.get_grad_norm(self.generator.parameters())

        self.generator_optimizer.step()

        g_loss_scalar = g_loss.item()
        gen_grad_norm_scalar = gen_grad_norm.item()

        del g_loss
        del gen_grad_norm
        gc.collect()

        return g_loss_scalar, gen_grad_norm_scalar

    def _optimize_rnn(self):
        torch.nn.utils.clip_grad_norm_(self.rnn.parameters(), self.cfg.grad_clip)
        rnn_grad_norm = _recurrent_gan.get_grad_norm(self.rnn.parameters())
        self.rnn_optimizer.step()
        self.rnn.zero_grad()

        gru_grad_norm = None
        torch.nn.utils.clip_grad_norm_(self.sentence_encoder.parameters(), self.cfg.grad_clip)
        gru_grad_norm = _recurrent_gan.get_grad_norm(self.sentence_encoder.parameters())
        self.sentence_encoder_optimizer.step()
        self.sentence_encoder.zero_grad()

        ce_grad_norm = _recurrent_gan.get_grad_norm(self.condition_encoder.parameters())
        ie_grad_norm = _recurrent_gan.get_grad_norm(self.image_encoder.parameters())
        self.feature_encoders_optimizer.step()
        self.condition_encoder.zero_grad()
        self.image_encoder.zero_grad()
        return rnn_grad_norm, gru_grad_norm, ce_grad_norm, ie_grad_norm

    def _discriminator_masked_loss(self, d_real, d_fake, d_wrong, aux_real, aux_fake,
                                   objects, aux_reg, mask):
        """Accumulates losses only for sequences that have not ended
        to avoid back-propagation through padding"""
        d_loss = []
        aux_losses = []
        for b, ended in enumerate(mask):
            if not ended:
                sample_loss = self.criterion.discriminator(d_real[b], d_fake[b], d_wrong[b],
                                                           self.cfg.wrong_fake_ratio)
                if aux_reg > 0:
                    aux_loss = aux_reg * (self.aux_criterion(aux_real[b], objects[b]).mean() +
                                          self.aux_criterion(aux_fake[b], objects[b]).mean())
                    sample_loss += aux_loss
                    aux_losses.append(aux_loss)

                d_loss.append(sample_loss)

        d_loss = torch.stack(d_loss).mean()

        if len(aux_losses) > 0:
            aux_losses = torch.stack(aux_losses).mean()
        else:
            aux_losses = 0

        return d_loss, aux_losses

    def _generator_masked_loss(self, d_fake, aux_fake, objects, aux_reg,
                               mu, logvar, mask):
        """Accumulates losses only for sequences that have not ended
        to avoid back-propagation through padding"""
        g_loss = []
        for b, ended in enumerate(mask):
            if not ended:
                sample_loss = self.criterion.generator(d_fake[b])
                if aux_reg > 0:
                    aux_loss = aux_reg * self.aux_criterion(aux_fake[b], objects[b]).mean()
                else:
                    aux_loss = 0
                if mu is not None:
                    kl_loss = self.cfg.cond_kl_reg * kl_penalty(mu[b], logvar[b])
                else:
                    kl_loss = 0

                g_loss.append(sample_loss + aux_loss + kl_loss)

        g_loss = torch.stack(g_loss)
        return g_loss.mean()

    def _masked_gradient_penalty(self, d_real, real_images, mask):
        gp_reg = gradient_penalty(d_real, real_images).mean()
        return gp_reg

    # region Helpers
    def _plot_losses(self, visualizer, g_loss, d_loss, aux_loss,
                     iteration):
        _recurrent_gan._plot_losses(self, visualizer, g_loss, d_loss,
                                    aux_loss, iteration)

    def _plot_gradients(self, visualizer, rnn, gen, disc, gru, ce,
                        ie, iteration):
        _recurrent_gan._plot_gradients(self, visualizer, rnn, gen, disc,
                                       gru, ce, ie, iteration)

    def _draw_images(self, visualizer, real, fake, nrow):
        _recurrent_gan.draw_images(self, visualizer, real, fake, nrow)

    def _save(self, fake, path, epoch, iteration):
        _recurrent_gan._save(self, fake, path, epoch, iteration)

    def save_model(self, path, epoch, iteration):
        _recurrent_gan.save_model(self, path, epoch, iteration)

    def load_model(self, snapshot_path):
        _recurrent_gan.load_model(self, snapshot_path)
    # endregion

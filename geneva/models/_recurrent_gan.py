# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""Delegate for RecurrentGAN class"""
import os

import numpy as np
import torch
import torchvision


def load_model(state, snapshot_path):
    snapshot = torch.load(snapshot_path)
    state.generator.load_state_dict(
        snapshot['generator_state_dict'])
    state.discriminator.load_state_dict(
        snapshot['discriminator_state_dict'])
    state.rnn.load_state_dict(
        snapshot['rnn_state_dict'])
    state.layer_norm.load_state_dict(
        snapshot['layer_norm_state_dict'])
    state.image_encoder.load_state_dict(
        snapshot['image_encoder_state_dict'])
    state.condition_encoder.load_state_dict(
        snapshot['condition_encoder_state_dict'])
    state.generator_optimizer.load_state_dict(
        snapshot['generator_optimizer_state_dict'])
    state.discriminator_optimizer.load_state_dict(
        snapshot['discriminator_optimizer_state_dict'])
    state.rnn_optimizer.load_state_dict(
        snapshot['rnn_optimizer_state_dict'])
    state.feature_encoders_optimizer.load_state_dict(
        snapshot['feature_encoders_optimizer_state_dict'])
    state.sentence_encoder.load_state_dict(
        snapshot['sentence_encoder_state_dict'])
    state.sentence_encoder_optimizer.load_state_dict(
        snapshot['sentence_encoder_optimizer_state_dict'])


def save_model(state, path, epoch, iteration):
    if not os.path.exists(path):
        os.mkdir(path)

    snapshot = {
        'epoch': epoch,
        'iteration': iteration,
        'generator_state_dict': state.generator.state_dict(),
        'discriminator_state_dict': state.discriminator.state_dict(),
        'rnn_state_dict': state.rnn.state_dict(),
        'layer_norm_state_dict': state.layer_norm.state_dict(),
        'image_encoder_state_dict': state.image_encoder.state_dict(),
        'condition_encoder_state_dict': state.condition_encoder.state_dict(),
        'generator_optimizer_state_dict': state.generator_optimizer.state_dict(),
        'discriminator_optimizer_state_dict': state.discriminator_optimizer.state_dict(),
        'rnn_optimizer_state_dict': state.rnn_optimizer.state_dict(),
        'feature_encoders_optimizer_state_dict': state.feature_encoders_optimizer.state_dict(),
        'cfg': state.cfg,
    }

    snapshot['sentence_encoder_state_dict'] = \
        state.sentence_encoder.state_dict()
    snapshot['sentence_encoder_optimizer_state_dict'] = \
        state.sentence_encoder_optimizer.state_dict()
    torch.save(snapshot, '{}/snapshot_{}.pth'.format(path, iteration))


def _save(state, fake, path, epoch, iteration):
    if not os.path.exists(path):
            os.mkdir(path)

    torchvision.utils\
        .save_image(fake.cpu().data,
                    os.path.join(path, '{}_fake.jpg'.format(iteration)),
                    normalize=True,
                    range=(-1, 1))


def draw_images(state, visualizer, real, fake, nrow):
    visualizer.draw('Real Samples', np.array(real), nrow=nrow)
    visualizer.draw('Generated Samples', np.array(fake), nrow=nrow)


def _plot_gradients(state, visualizer, rnn, gen, disc, gru, ce, ie, iteration):
    visualizer.plot('ConditionEncoder Gradient Norm', 'train', iteration,
                    ce)
    visualizer.plot('ImageEncoder Gradient Norm', 'train', iteration,
                    ie)
    visualizer.plot('RNN Gradient Norm', 'train', iteration,
                    rnn)
    if gru:
        visualizer.plot('GRU Gradient Norm', 'train', iteration,
                        gru)

    visualizer.plot('Generator Gradient Norm', 'train', iteration,
                    gen)
    visualizer.plot('Discriminator Gradient Norm', 'train', iteration,
                    disc)


def _plot_losses(state, visualizer, g_loss, d_loss, aux_loss,
                 iteration):
    visualizer.plot('Discriminator output', 'train',
                    iteration, d_loss)
    visualizer.plot('Generator output', 'train', iteration,
                    g_loss)
    if state.cfg.aux_reg > 0:
        visualizer.plot('Aux Loss', 'train', iteration,
                        aux_loss)


def get_grad_norm(params):
    l2_norm = 0
    for param in params:
        if param.grad is not None:
            param_norm = param.grad.data.norm(2)
            l2_norm += param_norm ** 2

    return l2_norm ** (0.5)

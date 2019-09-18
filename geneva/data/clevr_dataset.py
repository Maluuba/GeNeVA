# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""PyTorch Dataset implementation for Iterative CLEVR dataset"""
import json

import cv2
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from geneva.utils.config import keys


class ICLEVERDataset(Dataset):
    def __init__(self, path, cfg, img_size=128):
        super().__init__()
        self.dataset = None
        self.dataset_path = path

        self.glove = _parse_glove(keys['glove_path'])
        with h5py.File(path, 'r') as f:
            self.keys = list(f.keys())
            self.background = f['background'].value
            self.background = cv2.resize(self.background, (128, 128))
            self.background = self.background.transpose(2, 0, 1)
            self.background = self.background / 128. - 1
            self.background += np.random.uniform(size=self.background.shape,
                                                 low=0, high=1. / 128)
            self.entities = np.array(json.loads(f['entities'].value))

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        if self.dataset is None:
            self.dataset = h5py.File(self.dataset_path, 'r')

        example = self.dataset[self.keys[idx]]

        scene_id = example['scene_id'].value
        images = example['images'].value
        text = example['text'].value
        objects = example['objects'].value

        images = images[..., ::-1]
        images = images / 128. - 1
        images += np.random.uniform(size=images.shape, low=0, high=1. / 128)
        images = images.transpose(0, 3, 1, 2)

        text = json.loads(text)

        turns_tokenized = [t.split() for t in text]
        lengths = [len(t) for t in turns_tokenized]
        turn_word_embeddings = np.zeros((len(text), max(lengths), 300))

        for i, turn in enumerate(turns_tokenized):
            for j, w in enumerate(turn):
                turn_word_embeddings[i, j] = self.glove[w]

        sample = {
            'scene_id': scene_id,
            'image': images,
            'turn': text,
            'objects': objects,
            'turn_word_embedding': turn_word_embeddings,
            'turn_lengths': lengths,
            'background': self.background,
            'entities': self.entities,
        }

        return sample


def _parse_glove(glove_path):
    glove = {}
    with open(glove_path, 'r') as f:
        for line in f:
            splitline = line.split()
            word = splitline[0]
            embedding = np.array([float(val) for val in splitline[1:]])
            glove[word] = embedding

    return glove


def collate_data(batch):
    batch = sorted(batch, key=lambda x: len(x['image']), reverse=True)
    dialog_lengths = list(map(lambda x: len(x['image']), batch))
    max_len = max(dialog_lengths)

    batch_size = len(batch)
    _, c, h, w = batch[0]['image'].shape

    batch_longest_turns = [max(b['turn_lengths']) for b in batch]
    longest_turn = max(batch_longest_turns)

    stacked_images = np.zeros((batch_size, max_len, c, h, w))
    stacked_turns = np.zeros((batch_size, max_len, longest_turn, 300))
    stacked_turn_lengths = np.zeros((batch_size, max_len))
    stacked_objects = np.zeros((batch_size, max_len, 24))
    turns_text = []
    scene_ids = []

    background = None
    for i, b in enumerate(batch):
        img = b['image']
        turns = b['turn']
        background = b['background']
        entities = b['entities']
        turns_word_embedding = b['turn_word_embedding']
        turns_lengths = b['turn_lengths']

        dialog_length = img.shape[0]
        stacked_images[i, :dialog_length] = img
        stacked_turn_lengths[i, :dialog_length] = np.array(turns_lengths)
        stacked_objects[i, :dialog_length] = b['objects']
        turns_text.append(turns)
        scene_ids.append(b['scene_id'])

        for j, turn in enumerate(turns_word_embedding):
            turn_len = turns_lengths[j]
            stacked_turns[i, j, :turn_len] = turn[:turn_len]

    sample = {
        'scene_id': np.array(scene_ids),
        'image': torch.FloatTensor(stacked_images),
        'turn': np.array(turns_text),
        'turn_word_embedding': torch.FloatTensor(stacked_turns),
        'turn_lengths': torch.LongTensor(stacked_turn_lengths),
        'dialog_length': torch.LongTensor(np.array(dialog_lengths)),
        'background': torch.FloatTensor(background),
        'entities': entities,
        'objects': torch.FloatTensor(stacked_objects),
    }

    return sample

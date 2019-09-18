# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""Datasets Manager"""
from geneva.data import clevr_dataset
from geneva.data import codraw_dataset


DATASETS = {
    'codraw': codraw_dataset.CoDrawDataset,
    'iclevr': clevr_dataset.ICLEVERDataset,
}

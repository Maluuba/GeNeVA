# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
from setuptools import setup

setup(
    name='GeNeVA',
    version='1.0',
    url='http://github.com/Maluuba/GeNeVA',
    author='Microsoft Research',
    description='Code to train and evaluate the GeNeVA-GAN model and the object detector and localizer for GeNeVA metrics',
    packages=['geneva'],
    extras_require=dict(
        dev=['pytest', 'pytest-flake8', 'flake8<3.6', 'flaky'],
    ),
)

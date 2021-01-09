#!/usr/bin/env python3
""" Source: https://github.com/deepfakes/faceswap
    Original Trainer """

from ._base import TrainerBase


class Trainer(TrainerBase):
    """ Original is currently identical to Base """
    def __init__(self, *args, **kwargs):  # pylint:disable=useless-super-delegation
        super().__init__(*args, **kwargs)

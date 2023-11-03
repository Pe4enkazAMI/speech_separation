import torchaudio.transforms as tat 
from torch import Tensor
import random
from hw_ss.augmentations.base import AugmentationBase


class TimeStretching(AugmentationBase):
    def __init__(self, p, stretch_factor, *args, **kwargs):
        self.p = p
        self.stretch_factor = stretch_factor 
        self._aug = tat.TimeStretch(stretch_factor=self.stretch_factor, fixed_rate=True, *args, **kwargs)

    def __call__(self, data: Tensor):
        if random.random() < self.p:
            x = data.unsqueeze(1)
            return self._aug(x).squeeze(1)
        else: 
            return data

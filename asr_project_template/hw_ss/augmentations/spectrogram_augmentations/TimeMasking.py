import torchaudio.transforms as tat 
from torch import Tensor
import random
from hw_ss.augmentations.base import AugmentationBase


class TimeMasking(AugmentationBase):
    def __init__(self, p, *args, **kwargs):
        self.p = p 
        self._aug = tat.TimeMasking(*args, time_mask_param=80, p=self.p, **kwargs)

    def __call__(self, data: Tensor):
        if random.random() < self.p:
            x = data.unsqueeze(1)
            return self._aug(x).squeeze(1)
        else: 
            return data

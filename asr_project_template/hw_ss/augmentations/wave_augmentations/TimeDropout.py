import torch_audiomentations
from torch import Tensor

from hw_ss.augmentations.base import AugmentationBase

class TimeDropout(AugmentationBase):
    def __init__(self, *args, **kwargs):
        self.ass = 0
    def __call__(self, x):
        len_ = x.shape[1]
        shift = torch.randint(low=0, high=len_)
        start = torch.randint(low=0, high = len_ - shift)        
        end = start + shift
        x[..., start:end, ...] = 0
        return x


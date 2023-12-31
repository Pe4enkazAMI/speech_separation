from typing import List

import torch
from torch import Tensor

from hw_ss.base.base_metric import BaseMetric
from hw_ss.metric.utils import calc_sisdr
from torchmetrics.audio import ScaleInvariantSignalDistortionRatio
    
class SISDRMetric(BaseMetric):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.sisdr = ScaleInvariantSignalDistortionRatio(zero_mean=True)
    def __call__(self, *args, **kwargs):
        return self.sisdr(kwargs["source_1"].detach().cpu(), kwargs["audio_target"].detach().cpu()).item()

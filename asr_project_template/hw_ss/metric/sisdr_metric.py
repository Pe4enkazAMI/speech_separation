from typing import List

import torch
from torch import Tensor

from hw_ss.base.base_metric import BaseMetric
from hw_ss.metric.utils import calc_sisdr
from torchmetrics.audio import ScaleInvariantSignalDistortionRatio
    
class SISDRMetric(BaseMetric):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.sisdr = ScaleInvariantSignalDistortionRatio(zero_mean=False).to("cuda:0")
    def __call__(self, *args, **kwargs):
        return self.sisdr(kwargs["source_1"], kwargs["audio_target"])

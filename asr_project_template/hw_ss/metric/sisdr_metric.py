from typing import List

import torch
from torch import Tensor

from hw_ss.base.base_metric import BaseMetric
from hw_ss.metric.utils import calc_sisdr

    
class SISDRMetric(BaseMetric):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    def __call__(self, *args, **kwargs):
        return calc_sisdr(kwargs["source_1"], kwargs["audio_target"], False).mean().item()

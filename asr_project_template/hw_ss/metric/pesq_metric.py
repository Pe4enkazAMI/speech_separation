from typing import List
from torch import Tensor
from hw_ss.base.base_metric import BaseMetric
from torchmetrics.audio.pesq import PerceptualEvaluationSpeechQuality

class PESQ(BaseMetric):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pesq = PerceptualEvaluationSpeechQuality(fs=kwargs["sample_rate"], mode=kwargs["mode"])

    def __call__(self, pred: Tensor, ground_truth: Tensor):
        ret_val = self.pesq(pred, ground_truth)
        return ret_val.item()




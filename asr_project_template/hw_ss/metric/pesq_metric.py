from typing import List
from torch import Tensor
from hw_ss.base.base_metric import BaseMetric
from torchmetrics.audio.pesq import PerceptualEvaluationSpeechQuality

class PESQ(BaseMetric):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pesq = PerceptualEvaluationSpeechQuality(fs=kwargs["sample_rate"], mode="wb")

    def __call__(self, pred: Tensor, ground_truth: Tensor):
        prd_ = pred.clone()
        prd_ = prd_.detach().cpu()
        gt = ground_truth.clone()
        gt = gt.detach().cpu()
        ret_val = self.pesq(prd_, gt)
        return ret_val.item()




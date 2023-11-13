import torch
from torch import Tensor
from torchmetrics.audio.pesq import PerceptualEvaluationSpeechQuality

from hw_ss.base.base_metric import BaseMetric


class PESQMetric(BaseMetric):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pesq = PerceptualEvaluationSpeechQuality(fs=16000, mode='wb')


    def __call__(self, *args, **kwargs):
        metric = self.pesq.to(kwargs["source_1"].device)
        pred_ = 20 * kwargs["source_1"] / kwargs["source_1"].norm(dim=-1, keepdim=True)
        return metric(pred_, kwargs["audio_target"]).mean().item()
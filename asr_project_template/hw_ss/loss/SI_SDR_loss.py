from typing import Any
import torch
import torch.nn as nn 
import torch.nn.functional as F


class SISDRLoss:
    def __init__(self, *args, **kwargs):
        self.ce = nn.CrossEntropyLoss(*args, **kwargs)
        self.alpha = kwargs["alpha"]
        self.beta = kwargs["beta"]
        self.gamma = kwargs["gamma"]

    def __call__(self, batch) -> Any:
        sisdr_1 = (1 - self.alpha - self.beta) * self.compute(batch["audio_target"], batch["source_1"])
        sisdr_2 = self.alpha * self.compute(batch["audio_target"], batch["source_2"])
        sisdr_3 = self.beta * self.compute(batch["audio_target"], batch["source_3"])
        L_sisdr = -(sisdr_1 + sisdr_2 + sisdr_3)
        L_ce = self.ce(batch["speaker_id"], batch["logits"])

        return L_sisdr + self.gamma * L_ce
    
    def compute(self, s, s_hat):
        pred_hat = s_hat - s_hat.mean()
        target_hat = s - s.mean()
        pred_t = torch.inner(pred_hat, target_hat) * target_hat / (target_hat ** 2).sum()
        target_t = pred_hat - pred_t
        
        ret = 20 * torch.log10(torch.norm(pred_t) / (torch.norm(target_t) + 1e-6))
        return ret
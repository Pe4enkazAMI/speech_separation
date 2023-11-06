from typing import Any
import torch
import torch.nn as nn 
import torch.nn.functional as F


class SISDRLoss(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.ce = nn.CrossEntropyLoss()
        self.alpha = kwargs["alpha"]
        self.beta = kwargs["beta"]
        self.gamma = kwargs["gamma"]

    def forward(self, batch) -> Any:
        sisdr_1 = (1 - self.alpha - self.beta) * self.compute(batch["source_1"], batch["audio_target"])

        sisdr_2 = self.alpha * self.compute(batch["source_2"], batch["audio_target"])
        sisdr_3 = self.beta * self.compute(batch["source_3"], batch["audio_target"])
        L_sisdr = -(sisdr_1 + sisdr_2 + sisdr_3)
        L_ce = self.ce(batch["logits"], batch["speaker_id"].long())

        return L_sisdr.mean() + self.gamma * L_ce
    
    # def compute(self, pred, target, zero_mean=True):
    #     eps = torch.finfo(pred.dtype).eps
    #     pred = pred.squeeze(1)
    #     target = target.squeeze(1)
    #     assert pred.shape == target.shape, "pred shapes should be the same as target shapes"

    #     if zero_mean:
    #         target = target - torch.mean(target, dim=-1, keepdim=True)
    #         pred = pred - torch.mean(pred, dim=-1, keepdim=True)

    #     alpha = (torch.sum(pred * target, dim=-1, keepdim=True) + eps)/(torch.sum(target**2, dim=-1, keepdim=True) + eps)
    #     target_scaled = alpha * target
    #     noise = target_scaled - pred 

    #     val = (torch.sum(target_scaled**2, dim=-1) + eps) / (torch.sum(noise**2, dim=-1) + eps)
    #     val = 10 * torch.log10(val)
    #     return val

    def compute(self, est, target, zero_mean=True):
        if zero_mean:
            target - torch.mean(target, dim=-1, keepdim=True)
            est - torch.mean(est, dim=-1, keepdim=True)
        alpha = (target * est).sum(dim=-1, keepdim=True) / torch.linalg.norm(target, dim=-1, keepdim=True)**2
        return 20 * torch.log10(torch.linalg.norm(alpha * target, dim=-1) / (torch.linalg.norm(alpha * target - est, dim=-1) + 1e-6) + 1e-6)
       
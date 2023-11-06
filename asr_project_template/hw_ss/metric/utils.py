import torch
import torch.nn as nn

# def calc_sisdr(pred, target, zero_mean=True):
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


def calc_sisdr(est, target, zero_mean=False):
        alpha = (target * est).sum(dim=-1, keepdim=True) / torch.linalg.norm(target, dim=-1, keepdim=True)**2
        return 20 * torch.log10(torch.linalg.norm(alpha * target, dim=-1) / (torch.linalg.norm(alpha * target - est, dim=-1) + 1e-6) + 1e-6)

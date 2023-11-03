import torch_audiomentations
from torch import Tensor

from hw_ss.augmentations.base import AugmentationBase


class ImpulseResponse(AugmentationBase):
    def __init__(self, *args, **kwargs):
        pass
    def __call__(self, data: Tensor, rir: Tensor):
        left_pad = right_pad = rir.shape[-1] - 1
        flipped_rir = rir.squeeze().flip(0)
        audio = F.pad(audio, [left_pad, right_pad]).view(1, 1, -1)
        convolved_audio = torch.conv1d(audio, flipped_rir.view(1, 1, -1)).squeeze()
        # peak normalization
        if convolved_audio.abs().max() > 1:
            convolved_audio /= convolved_audio.abs().max()
        return convolved_audio

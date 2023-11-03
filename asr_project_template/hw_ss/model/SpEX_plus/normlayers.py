import torch
import torch.nn as nn

class ChannelWiseLayerNorm(nn.Module):
    def __init__(self, shape, affine=True):
        super().__init__()
        self.norm = nn.LayerNorm(normalized_shape=shape,
                                 elementwise_affine=affine)
    
    def forward(self, x):
        # x: BxCxE -> BxCxE
        assert len(x.shape) == 3, "3D only"
        n_channels = x.shape[-2]
        x = x.transpose(1, 2)
        x = self.norm(x)
        x = x.transpose(1, 2)
        return x
    
class GlobalLayerNorm(nn.Module):
    def __init__(self, shape, affine=True, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.affine = affine
        self.shapes = shape
        if self.affine:
            self.gamma = nn.Parameter(torch.ones(self.shapes, 1))
            self.beta = nn.Parameter(torch.zeros(self.shapes, 1))

        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)
        
    def forward(self, x):
        mean = torch.mean(x, (1, 2), keepdim=True)
        var = torch.mean((x - mean)**2, (1, 2), keepdim=True)
        if self.affine:
            out = ((self.gamma * (x - mean)) / ((var + self.eps)**0.5)) + self.beta
        else:
            out = (x - mean) / ((var + self.eps)**0.5)
        return out

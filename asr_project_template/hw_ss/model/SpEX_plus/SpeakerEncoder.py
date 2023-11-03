import torch.nn as nn
import torch

from .ConvBlocks import ResBlock
from .normlayers import ChannelWiseLayerNorm, GlobalLayerNorm
import torch.nn.functional as F


class SpeakerEncoder(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.spk_enc = []
        
        self.num_res_net_blocks = kwargs["num_res_net_blocks"]

        for i in range(self.num_res_net_blocks):
            if i < self.num_res_net_blocks - 2:
                self.spk_enc.append(ResBlock(in_channels=kwargs["extractor_emb_dim"],
                                              out_channels=kwargs["extractor_emb_dim"]))
            elif i == self.num_res_net_blocks - 2:
                self.spk_enc.append(ResBlock(in_channels=kwargs["extractor_emb_dim"],
                                              out_channels=kwargs["extractor_intermed_dim"]))
            else: 
                self.spk_enc.append(ResBlock(in_channels=kwargs["extractor_intermed_dim"],
                                              out_channels=kwargs["extractor_intermed_dim"]))

        self.spk_enc.append(nn.Conv1d(in_channels=kwargs["extractor_intermed_dim"],
                                      out_channels=kwargs["spk_emb"], 
                                      kernel_size=1))

        self.spk_enc = nn.Sequential(*self.spk_enc)

    def forward(self, x):
        return self.spk_enc(x)


class SpeakerClassificationHead(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        self.linear = nn.Linear(in_features=kwargs["spk_emb"], out_features=kwargs["num_spk"])

    def forward(self, x):
        return self.linear(x)
    

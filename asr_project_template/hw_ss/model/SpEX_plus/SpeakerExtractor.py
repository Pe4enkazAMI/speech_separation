import torch.nn as nn
import torch 
from .ConvBlocks import StackedTCN

class SpeakerExtractor(nn.Module):
    def __init__(self, in_channels, out_channels, *args, **kwargs) -> None:
        super().__init__()
        self.num_stacked_tcn = kwargs["num_stacked_tcn"]
        self.extractor = dict()
        for i in range(1, self.num_stacked_tcn + 1):
            self.extractor[f"StackedTCN #{i}"] = StackedTCN(num_blocks=kwargs["num_tcn_blocks_in_stack"],
                                                            spk_emb=kwargs["spk_emb"],
                                                            in_channels=in_channels, 
                                                            out_channels=out_channels,
                                                            kernel_size=kwargs["tcn_kernel_size"], # Q
                                                            dilation=1)
            
        self.extractor = nn.ModuleDict(self.extractor)

    def forward(self, x, ref):
        for i in range(1, self.num_stacked_tcn + 1):
            x = self.extractor[f"StackedTCN #{i}"](x, ref)
        return x

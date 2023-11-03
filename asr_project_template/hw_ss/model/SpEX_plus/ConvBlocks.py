import torch
import torch.nn as nn
from .normlayers import ChannelWiseLayerNorm, GlobalLayerNorm
from .resconnect import ResidualConnection


class TCNblock(nn.Module):
    def __init__(self, 
                 in_channels,
                 conv_channels, 
                 kernel_size, 
                 dilation
                 ):
        
        super().__init__()
        self.padding = dilation * (kernel_size - 1) // 2
        block = nn.Sequential(
            nn.Conv1d(in_channels=in_channels,
                      out_channels=conv_channels, 
                      kernel_size=1),
            nn.PReLU(),
            GlobalLayerNorm(conv_channels),
            nn.Conv1d(in_channels=conv_channels, 
                      out_channels=conv_channels,
                      kernel_size=kernel_size,
                      groups=conv_channels, 
                      padding=self.padding,
                      dilation=dilation),
            nn.PReLU(),
            GlobalLayerNorm(conv_channels),
            nn.Conv1d(in_channels=conv_channels,
                      out_channels=in_channels,
                      kernel_size=1)
        )

        self.block = ResidualConnection(block)
    def forward(self, x):
        return self.block(x)
    


class FTCNBlock(nn.Module):
    def __init__(self, 
                 in_channels,
                 speaker_emb, 
                 conv_channels, 
                 kernel_size, 
                 dilation
                 ):
        super().__init__()
        self.padding = dilation * (kernel_size - 1) // 2
        self.block = nn.Sequential(
            nn.Conv1d(in_channels=in_channels + speaker_emb,
                      out_channels=conv_channels, 
                      kernel_size=1),
            nn.PReLU(),
            GlobalLayerNorm(conv_channels),
            nn.Conv1d(in_channels=conv_channels, 
                      out_channels=conv_channels,
                      kernel_size=kernel_size,
                      groups=conv_channels, 
                      padding=self.padding,
                      dilation=dilation),
            nn.PReLU(),
            GlobalLayerNorm(conv_channels),
            nn.Conv1d(in_channels=conv_channels,
                      out_channels=in_channels,
                      kernel_size=1)
        )

    def forward(self, x, spk_emb):
        shape = x.shape[-1]
        # x: B x C1 x Emb, spk_emb : B x C2 
        spk_emb = spk_emb.unsqueeze(-1).repeat(1, 1, shape) # spk_emb -> B x C2 x Emb
        inpt = torch.cat([x, spk_emb], dim=1) # -> B x C2 + C1 x Emb
        out = self.block(inpt)
        return out + x
    

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.fblock = nn.Sequential(
            nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=1),
            nn.BatchNorm1d(out_channels),
            nn.PReLU(),
            nn.Conv1d(in_channels=out_channels, out_channels=out_channels, kernel_size=1),
            nn.BatchNorm1d(out_channels),
        )
        if in_channels != out_channels:
            self.downsample = True
            self.down = nn.Conv1d(in_channels=in_channels, 
                                  out_channels=out_channels, 
                                  kernel_size=1, 
                                  bias=False)
        else:
            self.downsample = False

        self.sblock = nn.Sequential(
            nn.PReLU(),
            nn.MaxPool1d(3)
        )        
    def forward(self, x):
        out = self.fblock(x)
        if self.downsample:
            out = out + self.down(x)
        else:
            out = out + x
        return self.sblock(out)
    


        
class StackedTCN(nn.Module):
    def __init__(self, num_blocks, spk_emb, in_channels, out_channels, kernel_size, dilation):
        super().__init__()
        self.num_blocks = num_blocks
        block_list = dict()
        block_list["FTCN"] = FTCNBlock(in_channels=in_channels,
                                speaker_emb=spk_emb, 
                                conv_channels=out_channels, 
                                kernel_size=kernel_size, 
                                dilation=dilation)
        for i in range(1, num_blocks):
            block_list[f"TCN #{i}"] = TCNblock(in_channels=in_channels,
                                    conv_channels=out_channels,
                                    kernel_size=kernel_size,
                                    dilation=(2 ** i))

        self.STCN = nn.ModuleDict(block_list)

    def forward(self, x, spk):
        out = self.STCN["FTCN"](x, spk)
        for i in range(1, self.num_blocks):
            out = self.STCN[f"TCN #{i}"](out) 
        return out




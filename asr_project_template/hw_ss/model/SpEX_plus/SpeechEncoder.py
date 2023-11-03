import torch
import torch.nn as nn 

import torch.nn.functional as F



class SpeechEncoder(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

        self.l1 = kwargs["L1"]
        self.l2 = kwargs["L2"]
        self.l3 = kwargs["L3"]

        self.encoder_short = nn.Conv1d(in_channels=1, 
                                       out_channels=kwargs["speech_encoder_out_channels"],
                                       kernel_size=kwargs["L1"], 
                                       stride= kwargs["L1"] // 2)
        
        self.encoder_middle = nn.Conv1d(in_channels=1, 
                                                 out_channels=kwargs["speech_encoder_out_channels"], 
                                                 kernel_size=kwargs["L2"], 
                                                 stride= kwargs["L1"] // 2)
        
        self.encoder_long = nn.Conv1d(in_channels=1, 
                                               out_channels=kwargs["speech_encoder_out_channels"], 
                                               kernel_size=kwargs["L3"], 
                                               stride=kwargs["L1"] // 2)

    def forward(self, x):
        out_short = F.relu(self.encoder_short(x))
        last_dim_out = out_short.shape[-1]
        x_len1 = x.shape[-1]
        x_len2 = (last_dim_out - 1) * (self.l1 // 2) + self.l2
        x_len3 = (last_dim_out - 1) * (self.l1 // 2) + self.l3

        out_middle = F.relu(self.encoder_middle(F.pad(x, (0, x_len2 - x_len1), "constant", 0)))

        out_long = F.relu(self.encoder_long(F.pad(x, (0, x_len3 - x_len1), "constant", 0)))

        return out_short, out_middle, out_long

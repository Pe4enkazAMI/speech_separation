import torch.nn as nn
import torch

class SpeechDecoder(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        self.decoder_short = nn.ConvTranspose1d(in_channels=kwargs["speech_encoder_out_channels"], 
                                                out_channels=1,
                                                kernel_size=kwargs["L1"], 
                                                stride= kwargs["L1"] // 2)
        self.decoder_middle = nn.ConvTranspose1d(in_channels=kwargs["speech_encoder_out_channels"], 
                                                 out_channels=1, 
                                                 kernel_size=kwargs["L2"], 
                                                 stride= kwargs["L1"] // 2)
        
        self.decoder_long = nn.ConvTranspose1d(in_channels=kwargs["speech_encoder_out_channels"], 
                                               out_channels=1, 
                                               kernel_size=kwargs["L3"], 
                                               stride= kwargs["L1"] // 2)
        
    def forward(self, s1, s2, s3):
        return self.decoder_short(s1), self.decoder_middle(s2), self.decoder_long(s3)
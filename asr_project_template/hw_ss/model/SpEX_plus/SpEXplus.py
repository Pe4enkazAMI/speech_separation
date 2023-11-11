import torch
import torch.nn as nn
from .normlayers import ChannelWiseLayerNorm, GlobalLayerNorm
from .SpeakerEncoder import SpeakerEncoder, SpeakerClassificationHead
from .SpeechDecoder import SpeechDecoder
from .SpeechEncoder import SpeechEncoder
from .SpeakerExtractor import SpeakerExtractor
import torch.nn.functional as F
import numpy as np


class SpEXPlus(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()

        self.l1, self.l2, self.l3 = kwargs["L1"], kwargs["L2"], kwargs["L3"]

        # encoder for reference and mix
        self.speech_and_speaker_encoder = SpeechEncoder(*args, **kwargs)

    
        self.channel_norm = ChannelWiseLayerNorm(shape=3 * kwargs["speech_encoder_out_channels"], affine=True)

        self.conv1x1_extractor = nn.Conv1d(in_channels=3 * kwargs["speech_encoder_out_channels"],
                                           out_channels=kwargs["extractor_emb_dim"], # O 
                                           kernel_size=1)
        
        self.tcn_extractors = SpeakerExtractor(in_channels=kwargs["extractor_emb_dim"],
                                               out_channels=kwargs["extractor_intermed_dim"], # P
                                               **kwargs)
        
        self.mask_short = nn.Sequential(
                                    nn.Conv1d(in_channels=kwargs["extractor_emb_dim"],
                                              out_channels=kwargs["speech_encoder_out_channels"],
                                              kernel_size=1),
                                    nn.ReLU()
                                    )
        self.mask_middle = nn.Sequential(
                                    nn.Conv1d(in_channels=kwargs["extractor_emb_dim"],
                                              out_channels=kwargs["speech_encoder_out_channels"],
                                              kernel_size=1),
                                    nn.ReLU()
                                    )
        self.mask_long = nn.Sequential(
                                    nn.Conv1d(in_channels=kwargs["extractor_emb_dim"],
                                              out_channels=kwargs["speech_encoder_out_channels"],
                                              kernel_size=1),
                                    nn.ReLU()
                                    )
        
        self.speech_decoder = SpeechDecoder(*args, **kwargs)

        # RHS _____________________________________________________________________________________

        self.channel_norm_speaker = ChannelWiseLayerNorm(shape=3 * kwargs["speech_encoder_out_channels"],
                                                          affine=True)
        self.conv1x1_speaker = nn.Conv1d(in_channels= 3 * kwargs["speech_encoder_out_channels"],
                                         out_channels=kwargs["extractor_emb_dim"], # O 
                                         kernel_size=1)

        self.speaker_encoder = SpeakerEncoder(*args, **kwargs)

        self.speaker_logits = SpeakerClassificationHead(*args, **kwargs)

    def forward(self, x, ref_audio, true_len):
        
        out_short, out_middle, out_long = self.speech_and_speaker_encoder(x)
        out = self.channel_norm(torch.cat([out_short, out_middle, out_long], dim=1))
        out = self.conv1x1_extractor(out)
        

        ref_out_short, ref_out_middle, ref_out_long = self.speech_and_speaker_encoder(ref_audio)
        ref_out = self.channel_norm_speaker(torch.cat([ref_out_short, ref_out_middle, ref_out_long], dim=1))
        ref_out = self.conv1x1_speaker(ref_out)
        ref_out = self.speaker_encoder(ref_out)

        # mean pooling
        true_len = (true_len - self.l1) // (self.l1 // 2) + 1
        true_len = ((true_len // 3) // 3) // 3
        ref_out = torch.sum(ref_out, dim=-1) / true_len.to(ref_out.device).to(torch.float).unsqueeze(-1)
        
        # add to left
        res_out = out.clone()
        res_out = self.tcn_extractors(res_out, ref_out) + out

        mask1 = self.mask_short(res_out)
        mask2 = self.mask_middle(res_out)
        mask3 = self.mask_long(res_out)
        
        source_1, source_2, source_3 = out_short * mask1, out_middle * mask2, out_long * mask3
        
        dec1, dec2, dec3 = self.speech_decoder(source_1, source_2, source_3)
        
        speaker_logits = self.speaker_logits(ref_out)
        dec1 = F.pad(dec1, (0, x.shape[-1] - dec1.shape[-1]))
        return dec1, dec2[..., :dec1.shape[-1]], dec3[..., :dec1.shape[-1]], speaker_logits

    def __str__(self):
        """
        Model prints with number of trainable parameters
        """
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return super().__str__() + "\nTrainable parameters: {}".format(params)

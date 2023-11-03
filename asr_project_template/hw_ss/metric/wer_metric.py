from typing import List

import torch
from torch import Tensor

from hw_ss.base.base_metric import BaseMetric
from hw_ss.base.base_text_encoder import BaseTextEncoder
from hw_ss.metric.utils import calc_wer


class ArgmaxWERMetric(BaseMetric):
    def __init__(self, text_encoder: BaseTextEncoder, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.text_encoder = text_encoder

    def __call__(self, log_probs: Tensor, log_probs_length: Tensor, text: List[str], **kwargs):
        wers = []
        predictions = torch.argmax(log_probs.cpu(), dim=-1).numpy()
        lengths = log_probs_length.detach().numpy()
        for log_prob_vec, length, target_text in zip(predictions, lengths, text):
            target_text = BaseTextEncoder.normalize_text(target_text)
            if hasattr(self.text_encoder, "ctc_decode"):
                pred_text = self.text_encoder.ctc_decode(log_prob_vec[:length])
            else:
                pred_text = self.text_encoder.decode(log_prob_vec[:length])
            wers.append(calc_wer(target_text, pred_text))
        return sum(wers) / len(wers)
    
class BeamSearchWER(BaseMetric):
    def __init__(self, text_encoder: BaseTextEncoder, beam_size=15, lm=True,  *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.text_encoder = text_encoder
        self.beam_size = beam_size
        self.lm = lm

    def __call__(self, log_probs, log_probs_length, text, **kwargs):
        metric = []
        if self.lm: 
            log_probs = torch.nn.functional.log_softmax(log_probs.detach().cpu(), -1)
            log_probs_length = log_probs_length.detach().cpu()
            best_hypos = self.text_encoder.ctc_beam_search_with_lm(log_probs, log_probs_length,
                                                                  self.beam_size)
            for pred_text, target_text in zip(best_hypos, text):
                    metric.append(calc_wer(target_text, pred_text))
            return sum(metric) / len(metric)

    
        preds = log_probs.detach().cpu().numpy()
        lens = log_probs_length.detach().numpy()
        for probs, len_, target in zip(preds, lens, text):
            target = self.text_encoder.normalize_text(target)
            hypos = self.text_encoder.ctc_beam_search(probs[:len_], self.beam_size)
            pred_text = hypos[0].text

            metric.append(calc_wer(predicted_text=pred_text, target_text=target))
        return sum(metric) / len(metric)

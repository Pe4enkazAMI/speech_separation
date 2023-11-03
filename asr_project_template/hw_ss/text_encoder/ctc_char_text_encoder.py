from typing import List, NamedTuple

import torch
from collections import defaultdict
from scipy.special import softmax
from .char_text_encoder import CharTextEncoder
from hw_ss.utils import ROOT_PATH
from pyctcdecode import build_ctcdecoder
import multiprocessing
class Hypothesis(NamedTuple):
    text: str
    prob: float


class CTCCharTextEncoder(CharTextEncoder):
    EMPTY_TOK = "^"

    def __init__(self, alphabet: List[str] = None, lm = True):
        super().__init__(alphabet)
        self.vocab = [self.EMPTY_TOK] + list(self.alphabet)
        self.ind2char = dict(enumerate(self.vocab))
        self.char2ind = {v: k for k, v in self.ind2char.items()}

        self.lm_use = lm
        self.kenlm = "/kaggle/input/libri-lm/3-gram.arpa"
        if self.lm_use:
            self.ldecoder = self._create_lm_decoder()


    def _create_lm_decoder(self):

        with open("/kaggle/working/ASR/asr_project_template/hw_asr/text_encoder/librispeech-vocab.txt") as f:
            unigrams = [w.strip() for w in f.readlines()]
            decoder = build_ctcdecoder(
                [""] + [w.upper() for w in self.alphabet],
                kenlm_model_path=str(self.kenlm),
                unigrams=unigrams
            )
        return decoder

    def ctc_decode(self, inds: List[int]) -> str:
        last_char = self.EMPTY_TOK
        text = []
        for idx in inds:
            if last_char == self.ind2char[idx]:
                continue
            else:
                text.append(self.ind2char[idx])
            last_char = self.ind2char[idx]
        return ("".join(text)).replace(self.EMPTY_TOK, "")
    
    def _extend_beam(self, beam, prob):
        if len(beam) == 0:
            for i in range(len(prob)):
                last_char = self.ind2char[i]
                beam[('', last_char)] += prob[i]
            return beam
        
        new_beam = defaultdict(float)
        
        for (text, last_char), v in beam.items():
            for i in range(len(prob)):
                if self.ind2char[i] == last_char:
                    new_beam[(text, last_char)] += v * prob[i]
                else:
                    new_last_char = self.ind2char[i]
                    new_text = (text + last_char).replace(self.EMPTY_TOK, "")
                    new_beam[(new_text, new_last_char)] += v * prob[i]
        return new_beam
    def _cut_beam(self, beam, beam_size):
        return dict(sorted(beam.items(), key=lambda x: -x[1])[:beam_size])

# спиздил с сема
    def ctc_beam_search(self, probs: torch.tensor,
                        beam_size: int = 100) -> List[Hypothesis]:

        assert len(probs.shape) == 2
        char_length, voc_size = probs.shape
        assert voc_size == len(self.ind2char)

        beam = defaultdict(float)
        probs = softmax(probs, axis=1)

        for prob in probs:
            beam = self._extend_beam(beam, prob)
            beam = self._cut_beam(beam, beam_size)

        final_beam = defaultdict(float)

        for (text, last_char), v in beam.items():
            final_text = (text + last_char).replace(self.EMPTY_TOK, "")
            final_beam[final_text] += v

        sorted_beam = sorted(final_beam.items(), key=lambda x: -x[1])
        result = [Hypothesis(text, v) for text, v in sorted_beam]
        return result
    

    def ctc_beam_search_with_lm(self, probs: torch.tensor, lengths: torch.tensor,
            beam_size: int = 100) -> List[str]:

        probs = torch.nn.functional.log_softmax(probs, -1)

        logits_list = [probs[i][:lengths[i]].cpu().numpy() for i in range(lengths.shape[0])]

        with multiprocessing.get_context("fork").Pool() as pool:
            text_list = self.ldecoder.decode_batch(pool, logits_list, beam_width=beam_size)

        text_list = [elem.lower().replace(self.EMPTY_TOK, "") for elem in text_list]

        return text_list

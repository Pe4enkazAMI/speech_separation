import logging
from pathlib import Path

import torchaudio

from hw_ss.base.base_dataset import BaseDataset

logger = logging.getLogger(__name__)


class CustomAudioDataset(BaseDataset):
    def __init__(self, data, *args, **kwargs):
        index = data
        print(index[0])
        for entry in data:
            entry["audio_mix"] = str(Path(entry["audio_mix"]).absolute().resolve())
            entry["audio_ref"] = str(Path(entry["audio_ref"]).absolute().resolve())
            entry["audio_target"] = str(Path(entry["audio_target"]).absolute().resolve())
        super().__init__(index, *args, **kwargs)

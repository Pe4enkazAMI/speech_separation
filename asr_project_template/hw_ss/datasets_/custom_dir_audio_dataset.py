import logging
from pathlib import Path
from hw_ss.base.base_dataset import BaseDataset
from hw_ss.datasets_.custom_audio_dataset import CustomAudioDataset
from glob import glob
import os

logger = logging.getLogger(__name__)


def _path(files):
    return {os.path.basename(path).split("-")[0]: path for path in files}

class CustomDirAudioDataset(CustomAudioDataset):
    def __init__(self, mix_dir, ref_dir, target_dir, *args, **kwargs):
        data = []
        mixes = _path(glob(os.path.join(mix_dir, '*-mixed.wav')))
        refs = _path(glob(os.path.join(ref_dir, '*-ref.wav')))
        targets = _path(glob(os.path.join(target_dir, '*-target.wav')))

        for id in (mixes.keys() & refs.keys() & targets.keys()):
            data.append({
                "audio_mix": mixes[id],
                "audio_ref": refs[id],
                "audio_target": targets[id],
                "speaker_id": -1
            })
        super().__init__(data, *args, **kwargs)

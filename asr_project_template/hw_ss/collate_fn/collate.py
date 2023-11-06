import logging
from typing import List
import torch
logger = logging.getLogger(__name__)
from torch.nn.utils.rnn import pad_sequence

def collate_fn(dataset_items: List[dict]):
    """
    Collate and pad fields in dataset items
    """

    batch_size = len(dataset_items)
    
    audio_mix = [item["audio_mix"] for item in dataset_items]
    audio_ref = [item["audio_ref"] for item in dataset_items]
    audio_target = [item["audio_target"] for item in dataset_items]

    audio_len_mix = [item.shape[1] for item in audio_mix]
    audio_len_ref = [item.shape[1] for item in audio_ref]
    audio_len_target = [item.shape[1] for item in audio_target]
   
    audios_path_mix = [item["audio_path_mix"] for item in dataset_items]
    audios_path_ref = [item["audio_path_ref"] for item in dataset_items]
    audios_path_target = [item["audio_path_target"] for item in dataset_items]

    speaker_ids = [item["speaker_id"] for item in dataset_items]

    audios_ref = pad_sequence([item["audio_ref"].squeeze(0) for item in dataset_items], batch_first=True)
    audios_mix = pad_sequence([item["audio_mix"].squeeze(0) for item in dataset_items], batch_first=True)
    audios_target = pad_sequence([item["audio_target"].squeeze(0) for item in dataset_items], batch_first=True)
        
    result_batch = {
        "audio_mix": audios_mix.unsqueeze(1),
        "audio_ref": audios_ref.unsqueeze(1),
        "audio_target": audios_target.unsqueeze(1),
        "audio_ref_len": torch.Tensor(audio_len_ref),
        "audio_path_mix": audios_path_mix,
        "audios_path_ref": audios_path_ref,
        "audios_path_target": audios_path_target,
        "speaker_id": torch.Tensor(speaker_ids)
    }
    return result_batch
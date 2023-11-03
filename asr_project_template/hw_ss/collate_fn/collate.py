import logging
from typing import List
import torch
logger = logging.getLogger(__name__)


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

    max_audio_len = max([item["audio_mix"].shape[1] for item in dataset_items] + \
                        [item["audio_ref"].shape[1] for item in dataset_items])
    
    audios_mix = torch.zeros(size=(batch_size, max_audio_len))  
    audios_ref = torch.zeros(size=(batch_size, max_audio_len))  
    audios_target = torch.zeros(size=(batch_size, max_audio_len))  

    audios_path_mix = [item["audio_path_mix"] for item in dataset_items]
    audios_path_ref = [item["audio_path_ref"] for item in dataset_items]
    audios_path_target = [item["audio_path_target"] for item in dataset_items]

    speaker_ids = [item["speaker_id"] for item in dataset_items]

    for el in range(batch_size):
        audios_mix[el, ..., :audio_len_mix[el]] = dataset_items[el]["audio_mix"].squeeze(0) 
        audios_ref[el, ..., :audio_len_ref[el]] = dataset_items[el]["audio_ref"].squeeze(0) 
        if audio_len_target[el] > audio_len_mix[el]:
            tmp = dataset_items[el]["audio_target"].squeeze(0)[:audio_len_mix[el]]
            audios_target[el, ..., :audio_len_mix[el]] = tmp
        else:
            audios_target[el, ..., :audio_len_target[el]] = dataset_items[el]["audio_target"].squeeze(0) 

    result_batch = {
        "audio_mix": audios_mix,
        "audio_ref": audios_ref,
        "audio_target": audios_target,
        "audio_path_mix": audios_path_mix,
        "audios_path_ref": audios_path_ref,
        "audios_path_target": audios_path_target,
        "speaker_id": torch.Tensor(speaker_ids)
    }
    return result_batch
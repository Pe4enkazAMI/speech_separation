# import json
# import logging
# import os
# import shutil
# from pathlib import Path

# import torchaudio
# from speechbrain.utils.data_utils import download_file
# from tqdm import tqdm

# from hw_ss.base.base_dataset import BaseDataset
# from hw_ss.utils import ROOT_PATH

# logger = logging.getLogger(__name__)

# URL_LINKS = {
#     "dev-clean": "https://www.openslr.org/resources/12/dev-clean.tar.gz",
#     "dev-other": "https://www.openslr.org/resources/12/dev-other.tar.gz",
#     "test-clean": "https://www.openslr.org/resources/12/test-clean.tar.gz",
#     "test-other": "https://www.openslr.org/resources/12/test-other.tar.gz",
#     "train-clean-100": "https://www.openslr.org/resources/12/train-clean-100.tar.gz",
#     "train-clean-360": "https://www.openslr.org/resources/12/train-clean-360.tar.gz",
#     "train-other-500": "https://www.openslr.org/resources/12/train-other-500.tar.gz",
# }


# class LibrispeechDataset(BaseDataset):
#     def __init__(self, part, data_dir=None, *args, **kwargs):
#         assert part in URL_LINKS or part == 'train_all'

#         if data_dir is None:
#             data_dir = ROOT_PATH / "data" / "datasets" / "librispeech"
#             data_dir.mkdir(exist_ok=True, parents=True)
#         self._data_dir = Path(data_dir)
#         if part == 'train_all':
#             index = sum([self._get_or_load_index(part)
#                          for part in URL_LINKS if 'train' in part], [])
#         else:
#             index = self._get_or_load_index(part)

#         super().__init__(index, *args, **kwargs)

#     def _load_part(self, part):
#         arch_path = self._data_dir / f"{part}.tar.gz"
#         print(f"Loading part {part}")
#         download_file(URL_LINKS[part], arch_path)
#         shutil.unpack_archive(arch_path, self._data_dir)
#         for fpath in (self._data_dir / "LibriSpeech").iterdir():
#             shutil.move(str(fpath), str(self._data_dir / fpath.name))
#         os.remove(str(arch_path))
#         shutil.rmtree(str(self._data_dir / "LibriSpeech"))

#     def _get_or_load_index(self, part):
#         # index_path = self._data_dir / f"{part}_index.json"
#         index_path = Path(f"/kaggle/input/libri-index-full/{part}_index.json")
#         if index_path.exists():
#             with index_path.open() as f:
#                 index = json.load(f)
#         else:
#             index = self._create_index(part)
#             with index_path.open("w") as f:
#                 json.dump(index, f, indent=2)
#         return index

#     def _create_index(self, part):
#         index = []
#         split_dir = self._data_dir / part
#         print("SPLIT DIR:", split_dir)
#         if not split_dir.exists():
#             self._load_part(part)

#         flac_dirs = set()
#         for dirpath, dirnames, filenames in os.walk(str(split_dir)):
#             if any([f.endswith(".wav") for f in filenames]):
#                 flac_dirs.add(dirpath)
#         for flac_dir in tqdm(
#                 list(flac_dirs), desc=f"Preparing librispeech folders: {part}"
#         ):
#             txt_dir = Path('/'.join(flac_dir.split('/')[:-1] + ['meta'] + [flac_dir.split('/')[-1]]))
#             flac_dir = Path(flac_dir)
#             trans_paths = list(txt_dir.glob("*.trans.txt"))
#             for trans_path in trans_paths:
#                 with trans_path.open() as f:
#                     for line in f:
#                         f_id = line.split()[0]
#                         f_text = " ".join(line.split()[1:]).strip()
#                         flac_path = flac_dir / f"{f_id}.wav"
#                         t_info = torchaudio.info(str(flac_path))
#                         length = t_info.num_frames / t_info.sample_rate
#                         index.append(
#                             {
#                                 "path": str(flac_path.absolute().resolve()),
#                                 "text": f_text.lower(),
#                                 "audio_len": length,
#                             }
#                         )
#         return index



import json
import logging
import os
import shutil
from pathlib import Path
import glob

import torchaudio
from speechbrain.utils.data_utils import download_file
from tqdm import tqdm

from scripts.Mixer import MixtureGenerator, LibriSpeechSpeakerFiles

from hw_ss.base.base_dataset import BaseDataset
from hw_ss.utils import ROOT_PATH

logger = logging.getLogger(__name__)

URL_LINKS = {
    "dev-clean": "https://www.openslr.org/resources/12/dev-clean.tar.gz",
    "dev-other": "https://www.openslr.org/resources/12/dev-other.tar.gz",
    "test-clean": "https://www.openslr.org/resources/12/test-clean.tar.gz",
    "test-other": "https://www.openslr.org/resources/12/test-other.tar.gz",
    "train-clean-100": "https://www.openslr.org/resources/12/train-clean-100.tar.gz",
    "train-clean-360": "https://www.openslr.org/resources/12/train-clean-360.tar.gz",
    "train-other-500": "https://www.openslr.org/resources/12/train-other-500.tar.gz",
}


class LibrispeechDataset(BaseDataset):
    def __init__(self, part, data_dir=None, mixer=None, index_path=None, *args, **kwargs):
        # assert part in URL_LINKS or part == 'train_all'

        if data_dir is None:
            data_dir = ROOT_PATH / "data" / "datasets" / "librispeech"
            data_dir.mkdir(exist_ok=True, parents=True)

        self._data_dir = data_dir

        if index_path is not None:
            index_path = Path(index_path)
        else: 
            index_path = self._data_dir
        index = self._get_or_load_index(part, index_path, mixer)

        super().__init__(index, *args, **kwargs)

    def _load_part(self, part):
        arch_path = self._data_dir / f"{part}.tar.gz"
        print(f"Loading part {part}")
        download_file(URL_LINKS[part], arch_path)
        shutil.unpack_archive(arch_path, self._data_dir)
        for fpath in (self._data_dir / "LibriSpeech").iterdir():
            shutil.move(str(fpath), str(self._data_dir / fpath.name))
        os.remove(str(arch_path))
        shutil.rmtree(str(self._data_dir / "LibriSpeech"))

    def _get_or_load_index(self, part, index_path, mixer):
        
        index_path = index_path / f"{part}_mix_index.json"
        if index_path.exists():
            with index_path.open() as f:
                index = json.load(f)
        else:
            index = self._create_index(part, mixer)
            index_path = self._data_dir / f"{part}_mix_index.json"
            with index_path.open("w") as f:
                json.dump(index, f, indent=2)
        return index
    
    def _create_index(self, part, mixer):
        if self._data_dir is Path("None"):
            index = []
            split_dir = self._data_dir / part
            speaker_id = [f.name for f in os.scandir(split_dir)]
            spk_files = [LibriSpeechSpeakerFiles(name, split_dir, "*.flac") for name in speaker_id]
            mix_path = self._data_dir / f"{part}-mix" 
            mix_path.mkdir(exist_ok=True, parents=True)
            mixer_ = MixtureGenerator(speakers_files=spk_files, out_folder=mix_path, nfiles=mixer["nfiles"])
            mixer_.generate_mixes(
                **mixer
            )
        else:
            mix_path = Path(self._data_dir) / Path(part)

        
        ref = sorted(glob.glob(os.path.join(mix_path, "*-ref.wav")))
        target = sorted(glob.glob(os.path.join(mix_path, "*-target.wav")))
        mix = sorted(glob.glob(os.path.join(mix_path, "*-mixed.wav")))
        id_ = [int(r.split("/")[-1].split("_")[0]) for r in ref]

        setik = list(set(id_))

        mapid = {
            true_value: maped_id for maped_id, true_value in enumerate(setik)}

        for i in range(len(id_)):
            index += [
                {
                    "mix": mix[i],
                    "target": target[i],
                    "reference": ref[i],
                    "speaker_id": mapid[id_[i]],
                    "true_speaker_id": id_[i],
                    "audio_len": mixer["audioLen"],
                }
            ]
        return index






import os
from pathlib import Path

import soundfile as sf
import torch
from torch.utils.data import Dataset
import tqdm


class AiffDataset(Dataset):
    def __init__(self, paths: list[os.PathLike], train: bool = False):
        self.paths = paths
        self.train = train
        if self.train:
            self.labels = torch.tensor(
                [int(Path(path).stem[-1]) for path in self.paths], dtype=torch.int16
            )
        else:
            self.labels = None

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        data, sample_rate = sf.read(self.paths[idx], always_2d=True)
        waveform = torch.from_numpy(data.T).float()
        if self.train:
            return {"feature": waveform, "label": self.labels[idx]}
        return {
            "feature": waveform,
        }


if __name__ == "__main__":
    import time

    paths = list(Path("/home/kneidell/Documents/deep_voice/train2").glob("*.aif"))
    aiff_dataset = AiffDataset(paths, train=True)
    print(len(aiff_dataset))
    tic = time.time()
    for sample in tqdm.tqdm(aiff_dataset):
        pass
    print(f"iterated over train dataset in {time.time() - tic}s")

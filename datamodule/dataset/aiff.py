import os
from pathlib import Path

import soundfile as sf
import torch
from torch.utils.data import Dataset
import tqdm


class AiffDataset(Dataset):
    """
    Dataset for AIFF audio clips from the ICML 2013 Whale Challenge.

    Labels are extracted from filenames: the last character of the stem is the
    binary label (e.g. "train_0123_1.aif" → label 1). This follows the Kaggle
    competition naming convention and will break for differently-named files.

    Args:
        paths: List of paths to .aif files.
        train: If True, extract labels from filenames and include them in each
               returned sample. If False (inference mode), only "feature" is returned.
    """
    def __init__(self, paths: list[os.PathLike], train: bool = False):
        self.paths = paths
        self.train = train
        if self.train:
            self.labels = torch.tensor(
                [int(Path(path).stem[-1]) for path in self.paths], dtype=torch.float32
            )
        else:
            self.labels = None

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        data, sample_rate = sf.read(self.paths[idx], always_2d=True)
        waveform = torch.from_numpy(data.squeeze(1)).float()
        if self.train:
            return {"feature": waveform, "target": self.labels[idx]}
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

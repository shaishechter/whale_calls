import os
from pathlib import Path
import pandas as pd
import soundfile as sf
import lightning as L
from torch.utils.data import DataLoader

from datamodule.dataset.aiff import AiffDataset


class WhaleCallsDataModule(L.LightningDataModule):
    def __init__(
        self,
        base_path: os.PathLike,
        num_frames: int = 4000,
        samplerate: int = 2000,
        channels: int = 1,
        batch_size: int = 32,
        num_workers: int = 4,
    ):
        super().__init__()
        self.base_path = Path(base_path)
        self.num_frames = num_frames
        self.samplerate = samplerate
        self.channels = channels
        self.train_data_base_path = self.base_path / "train2"
        self.test_data_base_path = self.base_path / "test2"
        self.train_paths = self.get_valid_paths(self.train_data_base_path)
        self.test_paths = self.get_valid_paths(self.test_data_base_path)
        self.batch_size = batch_size
        self.num_workers = num_workers

    @staticmethod
    def get_aiff_metadata(aiff_path: os.PathLike):
        info = sf.info(aiff_path)
        return pd.Series(
            {
                "fname": str(aiff_path),
                "frames": info.frames,
                "channels": info.channels,
                "samplerate": info.samplerate,
                "duration": info.duration,
            }
        )

    def get_valid_paths(self, base_path):
        valid_paths = (
            pd.Series(list(self.train_data_base_path.glob("*.aif")))
            .apply(self.get_aiff_metadata)
            .loc[
                lambda df: df["frames"].eq(self.num_frames)
                & df["channels"].eq(self.channels)
                & df["samplerate"].eq(self.samplerate),
                "fname",
            ]
            .tolist()
        )
        return valid_paths

    def train_dataloader(self):
        return DataLoader(
            AiffDataset(paths=self.train_paths, train=True),
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            AiffDataset(paths=self.test_paths, train=False),
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )


if __name__ == "__main__":
    import time

    datamodule = WhaleCallsDataModule("/home/kneidell/Documents/deep_voice")
    train_dataloader = datamodule.train_dataloader()
    val_dataloader = datamodule.val_dataloader()

    tic = time.time()
    for batch in iter(train_dataloader):
        pass
    toc = time.time()

    print(f"iterated over train dataloader in {toc - tic}s")

    tic = time.time()
    for batch in iter(val_dataloader):
        pass
    toc = time.time()

    print(f"iterated over val dataloader in {toc - tic}s")

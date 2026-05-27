import os
from pathlib import Path
import pandas as pd
import sklearn.model_selection
import soundfile as sf
import lightning as L
from torch.utils.data import DataLoader

from datamodule.dataset.aiff import AiffDataset
import logging

logger = logging.getLogger(__name__)


class WhaleCallsDataModule(L.LightningDataModule):
    def __init__(
        self,
        base_path: os.PathLike,
        val_frac: float = 0.2,
        num_frames: int = 4000,
        samplerate: int = 2000,
        channels: int = 1,
        batch_size: int = 32,
        num_workers: int = 4,
    ):
        super().__init__()
        self.base_path = Path(base_path)
        if not self.base_path.exists():
            self.base_path = self.download_competition_data(self.base_path)
        self.train_paths = None
        self.val_paths = None
        self.test_paths = None
        self.num_frames = num_frames
        self.samplerate = samplerate
        self.channels = channels
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_frac = val_frac

    def setup(self, stage=None):
        unlabaled_data_paths = self.get_valid_paths(self.base_path / "train2")
        self.train_paths, self.val_paths = sklearn.model_selection.train_test_split(
            unlabaled_data_paths, test_size=self.val_frac
        )
        self.test_paths = self.get_valid_paths(self.base_path / "test2")

    @staticmethod
    def download_competition_data(path: os.PathLike):
        import kagglehub
        import zipfile

        if ("KAGGLE_USERNAME" not in os.environ) or (
            "KAGGLE_API_TOKEN" not in os.environ
        ):
            logger.info(
                "Kaggle credentials not found in environment variables. Using kagglehub to prompt for credentials."
            )
            kagglehub.login()  # will prompt for credentials if not set in env

        tmp_path = Path(
            kagglehub.competition_download(
                "the-icml-2013-whale-challenge-right-whale-redux",
            )
        )
        path.mkdir(exist_ok=True, parents=True)
        for zipfile_name in ["train2", "test2"]:
            with zipfile.ZipFile(
                (tmp_path / zipfile_name).with_suffix(".zip"), "r"
            ) as zip_ref:
                zip_ref.extractall(path)
        return path

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
            pd.Series(list(base_path.glob("*.aif")))
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
            AiffDataset(paths=self.val_paths, train=True),
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )

    def predict_dataloader(self):
        return DataLoader(
            AiffDataset(paths=self.test_paths, train=False),
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )


if __name__ == "__main__":
    import time
    import kagglehub

    datamodule = WhaleCallsDataModule("/home/kneidell/Documents/deep_voice")
    datamodule.setup()
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

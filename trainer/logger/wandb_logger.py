import datetime
import os
from pathlib import Path
from typing import Optional

import pandas as pd
from matplotlib import pyplot as plt
import wandb
from lightning.pytorch.loggers import WandbLogger
import logging

logger = logging.getLogger(__name__)
ENTITY = "kneidell"


class LocalWandbLogger(WandbLogger):
    """
    A custom Weights & Biases logger that saves the logged figures locally, so we can save space on the Weights & Biases account.
    """

    def __init__(
        self,
        project: str,
        local: bool,
        save_dir: str,
        mode: str,
        name: str,
        *args,
        **kwargs,
    ):
        """
        Initialize the logger.
        :param local: bool, whether to save the logs locally or upload them to Weights & Biases
        :param save_dir: str, the directory to save the logs to (and other reproducibility info, even if local=False)
        :param args: arguments to pass to the Wand
        """
        if mode not in ["debug", "async"]:
            raise ValueError(f"Invalid mode: {mode}, should be 'debug' or 'async'")
        self._offline = mode == "debug"

        self.logs_dir = save_dir
        self._name = name
        self._idx = self._get_exp_idx()
        self.local = local
        super().__init__(
            project=project,
            *args,
            offline=self._offline,
            name=f"{self._name}-{self._idx}",
            **kwargs,
        )

    def _get_exp_idx(self) -> int:
        import sqlite3

        db_path = Path(self.logs_dir) / "idx.db"
        with sqlite3.connect(str(db_path)) as conn:
            cursor = conn.cursor()
            cursor.execute("""CREATE TABLE IF NOT EXISTS run_idx 
                                (name TEXT PRIMARY KEY, idx INTEGER)""")

            cursor.execute(
                """
                    INSERT INTO run_idx (name, idx) 
                    VALUES (?, 1)
                    ON CONFLICT(name) 
                    DO UPDATE SET idx = idx + 1
                """,
                (self.name,),
            )

            idx = cursor.execute(
                "SELECT idx FROM run_idx WHERE name = ?", (self.name,)
            ).fetchone()[0]
            conn.commit()
        return idx

    @property
    def save_dir(self) -> Optional[str]:
        """
        Get the directory to save the logs to.
        :return:
        """
        return str(
            Path(self.logs_dir)
            / self.experiment.project
            / "/".join(self.experiment.name.split("-"))
        )

    @property
    def name(self) -> str:
        return self._name

    def log_figure(self, figure: plt.Figure, name: str, epoch: Optional[int] = None):
        """
        Log a figure. If local=True, save the figure locally, otherwise upload it to Neptune.
        :param figure: plt.Figure, the figure to log
        :param name: str, the name of the figure
        :param epoch: int | None, the epoch number
        :return:
        """
        if self.local:
            self.save_figure_to_local(figure, name, epoch)
        else:
            super().log({name: figure}, step=epoch)

    def save_figure_to_local(
        self, figure: plt.Figure, name: str, epoch: Optional[int] = None
    ) -> None:
        """
        Save the figure locally, and creating the necessary directories.
        :param figure: plt.Figure, the figure to save
        :param name: str, the name of the figure
        :param epoch: int | None, the epoch number
        :return: None
        """
        saving_path = Path(self.save_dir) / "training"
        saving_path.mkdir(exist_ok=True)
        os.chmod(saving_path, 0o777)

        subdirs = name.split("/")
        for subdir in subdirs:
            saving_path = saving_path / subdir
            saving_path.mkdir(exist_ok=True)
            os.chmod(saving_path, 0o777)

        if epoch is None:
            saving_path = saving_path.with_suffix(".png")
        else:
            saving_path = saving_path / f"{epoch}.png"

        figure.savefig(saving_path, dpi=400)

    def finalize(self, status: str) -> None:
        """
        Finalize the logger, and save the reproducibility info.
        :param status: str, the status of the training
        :return:
        """
        super().finalize(status)
        wandb.finish()

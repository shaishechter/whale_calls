import datetime
import os
from pathlib import Path
from typing import Optional

import pandas as pd
from matplotlib import pyplot as plt
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
        name: str,
        save_dir: Optional[os.PathLike] = None,
        offline: bool = False,
        *args,
        **kwargs,
    ):
        """
        Initialize the logger.
        :param local: bool, whether to save the logs locally or upload them to Weights & Biases
        :param save_dir: str, the directory to save the logs to (and other reproducibility info, even if local=False)
        :param offline: bool, whether to run in offline mode
        :param name: str, the name of the experiment
        :param args: arguments to pass to the Wand
        """

        self.logs_dir = save_dir
        self.local = local
        self.datetime = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        super().__init__(
            project=project,
            *args,
            offline=offline,
            name=name,
            **kwargs,
        )

    @property
    def save_dir(self) -> Optional[str]:
        """
        Get the directory to save the logs to.
        :return:
        """
        return str(
            Path(self.logs_dir)
            / self.experiment.project
            / self.experiment.name
            / f"{self.datetime}_{self.experiment.id}"
        )

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
        saving_path.mkdir(exist_ok=True, parents=True)
        os.chmod(saving_path, 0o777)

        subdirs = name.split("/")
        for subdir in subdirs:
            saving_path = saving_path / subdir
            saving_path.mkdir(exist_ok=True, parents=True)
            os.chmod(saving_path, 0o777)

        if epoch is None:
            saving_path = saving_path.with_suffix(".png")
        else:
            saving_path = saving_path / f"{epoch}.png"

        figure.savefig(saving_path, dpi=400)

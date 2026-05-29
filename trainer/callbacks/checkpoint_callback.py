from pathlib import Path
from lightning.pytorch.callbacks import ModelCheckpoint


class BinaryCheckpointCallback(ModelCheckpoint):
    """
    ModelCheckpoint that saves into a 'checkpoints' subdir of the logger's save_dir,
    so checkpoints live alongside figures under the same experiment directory.
    """

    def __init__(self, monitor="val/loss", mode="min", save_top_k=3, save_last=True, **kwargs):
        super().__init__(monitor=monitor, mode=mode, save_top_k=save_top_k, save_last=save_last, **kwargs)

    def on_train_start(self, trainer, pl_module):
        self.dirpath = str(Path(trainer.logger.save_dir) / "checkpoints")
        super().on_train_start(trainer, pl_module)

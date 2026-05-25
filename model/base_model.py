import functools
from pathlib import Path
from typing import Optional
import torch.optim
import torch.nn as nn
import lightning as L

import logging
import time

from model.aug.base import BaseAugmentation

logger = logging.getLogger(__file__)


class Model(L.LightningModule):
    """
    This class is a subclass of PyTorch Lightning's LightningModule, and it defines
    the basic structure of our audio model, mainly the training and validation steps. The model expects the following
    keys in the batch dictionary:
    - "feature": the input features
    - "target": the target labels
    The model also logs the following metrics:
    - "train/loss": the training loss
    - "val/loss": the validation loss
    - "train/forward_time": the time taken for the forward pass during training
    - "val/forward_time": the time taken for the forward pass during validation
    """

    def __init__(
        self,
        arch: nn.Module,
        optimizer: functools.partial[torch.optim.Optimizer],
        scheduler: functools.partial[torch.optim.lr_scheduler.LRScheduler],
        loss: nn.Module,
        augmentations: Optional[list[BaseAugmentation]],
        weights: Path | str | None = None,
        loss_reduction: str = "mean",
    ):
        """
        Initialize the model.
        :param arch: pl.LightningModule, the architecture of the model as a PyTorch Lightning module object
        :param optimizer: The optimizer to use for training, as a partial function that expects the model parameters
        :param scheduler: The learning rate scheduler to use, as a partial function that expects the optimizer
        :param loss: The loss function to use for training
        :param weights: Path | str | None, the path to the model weights to load (for fine-tuning or inference)
        :param loss_reduction: str, the reduction method for the vector output of the loss function (default: "mean")
        """
        super(Model, self).__init__()
        self._optimizer = optimizer
        self._scheduler = scheduler
        self.model = arch
        self.loss_fn = loss
        self.batch_size = None
        self.loss_reduction = loss_reduction

        self.augmentations = augmentations

        if weights is not None:
            logger.info(f"Loading weights from {weights}")
            self.load_state_dict(torch.load(weights)["state_dict"])

        self.save_hyperparameters(ignore=["arch", "optimizer", "scheduler", "loss"])

    def compute_loss(self, preds, batch):
        targets = batch["target"]
        loss_vec = self.loss_fn(
            preds, targets.squeeze(1)
        )  # We don't want to squeeze the batch dimension
        if self.loss_reduction == "mean":
            loss = torch.mean(loss_vec)
        elif self.loss_reduction == "sum":
            loss = torch.sum(loss_vec)
        else:
            raise ValueError("Reduction must be either 'mean' or 'sum'.")
        return loss, loss_vec

    def forward(self, batch):
        features = batch["feature"]
        preds = self.model(features)
        return preds

    def training_step(self, batch, batch_idx) -> dict[str, float]:
        # run predictions and time the forward pass
        step_start = time.time()
        batch_size = batch["target"].shape[0]
        with torch.no_grad():
            for aug in self.augmentations:
                if hasattr(aug, "on_train") and getattr(aug, "on_train"):
                    batch = aug(batch)
        aug_end = time.time()
        preds = self.forward(batch=batch)
        loss, loss_vec = self.compute_loss(preds, batch)
        step_end = time.time()
        self.log(
            "train/loss",
            loss.item(),
            on_step=True,
            on_epoch=True,
            batch_size=batch_size,
        )
        self.log(
            "train/forward_time",
            step_end - aug_end,
            on_step=True,
            on_epoch=True,
            batch_size=batch_size,
        )
        self.log(
            "train/aug_time",
            aug_end - step_start,
            on_step=True,
            on_epoch=True,
            batch_size=batch_size,
        )
        self.log(
            "train/step_time",
            step_end - step_start,
            on_step=True,
            on_epoch=True,
            batch_size=batch_size,
        )
        return {
            "loss": loss,
            "preds": preds,
            "batch": batch,
            "loss_vec": loss_vec,
        }

    def validation_step(self, batch, batch_idx) -> dict[str, float]:
        """

        :param batch: (dict) unreduced batch that includes all the keys,
            including strings
        :param batch_idx:
        :return:
        """
        batch_size = batch["target"].shape[0]
        step_start = time.time()
        with torch.no_grad():
            for aug in self.augmentations:
                if hasattr(aug, "on_val") and getattr(aug, "on_val"):
                    batch = aug(batch)
            aug_end = time.time()
            preds = self.forward(batch=batch)
            step_end = time.time()

            loss, loss_vec = self.compute_loss(preds, batch)
        self.log(
            "val/loss", loss.item(), on_step=True, on_epoch=True, batch_size=batch_size
        )
        self.log(
            "val/forward_time",
            step_end - aug_end,
            on_step=True,
            on_epoch=True,
            batch_size=batch_size,
        )
        self.log(
            "val/aug_time",
            aug_end - step_start,
            on_step=True,
            on_epoch=True,
            batch_size=batch_size,
        )
        return {
            "loss": loss,
            "preds": preds,
            "batch": batch,
            "loss_vec": loss_vec,
        }

    def predict_step(self, batch, batch_idx) -> dict[str, float]:
        with torch.no_grad():
            for aug in self.augmentations:
                if hasattr(aug, "on_predict") and getattr(aug, "on_predict"):
                    batch = aug(batch)
            preds = self.forward(batch=batch)
            loss, loss_vec = self.compute_loss(preds, batch)
        return {
            "loss": loss,
            "preds": preds,
            "batch": batch,
            "loss_vec": loss_vec,
        }

    def configure_optimizers(self) -> dict:
        optimizer = self._optimizer(params=self.parameters())
        if self._scheduler is not None:
            scheduler = self._scheduler(optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {"scheduler": scheduler, "monitor": "val_loss"},
            }
        return {"optimizer": optimizer, "lr_scheduler": None}


if __name__ == "__main__":
    import functools
    from torch.utils.data import DataLoader, Dataset

    IN_FEATURES, HIDDEN, OUT_FEATURES = 16, 32, 1
    N = 256

    class RandomDataset(Dataset):
        def __init__(self, n):
            self.X = torch.randn(n, IN_FEATURES)
            self.y = torch.randn(n, 1)

        def __len__(self):
            return len(self.X)

        def __getitem__(self, idx):
            return {"feature": self.X[idx], "target": self.y[idx]}

    arch = nn.Sequential(
        nn.Linear(IN_FEATURES, HIDDEN),
        nn.ReLU(),
        nn.Linear(HIDDEN, OUT_FEATURES),
    )

    dataset = RandomDataset(N)
    train_loader = DataLoader(dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(dataset, batch_size=32)

    model = Model(
        arch=arch,
        optimizer=functools.partial(torch.optim.SGD, lr=0.01, momentum=0.9),
        scheduler=functools.partial(
            torch.optim.lr_scheduler.StepLR, step_size=5, gamma=0.5
        ),
        loss=nn.MSELoss(reduction="none"),
        augmentations=[],
    )

    trainer = L.Trainer(max_epochs=10, accelerator="cpu", log_every_n_steps=5)
    trainer.fit(model, train_loader, val_loader)

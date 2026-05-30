import torch
import torch.nn.functional as F
from lightning.pytorch.callbacks import Callback


class BinaryAccuracyCallback(Callback):
    """
    Tracks sensitivity, precision, specificity, and mean loss for binary classification.
    Collects per-batch preds/targets/losses and logs scalars at epoch end for both train and val.
    """

    def __init__(self, threshold: float = 0.5, log_every_n_epochs: int = 1):
        super().__init__()
        self.threshold = threshold
        self.log_every_n_epochs = log_every_n_epochs
        self._train_preds = []
        self._train_targets = []
        self._train_losses = []
        self._val_preds = []
        self._val_targets = []
        self._val_losses = []

    def on_train_epoch_start(self, trainer, pl_module):
        self._train_preds.clear()
        self._train_targets.clear()
        self._train_losses.clear()

    def on_validation_epoch_start(self, trainer, pl_module):
        self._val_preds.clear()
        self._val_targets.clear()
        self._val_losses.clear()

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        self._train_preds.append(outputs["preds"].detach().cpu().float().flatten())
        self._train_targets.append(batch["target"].detach().cpu().float().flatten())
        self._train_losses.append(outputs["loss_vec"].detach().cpu().float().flatten())

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        self._val_preds.append(outputs["preds"].detach().cpu().float().flatten())
        self._val_targets.append(batch["target"].detach().cpu().float().flatten())
        self._val_losses.append(outputs["loss_vec"].detach().cpu().float().flatten())

    def on_train_epoch_end(self, trainer, pl_module):
        if (trainer.current_epoch + 1) % self.log_every_n_epochs != 0:
            return
        self._run_epoch_end("train", self._train_preds, self._train_targets, self._train_losses, pl_module)

    def on_validation_epoch_end(self, trainer, pl_module):
        if trainer.sanity_checking:
            return
        if (trainer.current_epoch + 1) % self.log_every_n_epochs != 0:
            return
        self._run_epoch_end("val", self._val_preds, self._val_targets, self._val_losses, pl_module)

    def _run_epoch_end(self, split, preds_buf, targets_buf, losses_buf, pl_module):
        preds = torch.cat(preds_buf)
        targets = torch.cat(targets_buf)
        losses = torch.cat(losses_buf)

        probs = torch.sigmoid(preds)
        predicted = (probs >= self.threshold).long()
        targets_int = targets.long()

        tp = ((predicted == 1) & (targets_int == 1)).sum().float()
        tn = ((predicted == 0) & (targets_int == 0)).sum().float()
        fp = ((predicted == 1) & (targets_int == 0)).sum().float()
        fn = ((predicted == 0) & (targets_int == 1)).sum().float()

        sensitivity = tp / (tp + fn + 1e-8)
        specificity = tn / (tn + fp + 1e-8)
        precision = tp / (tp + fp + 1e-8)
        loss_mean = losses.mean()

        pl_module.log(f"{split}/sensitivity", sensitivity.item(), on_epoch=True, prog_bar=True)
        pl_module.log(f"{split}/specificity", specificity.item(), on_epoch=True, prog_bar=False)
        pl_module.log(f"{split}/precision", precision.item(), on_epoch=True, prog_bar=False)
        pl_module.log(f"{split}/loss_mean", loss_mean.item(), on_epoch=True, prog_bar=False)

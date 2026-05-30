import numpy as np
import matplotlib.pyplot as plt
import torch
from lightning.pytorch.callbacks import Callback
from sklearn.metrics import (
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score,
    confusion_matrix,
)


class BinaryMetricPlotsCallback(Callback):
    """
    Plots ROC curve, confusion matrices (raw + normalized), precision-recall curve, and
    loss density at the end of each epoch for both train and val splits.
    Expects binary classification with BCEWithLogitsLoss (preds are raw logits).
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
        self._run_epoch_end("train", self._train_preds, self._train_targets, self._train_losses, trainer, pl_module)

    def on_validation_epoch_end(self, trainer, pl_module):
        if trainer.sanity_checking:
            return
        if (trainer.current_epoch + 1) % self.log_every_n_epochs != 0:
            return
        self._run_epoch_end("val", self._val_preds, self._val_targets, self._val_losses, trainer, pl_module)

    def _run_epoch_end(self, split, preds_buf, targets_buf, losses_buf, trainer, pl_module):
        logits = torch.cat(preds_buf).numpy()
        targets = torch.cat(targets_buf).numpy().astype(int)
        losses = torch.cat(losses_buf).numpy()
        probs = 1.0 / (1.0 + np.exp(-logits))
        preds = (probs >= self.threshold).astype(int)
        epoch = trainer.current_epoch
        log_fig = lambda fig, name: pl_module.logger.log_figure(fig, name, epoch)

        self._plot_roc(targets, probs, split, log_fig)
        self._plot_confusion_matrix(targets, preds, split, weighted=False, log_fig=log_fig)
        self._plot_confusion_matrix(targets, preds, split, weighted=True, log_fig=log_fig)
        self._plot_precision_recall(targets, probs, split, log_fig)
        self._plot_loss_density(targets, losses, split, log_fig)

    def _plot_roc(self, targets, probs, split, log_fig):
        fpr, tpr, _ = roc_curve(targets, probs)
        roc_auc = auc(fpr, tpr)
        fig, ax = plt.subplots()
        ax.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
        ax.plot([0, 1], [0, 1], "k--")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title(f"{split} ROC Curve")
        ax.legend()
        ax.grid(True)
        plt.tight_layout()
        log_fig(fig, f"{split}/roc_curve")
        plt.close(fig)

    def _plot_confusion_matrix(self, targets, preds, split, weighted, log_fig):
        mat = confusion_matrix(targets, preds, labels=[0, 1]).astype(float)
        display_mat = mat / (mat.sum(axis=1, keepdims=True) + 1e-8) if weighted else mat
        overall_acc = np.diag(mat).sum() / (mat.sum() + 1e-8)

        fig, ax = plt.subplots()
        im = ax.imshow(display_mat, cmap="Blues")
        plt.colorbar(im, ax=ax)
        for i in range(2):
            for j in range(2):
                val = display_mat[i, j]
                text = f"{val:.3f}" if weighted else f"{int(mat[i, j])}"
                ax.text(j, i, text, ha="center", va="center",
                        color="white" if val > display_mat.max() / 2 else "black")
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["Pred 0", "Pred 1"])
        ax.set_yticklabels(["True 0", "True 1"])
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        suffix = " (normalized)" if weighted else ""
        ax.set_title(f"{split} Confusion Matrix{suffix} — acc={overall_acc:.3f}")
        plt.tight_layout()
        name = f"{split}/confusion_matrix{'_normalized' if weighted else ''}"
        log_fig(fig, name)
        plt.close(fig)

    def _plot_precision_recall(self, targets, probs, split, log_fig):
        precision, recall, _ = precision_recall_curve(targets, probs)
        avg_precision = average_precision_score(targets, probs)
        fig, ax = plt.subplots()
        ax.plot(recall, precision, label=f"AP = {avg_precision:.3f}")
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title(f"{split} Precision-Recall Curve")
        ax.legend()
        ax.grid(True)
        plt.tight_layout()
        log_fig(fig, f"{split}/precision_recall_curve")
        plt.close(fig)

    def _plot_loss_density(self, targets, losses, split, log_fig):
        log_losses = np.log(np.clip(losses, 1e-20, None))
        pos_mask = targets == 1
        neg_mask = targets == 0
        fig, ax = plt.subplots()
        bins = np.linspace(log_losses.min(), log_losses.max(), 40)
        if neg_mask.any():
            ax.hist(log_losses[neg_mask], bins=bins, alpha=0.6, label="class 0 (neg)", density=True)
        if pos_mask.any():
            ax.hist(log_losses[pos_mask], bins=bins, alpha=0.6, label="class 1 (pos)", density=True)
        ax.set_xlabel("log(loss)")
        ax.set_ylabel("density")
        ax.set_title(f"{split} Loss Density by Class")
        ax.legend()
        ax.grid(True)
        plt.tight_layout()
        log_fig(fig, f"{split}/loss_density")
        plt.close(fig)

# Whale Call Detection

Binary audio classification for the [ICML 2013 Whale Challenge](https://www.kaggle.com/competitions/whale-detection-challenge) — detecting North Atlantic right whale upcalls in 2-second AIFF recordings.

Two model architectures are supported:

| Model | Input | Preprocessing |
|---|---|---|
| **ResNet18** | Spectrogram image | STFT → dB-normalized 2D image |
| **BEATs** | Raw waveform | Resample 2 kHz → 16 kHz |

Training is managed by [PyTorch Lightning](https://lightning.ai), configuration by [Hydra](https://hydra.cc), and experiment tracking by [Weights & Biases](https://wandb.ai).

---

## Repository Structure

```
whale_calls/
├── train.py                  # CLI training entry point
├── train_colab.ipynb         # Google Colab training notebook
├── requirements.txt
├── conf/                     # Hydra configuration
│   ├── config.yaml           # Root config (compose defaults)
│   ├── resnet18.yaml         # Pre-composed ResNet18 config
│   ├── beats.yaml            # Pre-composed BEATs config
│   ├── hparams/default.yaml  # Learning rate, epochs, scheduler
│   ├── datamodule/           # Data paths, batch size, splits
│   ├── model/                # Architecture, augmentation, loss, optimizer
│   └── trainer/              # Lightning Trainer, logger, callbacks
├── datamodule/               # WhaleCallsDataModule + AIFF dataset
├── model/                    # Arch (ResNet18, BEATs), augmentations, loss
└── trainer/                  # Callbacks (metrics, checkpoints) + W&B logger
```

---

## Setup

```bash
git clone https://github.com/shaishechter/whale_calls.git
cd whale_calls
pip install -r requirements.txt
```

**BEATs only:** the BEATs model depends on Microsoft's UniLM library, which must be cloned into the repo root:

```bash
git clone https://github.com/microsoft/unilm.git unilm
```

The BEATs checkpoint (`BEATs_iter3_plus_AS2M.pt`) is downloaded automatically from HuggingFace Hub on first run.

---

## Data

The dataset is the [Whale Detection Challenge](https://www.kaggle.com/competitions/whale-detection-challenge) from Kaggle (2000 Hz mono AIFF clips, binary labels encoded in filenames).

If the data directory does not exist, it is downloaded automatically via `kagglehub`. Set your Kaggle credentials beforehand:

```bash
export KAGGLE_USERNAME=your_username
export KAGGLE_KEY=your_api_key
```

The default data path is `/home/kneidell/data/whale_calls`. Override it with the `data_base_path` Hydra key (see below).

---

## Training

### CLI

The `model` key is required — choose `resnet18` or `beats`:

```bash
python train.py model=resnet18
python train.py model=beats
```

A W&B run name must also be provided:

```bash
python train.py model=resnet18 trainer.logger.name=my_run
```

**Common overrides:**

| Key | Default | Description |
|---|---|---|
| `hparams.lr` | `1e-3` | Learning rate |
| `hparams.max_epochs` | `50` | Number of epochs |
| `datamodule.batch_size` | `32` | Batch size |
| `datamodule.num_workers` | `4` | Dataloader workers |
| `data_base_path` | `/home/kneidell/data` | Root directory for raw data |
| `logs_base_path` | `/home/kneidell/Documents/logs` | Root directory for logs and checkpoints |
| `trainer.logger.name` | *(required)* | W&B run name |
| `trainer.logger.local` | `true` | Save figures locally instead of uploading to W&B |

**Examples:**

```bash
# ResNet18, custom LR and batch size
python train.py model=resnet18 trainer.logger.name=resnet_run \
  hparams.lr=1e-4 datamodule.batch_size=64

# BEATs, custom data path
python train.py model=beats trainer.logger.name=beats_run \
  data_base_path=/mnt/data

# Grid search over LR and model
python train.py --multirun model=resnet18,beats hparams.lr=1e-3,1e-4 \
  trainer.logger.name=sweep
```

---

### Google Colab

Open `train_colab.ipynb` in Colab (GPU runtime required).

**Before running:**

1. Add the following secrets via *Settings → Secrets* in Colab:
   - `KAGGLE_USERNAME` and `KAGGLE_KEY` — for dataset download
   - `WANDB_API_KEY` — for experiment tracking
2. Mount Google Drive when prompted (used to persist logs and checkpoints).

**Steps:**

1. Set `MODEL_ARCH` at the top of the notebook (`"resnet18"` or `"beats"`).
2. Run all cells. The notebook will:
   - Clone this repo and install dependencies
   - Clone the UniLM library (needed for BEATs)
   - Download the dataset via Kaggle
   - Compose the Hydra config and launch training

---

## Experiment Tracking

By default, logs and figures are saved locally under `{logs_base_path}/whale_calls/{run_name}/`. Each run produces:

- **Metrics**: loss, sensitivity, specificity, precision (logged per epoch)
- **Plots**: ROC curve, confusion matrix, precision-recall curve, loss distribution
- **Checkpoints**: top-3 models by validation loss, plus the last epoch (`{logs_base_path}/whale_calls/checkpoints/`)

Set `trainer.logger.local=false` to upload figures to the W&B cloud dashboard instead.

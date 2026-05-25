import sys
import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download
from pathlib import Path

# Requires BEATs source vendored from https://github.com/microsoft/unilm/tree/master/beats
#
# Setup steps:
#   1. Copy BEATs.py, backbone.py, and modules.py from the repo above into this project.
#   2. Instantiate BEATsClassifier with a checkpoint name, e.g. BEATsClassifier("iter3").
#      The checkpoint will be downloaded automatically to model/artifacts/ if not present.
#
# The pretrained model expects 16 kHz mono waveforms. If your data is at a different
# sample rate, apply ResampleAugmentation (model/aug/resample.py) before this model.

UNILM_PATH = Path(__file__).parent.parent.parent / "unilm"
BEATS_PATH = UNILM_PATH / "beats"
if not UNILM_PATH.exists():
    raise ImportError(
        "Microsoft unilm repo not found, please clone from https://github.com/microsoft/unilm.git"
    )

sys.path.insert(0, str(BEATS_PATH))
from BEATs import BEATs, BEATsConfig  # type: ignore[import]

ARTIFACTS_DIR = Path(__file__).parent.parent / "artifacts"

# Each entry: (repo_id, filename, repo_type)
MODEL_CHECKPOINTS: dict[str, tuple[str, str, str]] = {
    "iter3": ("Bencr/beats-checkpoints", "BEATs_iter3_plus_AS2M.pt", "dataset"),
}


def download_checkpoint(name: str, dest_dir: Path = ARTIFACTS_DIR) -> Path:
    if name not in MODEL_CHECKPOINTS:
        raise ValueError(
            f"Unknown checkpoint '{name}'. Available: {list(MODEL_CHECKPOINTS)}"
        )
    repo_id, filename, repo_type = MODEL_CHECKPOINTS[name]
    dest = dest_dir / filename
    if dest.exists():
        return dest
    dest_dir.mkdir(parents=True, exist_ok=True)
    return Path(
        hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            repo_type=repo_type,
            local_dir=str(dest_dir),
        )
    )


class BEATsClassifier(nn.Module):
    def __init__(self, checkpoint_name: str | None = None):
        super().__init__()

        if checkpoint_name is not None:
            checkpoint_path = download_checkpoint(checkpoint_name)
            checkpoint = torch.load(
                checkpoint_path, map_location="cpu", weights_only=False
            )
            cfg = BEATsConfig(checkpoint["cfg"])
            self.beats = BEATs(cfg)
            self.beats.load_state_dict(checkpoint["model"])
        else:
            cfg = BEATsConfig()
            cfg.input_patch_size = (
                16  # default from BEATs paper; -1 sentinel requires a checkpoint
            )
            self.beats = BEATs(cfg)

        self.classifier = nn.Linear(cfg.encoder_embed_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T) raw waveform at 16 kHz
        padding_mask = torch.zeros(x.shape, dtype=torch.bool, device=x.device)
        features, _ = self.beats.extract_features(x, padding_mask=padding_mask)
        # features: (B, seq_len, encoder_embed_dim) — mean-pool over time
        return self.classifier(features.mean(dim=1)).squeeze(1)  # (B,)


if __name__ == "__main__":
    # Random-init smoke test (no checkpoint needed):
    model = BEATsClassifier("iter3")
    x = torch.randn(4, 16_000)  # 4 clips × 1 s at 16 kHz
    out = model(x)
    print(f"input:  {tuple(x.shape)}")
    print(f"output: {tuple(out.shape)}")  # expect (4,)
    assert out.shape == (4,), f"unexpected output shape: {out.shape}"
    print("OK")

    # To run with pretrained weights (downloads automatically if not cached):
    #   model = BEATsClassifier("iter3")
    # The classifier head is randomly initialised even with pretrained weights —
    # fine-tune the full model on your labelled data before running inference.

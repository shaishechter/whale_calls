import torch
import torch.nn as nn

# Requires BEATs source vendored from https://github.com/microsoft/unilm/tree/master/beats
#
# Setup steps:
#   1. Copy BEATs.py, backbone.py, and modules.py from the repo above into this project.
#   2. Download a pretrained checkpoint, e.g. BEATs_iter3_plus_AS2M.pt:
#      https://valle.blob.core.windows.net/share/BEATs/BEATs_iter3_plus_AS2M.pt
#   3. Pass the checkpoint path when instantiating BEATsClassifier (see __main__ below).
#
# The pretrained model expects 16 kHz mono waveforms. If your data is at a different
# sample rate, apply ResampleAugmentation (model/aug/resample.py) before this model.
from beats import BEATs, BEATsConfig


class BEATsClassifier(nn.Module):
    def __init__(self, checkpoint_path: str | None = None):
        super().__init__()

        if checkpoint_path is not None:
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
            cfg = BEATsConfig(checkpoint["cfg"])
            self.beats = BEATs(cfg)
            self.beats.load_state_dict(checkpoint["model"])
        else:
            cfg = BEATsConfig()
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
    model = BEATsClassifier()
    x = torch.randn(4, 16_000)  # 4 clips × 1 s at 16 kHz
    out = model(x)
    print(f"input:  {tuple(x.shape)}")
    print(f"output: {tuple(out.shape)}")  # expect (4,)
    assert out.shape == (4,), f"unexpected output shape: {out.shape}"
    print("OK")

    # To run with pretrained weights:
    #   model = BEATsClassifier(checkpoint_path="path/to/BEATs_iter3_plus_AS2M.pt")
    # The classifier head is randomly initialised even with pretrained weights —
    # fine-tune the full model on your labelled data before running inference.

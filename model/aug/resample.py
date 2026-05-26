import torch
import torchaudio.functional as F

from model.aug.base import BaseAugmentation


class ResampleAugmentation(BaseAugmentation):
    def __init__(self, from_freq: int, to_freq: int, **kwargs):
        super().__init__(on_train=True, on_val=True, on_predict=True, **kwargs)
        self.from_freq = from_freq
        self.to_freq = to_freq

    def forward(self, batch: dict) -> dict:
        # x: (B, T) or (B, C, T)
        batch["feature"] = F.resample(batch["feature"], self.from_freq, self.to_freq)
        return batch


if __name__ == "__main__":
    import torch

    B, T = 4, 16_000  # 1 s at 16 kHz
    batch = {"feature": torch.randn(B, T), "target": torch.zeros(B)}

    aug = ResampleAugmentation(
        from_freq=16_000,
        to_freq=8_000,
    )
    out = aug(batch)
    expected_len = T * 8_000 // 16_000
    print(f"input length:  {T}")
    print(f'output shape: {out["feature"].shape}')
    print(f"output length: {out['feature'].shape[-1]}")  # expect 8000
    assert out["feature"].shape[-1] == expected_len
    print("OK")

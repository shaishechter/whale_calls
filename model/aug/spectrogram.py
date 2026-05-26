import numpy as np
import torch
import torchaudio.functional as T
from typing import Optional

from model.aug.base import BaseAugmentation


class SpectrogramAugmentation(BaseAugmentation):
    """
    Spectrogram augmentation using torchaudio. The augmentation handles complex and real input differently.
    Complex input is expected to be comprised of 2 real channels, representing the real and imaginary parts of the signal.
    Given complex input, the full spectrogram is computed and then split into positive and negative frequencies.
    Real input is expected to be a single channel, and computed using one-sided FFT.
    """

    def __init__(
        self, fs=500, nperseg=500, noverlap=250, nfft: Optional[int] = None, **kwargs
    ):
        super().__init__(on_train=True, on_val=True, on_predict=True, **kwargs)
        self.fs = fs
        self.nperseg = nperseg
        self.noverlap = noverlap
        self.nfft = nfft if nfft is not None else nperseg

    def forward(self, batch):
        feature = batch["feature"]
        specs = []
        if feature.dim() <= 2:
            is_cpx = False
        elif feature.dim() == 3:
            if feature.shape[1] == 1:
                feature = feature[:, 0, :]
                is_cpx = False
            elif feature.shape[1] == 2:
                feature = feature[:, 0, :] + 1j * feature[:, 1, :]
                is_cpx = True
            else:
                raise ValueError("Expected 1 or 2 channel input")
        spec = T.spectrogram(
            feature,
            pad=0,
            window=torch.hann_window(self.nperseg).to(feature.device),
            win_length=self.nperseg,
            power=2.0,
            normalized=True,
            center=False,
            n_fft=self.nfft,
            hop_length=self.nperseg - self.noverlap,
            onesided=not is_cpx,
        )
        if is_cpx:
            pos_spec = spec[:, : spec.shape[1] // 2 + 1, :]
            neg_spec = torch.flip(spec[:, spec.shape[1] // 2 - 1 :, :], dims=[1])
            specs.append(pos_spec)
            specs.append(neg_spec)
        else:
            specs.append(spec)
        feature = torch.stack(specs, dim=1)
        batch["feature"] = 10 * torch.log10(feature + 1e-10)
        return batch


if __name__ == "__main__":
    from torch.utils.data import default_collate

    # mock data for debugging/testing

    mock_cpx = (
        np.exp(100 * 1j * 2 * np.pi * np.arange(0, 60, 1 / 500))
        + np.exp(200 * 1j * 2 * np.pi * np.arange(0, 60, 1 / 500))
        + np.exp(-150 * 1j * 2 * np.pi * np.arange(0, 60, 1 / 500))
    )
    sample = {
        "feature": np.array([mock_cpx.real, mock_cpx.imag], dtype=np.float32),
    }
    batch = default_collate([sample])

    fs = 500
    nperseg = 500
    noverlap = 450

    ### run bandpass filter using torchaudio on cpx_sample

    spectrogram_aug = SpectrogramAugmentation(fs=fs, nperseg=nperseg, noverlap=noverlap)
    spectrogram_pytorch = spectrogram_aug(batch)["feature"].squeeze(0)

    import matplotlib.pyplot as plt
    import seaborn as sns

    fig, ax = plt.subplots(2, 1, figsize=(10, 10), sharey=False)
    sns.heatmap(spectrogram_pytorch[0], ax=ax[0])
    sns.heatmap(spectrogram_pytorch[1], ax=ax[1])

    ax[0].set_title("Positive freqs")
    ax[1].set_title("Negative freqs")

    plt.suptitle(
        "Spectrogram Example\n" r"$\exp(200\pi it)+\exp(400\pi it)+exp(-300\pi it)$"
    )
    plt.show()
    plt.close()

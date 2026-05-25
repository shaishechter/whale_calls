import torch
import torch.nn as nn
from torchvision.models import resnet18


class ResNet18(nn.Module):
    def __init__(self, pretrained: bool = False):
        super().__init__()
        weights = "IMAGENET1K_V1" if pretrained else None
        backbone = resnet18(weights=weights)

        # Replace first conv to accept single-channel input
        backbone.conv1 = nn.Conv2d(
            1, 64, kernel_size=7, stride=2, padding=3, bias=False
        )

        # Replace classifier head with a single logit output
        backbone.fc = nn.Linear(backbone.fc.in_features, 1)

        self.model = backbone

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x).squeeze(1)


if __name__ == "__main__":
    model = ResNet18()
    x = torch.randn(4, 1, 128, 128)  # batch of 4 single-channel spectrograms
    out = model(x)
    print(f"input:  {tuple(x.shape)}")
    print(f"output: {tuple(out.shape)}")  # expect (4,)
    assert out.shape == (4,), f"unexpected output shape: {out.shape}"
    print("OK")

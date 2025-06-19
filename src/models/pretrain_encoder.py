"""ResNet‑based encoder for SimCLR/BYOL."""

import torch.nn as nn
import torchvision.models as models


class ResNetEncoder(nn.Module):
    def __init__(self, name="resnet18", pretrained=False, projection_dim: int = 128):
        super().__init__()
        backbone = getattr(models, name)(weights=None if not pretrained else "IMAGENET1K_V1")
        self.features = nn.Sequential(*list(backbone.children())[:-1])  # remove FC
        self.projection_head = nn.Sequential(
            nn.Linear(backbone.fc.in_features, projection_dim),
            nn.ReLU(inplace=True),
            nn.Linear(projection_dim, projection_dim),
        )

    def forward(self, x):
        h = self.features(x).flatten(1)
        z = self.projection_head(h)
        return h, z

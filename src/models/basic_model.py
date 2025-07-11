import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet18_Weights


class BasicMLC(nn.Module):
    def __init__(self, num_classes):
        super(BasicMLC, self).__init__()
        self.resnet = models.resnet18(weights=ResNet18_Weights.DEFAULT)
        self.in_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(self.in_features, num_classes)
        self.pretrained = True

    def forward(self, x):
        x = self.resnet(x)
        return x

    @property
    def backbone(self):
        """Alias kept for backward-compatibility with training code."""
        return self.resnet

"""
PhotoTagNet –  a configurable CNN backbone + sigmoid head for multi-label tagging.
"""
from __future__ import annotations

import timm                    # falls back to torchvision if unavailable
import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet18_Weights

from src.config import ModelConfig, DEFAULT_CLASSES


class PhotoTagNet(nn.Module):
    """Backbone (ResNet-50 / EfficientNet-B0) with a linear multi-label head."""

    def __init__(self, cfg: ModelConfig, num_classes: int) -> None:
        super(PhotoTagNet, self).__init__()
        self.cfg = cfg
        self.num_classes = num_classes
        self.backbone, in_feats = self._build_backbone()
        self.classifier = nn.Linear(in_feats, num_classes)
        
    # ------------------------------------------------------------------ #
    #                     PRIVATE HELPERS                                
    def _build_backbone(self):
        """Create backbone and return (model_without_head, num_features)."""
        if self.cfg.backbone == "resnet50":
            model = models.resnet50(weights="IMAGENET1K_V2" if self.cfg.pretrained else None)
            in_feats = model.fc.in_features
            # strip head
            modules = list(model.children())[:-1]          # keep up to pool
            backbone = nn.Sequential(*modules, nn.Flatten(1))
        elif self.cfg.backbone == "efficientnet_b0":
            model = timm.create_model("efficientnet_b0",
                                      pretrained=self.cfg.pretrained,
                                      num_classes=0,
                                      global_pool="avg")
            in_feats = model.num_features
            backbone = model                # already returns pooled features
        elif self.cfg.backbone == "resnet18":
            model = models.resnet18(weights=ResNet18_Weights.DEFAULT if self.cfg.pretrained else None)
            in_feats = model.fc.in_features
            # strip head (fully connected layer (500 in & 1000 out))
            modules = list(model.children())[:-1]
            backbone = nn.Sequential(*modules, nn.Flatten(1))
        else:  # type: ignore[unreachable]
            raise ValueError(f"Unknown backbone: {self.cfg.backbone}")
        
        if self.cfg.freeze_backbone:
            for p in backbone.parameters():
                p.requires_grad = False
        return backbone, in_feats

    # ------------------------------------------------------------------ #
    #                          FORWARD                                   
    def forward(self, x):
        feats = self.backbone(x)
        logits = self.classifier(feats)
        return logits     # BCEWithLogitsLoss internally applies sigmoid

"""
PhotoTagNet – for multi-label tagging.
"""

from __future__ import annotations
import timm
import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet18_Weights, ResNet50_Weights
from src.config import ModelConfig, DEFAULT_CLASSES


class PhotoTagNet(nn.Module):
    def __init__(self, cfg: ModelConfig, num_classes: int) -> None:
        super(PhotoTagNet, self).__init__()
        self.cfg = cfg
        self.num_classes = num_classes

        self.backbone, in_feats = self._build_backbone()

        # Classifier head
        classifier_modules = []
        # Add dropout for regularization if specified in config and rate > 0
        # Helps prevent overfitting and improves generalization.
        if self.cfg.dropout_rate and self.cfg.dropout_rate > 0:
            classifier_modules.append(nn.Dropout(self.cfg.dropout_rate))
        classifier_modules.append(nn.Linear(in_feats, num_classes))
        # Modular classifier head for readability and future extensibility.
        self.classifier_head = nn.Sequential(*classifier_modules)

        self._init_head_weights()

    # ----------------------------- PRIVATE HELPERS -----------------------------
    def _build_backbone(self):
        """
        backbone : nn.Module producing a flat tensor
        feat_dim : int, size of that tensor
        """
        name = self.cfg.backbone.lower()

        # ---- ResNet18 ------------------------------------------------------
        if name == "resnet18":
            # • We grab official ImageNet weights when requested, else random.
            model = models.resnet18(
                weights=ResNet18_Weights.DEFAULT if self.cfg.pretrained else None
            )
            feat_dim = model.fc.in_features  # 512 for resnet18
            #   Chop off original classifier and flatten spatial dims.
            backbone = nn.Sequential(*list(model.children())[:-1], nn.Flatten(1))

        # ---- ResNet50 ------------------------------------------------------
        elif name == "resnet50":
            model = models.resnet50(
                weights=ResNet50_Weights.IMAGENET1K_V2 if self.cfg.pretrained else None
            )
            feat_dim = model.fc.in_features  # 2048 for resnet50
            backbone = nn.Sequential(*list(model.children())[:-1], nn.Flatten(1))

        # ---- EfficientNet-B0 via timm --------------------------------------
        elif name == "efficientnet_b0":
            if timm is None:
                raise ImportError(
                    "Install `timm` to use EfficientNet or other timm backbones."
                )
            # • In timm: num_classes=0 ⇒ no classifier layer is created.
            # • global_pool="avg"/"max" chooses how spatial dims are collapsed.
            model = timm.create_model(
                "efficientnet_b0",
                pretrained=self.cfg.pretrained,
                num_classes=0,
                global_pool=self.cfg.global_pool,
            )
            feat_dim = model.num_features
            backbone = model  # already returns flat features

        else:
            raise ValueError(f"Unsupported backbone '{self.cfg.backbone}'")

        # (optional) freeze everything except the head
        if self.cfg.freeze_backbone:
            for p in backbone.parameters():
                p.requires_grad_(False)

        return backbone, feat_dim

    # ----------------------------- TINY FORWARD -----------------------------
    def forward(self, x: torch.Tensor):
        feats = self.backbone(x)
        logits = self.classifier_head(feats)  # Use the new classifier_head
        return (
            logits  # will be passed to BCEWithLogitsLoss -> internally applies sigmoid
        )

    # ----------------------------- WEIGHT INITIALIZATION -----------------------------
    def _init_head_weights(self) -> None:
        """
        Use Kaiming normal (a.k.a. He) is default for linear layers
        feeding ReLU, but still a sensible starting point for logits.
        """
        for m in self.classifier_head.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="linear")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def trainable_parameters(self):
        """
        Gett only params that will update
        Convenience for `optim.Adam(model.trainable_parameters(), …)`.
        """
        return (p for p in self.parameters() if p.requires_grad)

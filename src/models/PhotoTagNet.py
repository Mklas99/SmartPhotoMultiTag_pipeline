"""
PhotoTagNet – for multi-label tagging.
"""
from __future__ import annotations
import timm
import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet18_Weights
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
        if hasattr(cfg, 'dropout_rate') and isinstance(cfg.dropout_rate, float) and cfg.dropout_rate > 0:
            classifier_modules.append(nn.Dropout(cfg.dropout_rate))
        classifier_modules.append(nn.Linear(in_feats, num_classes))
        # Modular classifier head for readability and future extensibility.
        self.classifier_head = nn.Sequential(*classifier_modules)
        
    # ----------------------------- PRIVATE HELPERS -----------------------------                              
    def _build_backbone(self):
        """Create backbone and return (model_without_head, num_features)."""
        if self.cfg.backbone == "resnet50":
            # Load ResNet50 model. Uses "IMAGENET1K_V2" pretrained weights if self.cfg.pretrained is True, otherwise no pretrained weights.
            model = models.resnet50(weights="IMAGENET1K_V2" if self.cfg.pretrained else None)
            # Get the number of input features to the original fully connected (fc) layer.
            in_feats = model.fc.in_features
            # Remove the original classification head (the last layer, which is model.fc).
            modules = list(model.children())[:-1]
            # New sequential model consisting of all layers except the original head, Flatten layer to ensure the output is 1D for the classifier.
            backbone = nn.Sequential(*modules, nn.Flatten(1))
            
        elif self.cfg.backbone == "efficientnet_b0":
            # Create EfficientNet-B0 model, load pretrained weights if self.cfg.pretrained is True, otherwise no pretrained weights.
            # num_classes=0 ensures the model returns features before the classifier.
            # global_pool="avg" applies global average pooling to the feature map.
            model = timm.create_model("efficientnet_b0",
                                      pretrained=self.cfg.pretrained,
                                      num_classes=0, 
                                      global_pool="avg")
            # Get the number of output features from the EfficientNet model.
            in_feats = model.num_features
            # The timm model with num_classes=0 and global_pool="avg" already serves as the backbone.
            backbone = model
            
        elif self.cfg.backbone == "resnet18":
            # Load the ResNet18 model. Use default ResNet18_Weights if pretrained is set
            model = models.resnet18(weights=ResNet18_Weights.DEFAULT if self.cfg.pretrained else None)
            # Get the number of input features to the original fully connected (fc) layer.
            in_feats = model.fc.in_features
            # Remove the original classification head.
            modules = list(model.children())[:-1]
            # Create a new sequential model with ResNet18 layers (excluding head) and a Flatten layer.
            backbone = nn.Sequential(*modules, nn.Flatten(1))
        else:
            raise ValueError(f"Unknown backbone: {self.cfg.backbone}")
        
        # If config specifies to freeze the backbone's weights
        if self.cfg.freeze_backbone:
            # Set all parameters´s (weights and biases) requires_grad to False to prevent these parameters from being updated during training.
            for p in backbone.parameters():
                # Set requires_grad to False to prevent these parameters from being updated during training.
                p.requires_grad = False

        # Return the constructed backbone model and the number of its output features (in_feats).
        return backbone, in_feats

    # ----------------------------- FORWARD -----------------------------
    def forward(self, x):
        feats = self.backbone(x)
        logits = self.classifier_head(feats) # Use the new classifier_head
        return logits     # BCEWithLogitsLoss internally applies sigmoid

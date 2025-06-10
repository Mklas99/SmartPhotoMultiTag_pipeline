import timm
import torch.nn as nn
from src.config import ModelConfig, DEFAULT_CLASSES

def build_model(cfg: ModelConfig) -> nn.Module:
    model = timm.create_model(
        cfg.backbone,
        pretrained=cfg.pretrained,
        num_classes=len(DEFAULT_CLASSES),
        drop_rate=cfg.drop_rate,
    )
    return model

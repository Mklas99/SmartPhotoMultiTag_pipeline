from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple  # Import Optional

from torchvision import transforms as T

# -------- Project Root --------
PROJECT_ROOT = Path(__file__).resolve().parent.parent

IMAGE_CNT:   int = 10000
IMAGE_SIZE:  int = 224                   # standard size for ResNet/EfficientNet
BATCH_SIZE:  int = 16                    # smaller ⇒ noisier grads ⇒ more regularisation
NUM_WORKERS: int = 6

DEFAULT_CLASSES: List[str] = "person", "dog", "car", "bicycle", "cell phone", "laptop"

DATASET_ROOT = PROJECT_ROOT / "src" / "data" / "coco"
DATASET_DIR = PROJECT_ROOT / "src" / "data" / "coco"
META_PATH = PROJECT_ROOT / "src" / "data" / "coco" / "dataset_metadata.json"  # Metadata summary
NOTEBOOK_PATH = PROJECT_ROOT / "notebooks"  # Jupytext file
HTML_REPORT_PATH = NOTEBOOK_PATH.with_suffix(".html")  # nbconvert output

# Where to store checkpoints
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
CHECKPOINT_DIR.mkdir(exist_ok=True, parents=True)

# Directory for results like plots and metrics
RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True, parents=True)

# ImageNet normalization (mean, std for R,G,B channels)
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

# ------------------- Training pipeline -------------------
train_transforms = T.Compose([
    # Spatial ------------------------------------------------
    T.RandomResizedCrop(
        IMAGE_SIZE,                       # 224 after the config tweak
        scale=(0.6, 1.0),                 # was (0.8, 1.0) → more aggressive cropping
        ratio=(0.75, 1.33)                # keep “normal” aspect ratios
    ),
    T.RandomHorizontalFlip(p=0.5),
    # 50 % of the time, apply one geometric distortion
    T.RandomApply([
        T.RandomRotation(degrees=15),
        T.RandomPerspective(distortion_scale=0.2, p=1.0)
    ], p=0.5),

    # Photometric -------------------------------------------
    # 80 % chance to jitter colour; magnitude slightly higher than before
    T.RandomApply([
        T.ColorJitter(brightness=0.4,
                      contrast=0.4,
                      saturation=0.4,
                      hue=0.1)
    ], p=0.8),
    T.RandomGrayscale(p=0.1),

    # Tensor & pixel-level ----------------------------------
    T.ToTensor(),
    # RandomErasing after ToTensor ⇒ makes the model robust to occlusions
    T.RandomErasing(p=0.25,
                    scale=(0.02, 0.15),
                    ratio=(0.3, 3.3),
                    value='random'),
    T.Normalize(mean=MEAN, std=STD),
])

# ------------------- Validation / test pipeline -------------------
val_transforms = T.Compose([
    T.Resize(IMAGE_SIZE + 32),     # 256 → keeps comparison fair
    T.CenterCrop(IMAGE_SIZE),      # 224
    T.ToTensor(),
    T.Normalize(mean=MEAN, std=STD),
])

# -------- Config dataclasses --------
@dataclass
class ModelConfig:
    backbone: Literal["resnet18", "resnet50", "efficientnet_b0"] = "resnet18"
    pretrained: bool = True               # keep ImageNet weights
    freeze_backbone: bool = True          # <-- freeze for now; unfreeze later if needed
    dropout_rate: float = 0.3             # ↑ stronger regularisation
    staged_unfreeze_layer: str = "layer4" # which layer to unfreeze at stage 2
    staged_unfreeze_epoch: int = 4       # epoch to unfreeze last block
    full_unfreeze_epoch: int = 5         # epoch to unfreeze all


@dataclass
class OptimConfig:
    optim: Literal["adamw", "sgd"] = "adamw"
    lr: float = 1e-4                      # a hair higher because only the head trains
    weight_decay: float = 5e-4            # a bit stronger than before
    betas: Tuple[float, float] = (0.9, 0.999)
    scheduler: Literal["none", "step", "plateau", "cosine", "cosine_warm_restart"] = "cosine_warm_restart"
    patience: int = 2                     # plateau patience (epochs)
    gamma: float = 0.5                    # LR will be halved on plateau
    step_size: int = 1                    # ignored by ReduceLROnPlateau
    momentum: float = 0.9                 # (only for SGD)
    backbone_lr: float = 1e-5             # for staged fine-tuning
    head_lr: float = 1e-4                 # for staged fine-tuning
    cosine_T_0: int = 5                   # CosineAnnealingWarmRestarts param
    cosine_T_mult: int = 2                # CosineAnnealingWarmRestarts param


@dataclass
class TrainConfig:
    epochs: int = 30                      # let it run, but…
    early_stop_patience: int = 5         # …stop soon after val loss stops improving
    seed: int = 42
    precision_at_k: int = 3
    use_mixup: bool = True               # enable MixUp
    mixup_alpha: float = 0.4              # MixUp alpha
    label_smoothing: float = 0.1          # label smoothing for BCEWithLogitsLoss


def get_num_classes(custom_classes: Optional[List[str]] | None = None) -> int:
    """
    Returns number of label classes, preferring *custom_classes* if given.
    """
    return len(custom_classes or DEFAULT_CLASSES)


# Convenience for MLflow logging
def as_flat_dict() -> Dict[str, float | int | str]:
    """Return every config field flattened into one dict."""
    flat: Dict[str, float | int | str] = {
        "image_size": IMAGE_SIZE,
        "batch_size": BATCH_SIZE,
        "num_workers": NUM_WORKERS,
    }
    flat.update({f"model_{k}": v for k, v in asdict(ModelConfig()).items()})
    flat.update({f"optim_{k}": v for k, v in asdict(OptimConfig()).items()})
    flat.update({f"train_{k}": v for k, v in asdict(TrainConfig()).items()})
    return flat

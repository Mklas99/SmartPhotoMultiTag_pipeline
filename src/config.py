from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Literal, Optional  # Import Optional
from torchvision import transforms as T

# -------- Project Root --------
# Assuming this file (config.py) is in /src, the project root is its parent's parent
PROJECT_ROOT = Path(__file__).resolve().parent.parent

IMAGE_CNT: int = 10
# -------- Core hyper‑parameters --------
IMAGE_SIZE: int = 100
BATCH_SIZE: int = 5
NUM_WORKERS: int = 4

DEFAULT_CLASSES: List[str] = [
    "person", "dog"]
#    "cat", "car", "bus", "bicycle",
#    "pizza", "apple", "cell phone", "laptop"
#]

DATASET_ROOT = PROJECT_ROOT / "src" / "data" / "coco"
DATASET_DIR = PROJECT_ROOT / "src" / "data" / "coco"
META_PATH = PROJECT_ROOT / "src" / "data" / "coco" / "dataset_metadata.json"     # Metadata summary
NOTEBOOK_PATH = PROJECT_ROOT / "notebooks" # Jupytext file
HTML_REPORT_PATH = NOTEBOOK_PATH.with_suffix(".html")           # nbconvert output

# Where to store checkpoints
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
CHECKPOINT_DIR.mkdir(exist_ok=True, parents=True)

# Directory for results like plots and metrics
RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True, parents=True)

# ImageNet normalization (mean, std for R,G,B channels)
MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]

train_transforms = T.Compose([
    T.RandomResizedCrop(IMAGE_SIZE, scale=(0.8, 5.0)),
    T.RandomHorizontalFlip(),
    T.RandomRotation(10),  
    T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    T.ToTensor(),
    T.Normalize(mean=MEAN, std=STD),
])

val_transforms = T.Compose([
    T.Resize(IMAGE_SIZE + 32),  # Standard practice: resize to slightly larger, then center crop
    T.CenterCrop(IMAGE_SIZE),
    T.ToTensor(),
    T.Normalize(mean=MEAN, std=STD),
])

# -------- Config dataclasses --------
@dataclass
class ModelConfig:
    backbone: Literal["resnet18", "resnet50", "efficientnet_b0"] = "resnet18"
    pretrained: bool = True
    freeze_backbone: bool = False      # freeze all BUT final block + head
    drop_rate: float = 0.0             # dropout before classifier (if any)

@dataclass
class OptimConfig:
    optim: Literal["adamw", "sgd"] = "adamw"
    lr: float = 3e-4
    weight_decay: float = 1e-4
    momentum: float = 0.9              # SGD only
    scheduler: Literal["none", "step", "plateau"] = "step"
    step_size: int = 5                 # StepLR
    gamma: float = 0.5                 # StepLR
    patience: int = 3                  # ReduceLROnPlateau

@dataclass
class TrainConfig:
    epochs: int = 30
    seed: int = 42
    precision_at_k: int = 5            # K for Precision@K
    early_stop_patience: int = 7       # epochs without improvement

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
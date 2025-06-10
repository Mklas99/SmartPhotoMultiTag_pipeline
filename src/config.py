from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Literal

# -------- Core hyper‑parameters --------
IMAGE_SIZE: int = 224
BATCH_SIZE: int = 32
NUM_WORKERS: int = 2

DEFAULT_CLASSES: List[str] = [
    "person", "dog", "cat", "car", "bus", "bicycle",
    "pizza", "apple", "cell phone", "laptop"
]

# Where to store checkpoints
CHECKPOINT_DIR = Path("checkpoints")
CHECKPOINT_DIR.mkdir(exist_ok=True, parents=True)

# -------- Config dataclasses --------
@dataclass
class ModelConfig:
    backbone: Literal["resnet18", "resnet50", "efficientnet_b0"] = "resnet18"
    pretrained: bool = True
    drop_rate: float = 0.0

@dataclass
class OptimConfig:
    optim: Literal["adamw", "sgd"] = "adamw"
    lr: float = 3e-4
    weight_decay: float = 1e-4
    momentum: float = 0.9     # only for SGD

@dataclass
class TrainConfig:
    epochs: int = 5
    seed: int = 42
    early_stop_patience: int = 3

def all_configs():
    """Return a dict of every nested config for MLflow param logging."""
    cfgs = {}
    for cfg in (ModelConfig(), OptimConfig(), TrainConfig()):
        cfgs.update(asdict(cfg))
    return cfgs

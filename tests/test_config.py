from pathlib import Path
from PIL import Image
import numpy as np
from src.config import (
    TrainConfig, ModelConfig, OptimConfig, DEFAULT_CLASSES,
    get_num_classes, as_flat_dict, IMAGE_SIZE, BATCH_SIZE, NUM_WORKERS,
    MEAN, STD, train_transforms, val_transforms, CHECKPOINT_DIR, RESULTS_DIR
)

def test_train_config():
    cfg = TrainConfig()
    assert cfg.epochs > 0
    assert cfg.seed == 42

def test_classes_present():
    assert "person" in DEFAULT_CLASSES

def test_train_config():
    cfg = TrainConfig()
    assert cfg.epochs > 0
    assert cfg.seed == 42

def test_classes_present():
    assert "person" in DEFAULT_CLASSES
    assert isinstance(DEFAULT_CLASSES, list)

def test_model_config_defaults():
    cfg = ModelConfig()
    assert cfg.backbone == "resnet50"
    assert cfg.pretrained is True
    assert cfg.freeze_backbone is False
    assert 0.0 <= cfg.dropout_rate <= 1.0
    assert 0.0 <= cfg.dropout_rate <= 1.0

def test_optim_config_defaults():
    cfg = OptimConfig()
    assert cfg.optim in ["adamw", "sgd"]
    assert cfg.lr > 0
    assert cfg.weight_decay >= 0
    assert cfg.scheduler in ["none", "step", "plateau"]

def test_get_num_classes_default():
    assert get_num_classes() == len(DEFAULT_CLASSES)

def test_get_num_classes_custom():
    custom = ["a", "b", "c"]
    assert get_num_classes(custom) == 3

def test_as_flat_dict_keys():
    flat = as_flat_dict()
    assert "image_size" in flat
    assert "model_backbone" in flat
    assert "optim_lr" in flat
    assert "train_epochs" in flat

def test_checkpoint_and_results_dir_exist():
    assert isinstance(CHECKPOINT_DIR, Path)
    assert CHECKPOINT_DIR.exists()
    assert isinstance(RESULTS_DIR, Path)
    assert RESULTS_DIR.exists()

def test_train_transforms_callable():
    img = Image.fromarray((np.random.rand(256, 256, 3) * 255).astype("uint8"))
    out = train_transforms(img)
    assert out.shape[1:] == (IMAGE_SIZE, IMAGE_SIZE)

def test_val_transforms_callable():
    img = Image.fromarray((np.random.rand(256, 256, 3) * 255).astype("uint8"))
    out = val_transforms(img)
    assert out.shape[1:] == (IMAGE_SIZE, IMAGE_SIZE)

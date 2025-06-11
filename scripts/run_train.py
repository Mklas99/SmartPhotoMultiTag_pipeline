import argparse
import os
import sys

# Add the project root directory to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src import config
from src.train import run_training

def main():
    p = argparse.ArgumentParser(description="Train Photo Auto-Tagger (Step 3)")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--backbone", choices=["resnet50", "efficientnet_b0"], default="resnet18")
    p.add_argument("--freeze", action="store_true", help="Freeze backbone (feature extractor)")
    p.add_argument("--lr", type=float, default=3e-4)
    args = p.parse_args()

    model_cfg = config.ModelConfig(backbone=args.backbone, freeze_backbone=args.freeze)
    optim_cfg = config.OptimConfig(lr=args.lr)

    run_training(model_cfg=model_cfg, optim_cfg=optim_cfg, train_cfg=config.TrainConfig(epochs=args.epochs))

if __name__ == "__main__":
    main()
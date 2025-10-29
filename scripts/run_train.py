import argparse
import os
import sys

# Add the project root directory to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src import config
from src.train import run_training


def main():
    # Initialize configs to get defaults
    # This ensures that if run_train.py is called standalone,
    # it uses the defaults from config.py unless overridden by CLI args.
    default_model_cfg = config.ModelConfig()
    default_optim_cfg = config.OptimConfig()
    default_train_cfg = config.TrainConfig()

    p = argparse.ArgumentParser(description="Train Photo Auto-Tagger")

    p.add_argument(
        "--epochs",
        type=int,
        default=default_train_cfg.epochs,
        help=f"Number of training epochs (default: {default_train_cfg.epochs})",
    )
    p.add_argument(5
        "--backbone",
        choices=["resnet18", "resnet50", "efficientnet_b0"],
        default=default_model_cfg.backbone,
        help=f"Model backbone (default: {default_model_cfg.backbone})",
    )
    p.add_argument(
        "--max-samples",
        type=int,
        # Assuming TrainConfig will have max_samples. If not, use config.IMAGE_CNT or None
        default=getattr(default_train_cfg, "max_samples", config.IMAGE_CNT),
        help="Maximum number of data samples to use (default: uses config.IMAGE_CNT or TrainConfig default)",
    )
    p.add_argument(
        "--image-size",
        type=int,
        default=config.IMAGE_SIZE,
        help=f"Size of the images (default: {config.IMAGE_SIZE})",
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=config.BATCH_SIZE,
        help=f"Batch size for training (default: {config.BATCH_SIZE})",
    )
    p.add_argument(
        "--num-workers",
        type=int,
        default=config.NUM_WORKERS,
        help=f"Number of workers for data loading (default: {config.NUM_WORKERS})",
    )
    p.add_argument(
        "--early-stop-patience",
        type=int,
        default=default_train_cfg.early_stop_patience,
        help=f"Patience for early stopping (default: {default_train_cfg.early_stop_patience})",
    )
    p.add_argument(
        "--report",
        action="store_true",
        default=getattr(default_train_cfg, "report", True),
        help="Generate detailed report during training (default: False, or TrainConfig default)",
    )

    # Existing arguments specific to run_train.py or not yet in run_pipeline.py
    p.add_argument(
        "--freeze",
        action="store_true",
        default=default_model_cfg.freeze_backbone,
        help="Freeze backbone (feature extractor)",
    )
    p.add_argument(
        "--lr",
        type=float,
        default=default_optim_cfg.lr,
        help=f"Learning rate (default: {default_optim_cfg.lr})",
    )

    args = p.parse_args()

    # Override global config constants based on CLI arguments
    config.IMAGE_SIZE = args.image_size
    config.BATCH_SIZE = args.batch_size
    config.NUM_WORKERS = args.num_workers
    config.IMAGE_CNT = args.max_samples

    # Start with default configs and override only those provided via args
    model_cfg = default_model_cfg
    model_cfg.backbone = args.backbone
    model_cfg.freeze_backbone = args.freeze

    optim_cfg = default_optim_cfg
    optim_cfg.lr = args.lr

    train_cfg = default_train_cfg
    train_cfg.epochs = args.epochs
    train_cfg.early_stop_patience = args.early_stop_patience

    if hasattr(train_cfg, "max_samples"):
        train_cfg.max_samples = args.max_samples
    if hasattr(train_cfg, "report"):
        train_cfg.report = args.report

    print("Effective configurations for this run:")
    print(f"  Global: IMAGE_CNT={config.IMAGE_CNT}, IMAGE_SIZE={config.IMAGE_SIZE}, BATCH_SIZE={config.BATCH_SIZE}, NUM_WORKERS={config.NUM_WORKERS}")
    print(f"  ModelConfig: {model_cfg}")
    print(f"  OptimConfig: {optim_cfg}")
    print(f"  TrainConfig: {train_cfg}")

    run_training(model_cfg=model_cfg, optim_cfg=optim_cfg, train_cfg=train_cfg)


if __name__ == "__main__":
    main()

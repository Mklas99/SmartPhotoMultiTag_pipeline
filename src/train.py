import os
import string
import time
import numpy as np
from pyparsing import C
import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau

import mlflow

from dataclasses import asdict
from tqdm.auto import tqdm
import yaml

from src import config
from src.config import (
    BATCH_SIZE,
    DATASET_ROOT,
    NUM_WORKERS,
    RESULTS_DIR,
    ModelConfig,
    OptimConfig,
    TrainConfig,
    CHECKPOINT_DIR,
    train_transforms,
    val_transforms
)
from src.data.loader import load_data, CocoDataset, collate_fn
from src.models.basic_model import BasicMLC
from src.models.PhotoTagNet import PhotoTagNet
from src.utils.seed import set_seed
from src.utils.plot import save_loss_plot, save_sample_preds
from src.utils.train_util import _run_one_epoch, _validate,  _build_optimizer, _build_scheduler

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def run_training(
    model_cfg: ModelConfig = None,
    optim_cfg: OptimConfig = None,
    train_cfg: TrainConfig = None,
    custom_classes: list[str] | None = config.DEFAULT_CLASSES,
    generate_report: bool = False
):
    # ---- configs ----
    model_cfg = model_cfg or ModelConfig()
    optim_cfg = optim_cfg or OptimConfig()
    train_cfg = train_cfg or TrainConfig()

    set_seed(train_cfg.seed)
    
    # ---- Ensure directories and config.yaml exist ----
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR / "plots", exist_ok=True)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    config_path = RESULTS_DIR / "config.yaml"
    if not config_path.exists():
        with open(config_path, "w") as f:
            yaml.dump(config.as_flat_dict(), f)

    # ---- MLflow ----
    mlflow.set_experiment("photo-tagger")
    with mlflow.start_run(run_name="photo-tagger" + time.strftime("%Y%m%d-%H%M%S")):
        
        # ----------------------------- DATA & DATALOADERS -----------------------------
        num_classes = config.get_num_classes(custom_classes)
        train_ds = CocoDataset(
            DATASET_ROOT / "train/data", 
            DATASET_ROOT / "train/labels.json", 
            transform=train_transforms,
            target_category_names=custom_classes
        )
        val_ds = CocoDataset(
            DATASET_ROOT / "val/data", 
            DATASET_ROOT / "val/labels.json", 
            transform=val_transforms,
            target_category_names=custom_classes
        )
        
        train_ld = DataLoader(
            train_ds,
            batch_size=BATCH_SIZE,
            shuffle=True,           # shuffle for every epoch
            num_workers=NUM_WORKERS,
            collate_fn=collate_fn,
            #pin_memory=True, # pin_memory=True is useful for GPU training
        )
        val_ld = DataLoader(
            val_ds,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=NUM_WORKERS,
            collate_fn=collate_fn,
            #pin_memory=True,
        )

        # ----------------------------- MODEL, OPT & SCHEDULER -----------------------------
        print("Building model, optimizer and scheduler...")                               
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        #net = PhotoTagNet(model_cfg, num_classes).to(device)

        net = BasicMLC(num_classes).to(device)

        opt = _build_optimizer(net, optim_cfg)
        scheduler = _build_scheduler(opt, optim_cfg)

        # ----------------------------- LOG PARAMS -----------------------------
        print("Logging parameters to MLflow...")                                                
        mlflow.log_params(config.as_flat_dict())
        mlflow.log_params({"num_classes": num_classes, **model_cfg.__dict__})

        # ----------------------------- TRAIN LOOP -----------------------------
        print("Starting training loop...")                           
        best_val = -np.inf
        patience_left = train_cfg.early_stop_patience

        for epoch in tqdm(range(1, train_cfg.epochs + 1)):
            train_loss = _run_one_epoch(
                net,
                train_ld,
                opt,
                device,
                train=True,
            )
            val_loss, val_metrics = _validate(
                net,
                val_ld,
                device,
                k=train_cfg.precision_at_k,
            )

            # scheduler step
            if isinstance(scheduler, ReduceLROnPlateau):
                scheduler.step(val_loss)
            else:
                scheduler.step()

            # MLflow logging
            mlflow.log_metric("train_loss", train_loss, step=epoch)
            mlflow.log_metric("val_loss", val_loss, step=epoch)
            for k, v in val_metrics.items():
                mlflow.log_metric(k, v, step=epoch)

            # checkpoint if best macro-F1 improves
            if val_metrics["macro_F1"] > best_val:
                best_val = val_metrics["macro_F1"]
                patience_left = train_cfg.early_stop_patience
                ckpt_path = CHECKPOINT_DIR / f"best_epoch_{epoch}.pt"
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state": net.state_dict(),
                        "optim_state": opt.state_dict(),
                    },
                    ckpt_path,
                )
                mlflow.log_artifact(str(ckpt_path))
                # qualitative artefacts
                preds_path = save_sample_preds(net, val_ds, device, train_ds.category_names)
                mlflow.log_artifact(str(preds_path))
            else:
                patience_left -= 1

            print(
                f"[{epoch}/{train_cfg.epochs}] "
                f"train={train_loss:.4f}  val={val_loss:.4f}  "
                f"macroF1={val_metrics['macro_F1']:.3f}"
            )

            if patience_left == 0:
                print("Early stopping – no improvement.")
                break

        mlflow.log_artifact(str(RESULTS_DIR / "config.yaml"), artifact_path="run_config")
        mlflow.end_run()
        print("Training completed and logged to MLflow.")

if __name__ == "__main__":
    import argparse, os
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=None, help="Override epochs")
    parser.add_argument("--report", action="store_true", help="Generate detailed report")
    args = parser.parse_args()
    run_training(args.epochs, generate_report=args.report)

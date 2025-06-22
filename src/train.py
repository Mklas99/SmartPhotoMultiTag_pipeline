import os
import time
import mlflow
import numpy as np
import torch
import yaml
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingWarmRestarts
from torchvision.transforms.v2 import MixUp
from tqdm.auto import tqdm

from src import config
from src.config import (
    BATCH_SIZE,
    CHECKPOINT_DIR,
    IMAGE_CNT,
    NUM_WORKERS,
    RESULTS_DIR,
    ModelConfig,
    OptimConfig,
    TrainConfig,
)
from src.data.loader import load_data
from src.models.PhotoTagNet_model import PhotoTagNet
from src.models.basic_model import BasicMLC
from src.utils.predict import predict
from src.utils.plot import (
    save_loss_plot,
    save_sample_preds,
)
from src.utils.seed import set_seed
from src.utils.train_util import (
    _build_optimizer,
    _build_scheduler,
    _run_one_epoch,
    _validate,
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def run_training(
    model_cfg: ModelConfig = None,
    optim_cfg: OptimConfig = None,
    train_cfg: TrainConfig = None,
    custom_classes: list[str] | None = config.DEFAULT_CLASSES,
    generate_report: bool = True,
    use_mixup: bool = False,
    label_smoothing: float = 0.0,
):

    # ---- configs ----
    model_cfg = model_cfg or ModelConfig()
    optim_cfg = optim_cfg or OptimConfig()
    train_cfg = train_cfg or TrainConfig()

    set_seed(train_cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---- Ensure directories and config.yaml exist ----
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR / "plots", exist_ok=True)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    config_path = RESULTS_DIR / "config.yaml"
    if not config_path.exists():
        with open(config_path, "w") as f:
            yaml.dump(config.as_flat_dict(), f)

    # ---- MLflow ----
    mlflow.set_experiment("photo-tagger-experiment")
    with mlflow.start_run(run_name="photo-tagger-experiment1k" + time.strftime("%Y%m%d-%H%M%S")):

        # ----------------------------- DATA / DATALOADERS -----------------------------
        num_classes = config.get_num_classes(custom_classes)

        print("Loading datasets and creating dataloaders...")
        mixup = MixUp(alpha=0.4) if use_mixup else None
        train_dataset, val_dataset, train_loader, val_loader = load_data(
            classes=custom_classes,
            max_samples=IMAGE_CNT,
            batch_size=BATCH_SIZE,
            num_workers=NUM_WORKERS,
            mixup_collate_fn=mixup if use_mixup else None,
        )
        print(f"Data loaded. Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

        # ----------------------------- MODEL, OPT & SCHEDULER -----------------------------
        print("Building model, optimizer and scheduler...")
        net = BasicMLC(num_classes).to(device)
        #net = PhotoTagNet(model_cfg, num_classes).to(device)
        opt = _build_optimizer(net, optim_cfg)
        if optim_cfg.scheduler == "cosine_warm_restart":
            scheduler = CosineAnnealingWarmRestarts(opt, T_0=5, T_mult=2)
        else:
            scheduler = _build_scheduler(opt, optim_cfg)

        # ----------------------------- LOG PARAMS -----------------------------
        print("Logging parameters to MLflow...")
        mlflow.log_params(config.as_flat_dict())
        mlflow.log_params(
            {
                "num_classes": num_classes,
                "classes": custom_classes,
                **model_cfg.__dict__,
            }
        )

        criterion = torch.nn.BCEWithLogitsLoss() if label_smoothing == 0 else torch.nn.BCEWithLogitsLoss(label_smoothing=label_smoothing)

        # ----------------------------- TRAIN LOOP -----------------------------
        print("Starting training loop...")
        best_val = -np.inf
        patience_left = train_cfg.early_stop_patience
        train_losses, val_losses = [], []

        for epoch in tqdm(range(1, train_cfg.epochs + 1)):
            # --- Staged unfreezing ---
            if epoch == 3:
                for name, p in net.backbone.named_parameters():
                    if "layer4" in name:
                        p.requires_grad = True
            if epoch == 6:
                for p in net.backbone.parameters():
                    p.requires_grad = True
            train_loss = _run_one_epoch(
                net,
                train_loader,
                opt,
                device,
                epoch_nbr=epoch,
                criterion=criterion,
            )
            val_loss, val_metrics = _validate(
                net,
                val_loader,
                device,
                k=train_cfg.precision_at_k,
                epoch_nbr=epoch,
                criterion=criterion,
            )

            train_losses.append(train_loss)
            val_losses.append(val_loss)

            # ---------------- scheduler step
            if isinstance(scheduler, ReduceLROnPlateau):
                scheduler.step(val_loss)
            else:
                scheduler.step()

            # ---------------- MLflow logging
            mlflow.log_metric("train_loss", train_loss, step=epoch)
            mlflow.log_metric("val_loss", val_loss, step=epoch)

            # if generate_report:
            #     save_loss_plot_path = save_loss_plot(train_loss, val_loss, titel=f"epoch_{epoch}_of_{train_cfg.epochs}")
            #     mlflow.log_artifact(str(save_loss_plot_path))

            for k, v in val_metrics.items():
                mlflow.log_metric(k, v, step=epoch)

            # ---------------- checkpoint if best macro-F1 improves
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
                if generate_report:
                    preds_path = save_sample_preds(
                        net,
                        val_dataset,
                        device,
                        train_dataset.category_names,
                        f"epoch_{epoch}_of_{train_cfg.epochs}",
                    )
                    mlflow.log_artifact(str(preds_path))
                    print(f"Saved sample predictions to {preds_path}")
            else:
                patience_left -= 1

            print(f"Epoch {epoch+1}/{train_cfg.epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

            if patience_left == 0:
                print("Early stopping – no improvement.")
                break

        if generate_report:
            # Save final loss curve
            save_loss_plot_path = save_loss_plot(train_losses, val_losses, "Training & Validation Loss")
            mlflow.log_artifact(str(save_loss_plot_path))
            print(f"Loss curve saved to {save_loss_plot_path}")

            # Export loss data to a .txt file
            loss_data_path = RESULTS_DIR / "loss_data.csv"
            with open(loss_data_path, "w") as f:
                f.write("Epoch,TrainLoss,ValLoss\n")
                for i, (t_loss, v_loss) in enumerate(zip(train_losses, val_losses)):
                    f.write(f"{i+1},{t_loss:.4f},{v_loss:.4f}\n")
            mlflow.log_artifact(str(loss_data_path))
            print(f"Loss data exported to {loss_data_path}")

        mlflow.log_artifact(str(RESULTS_DIR / "config.yaml"), artifact_path="run_config")
        mlflow.end_run()
        print("Training completed and logged to MLflow.")
        print(f"Best validation macro-F1: {best_val:.4f} at epoch {epoch}")

        # Save the best performing checkpoint as model
        best_ckpt = max(CHECKPOINT_DIR.glob("best_epoch_*.pt"), key=os.path.getctime, default=None)
        if best_ckpt:
            model_save_path = RESULTS_DIR / "best_model.pt"
            torch.save(torch.load(best_ckpt), model_save_path)
            mlflow.log_artifact(str(model_save_path))
            y_pred_prob, y_val = predict(net, val_loader)
            y_val_path = RESULTS_DIR / "y_val.npy"
            y_pred_prob_path = RESULTS_DIR / "y_pred_prob.npy"
            np.save(y_val_path, y_val)
            np.save(y_pred_prob_path, y_pred_prob)
            mlflow.log_artifact(str(y_val_path))
            mlflow.log_artifact(str(y_pred_prob_path))
        print(f"Best model saved to {model_save_path}")

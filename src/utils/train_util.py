import warnings
from typing import Dict, Tuple

import mlflow
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD, AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau, StepLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.config import OptimConfig
from src.utils.metrics import macro_f1, mAP, micro_f1, precision_at_k, roc_auc
from src.utils.plot import save_confusion_matrix, save_roc_curves


def _build_optimizer(net: nn.Module, cfg: OptimConfig):
    params = net.trainable_parameters()
    if cfg.optim == "adamw":
        # AdamW = Adam + "decoupled" weight-decay (better for modern nets) adaptable learning rate
        return AdamW(params, lr=cfg.lr, betas=cfg.betas)
    elif cfg.optim == "sgd":
        return SGD(
            params,
            lr=cfg.lr,
            momentum=cfg.momentum,
            weight_decay=cfg.weight_decay,
            nesterov=True,  # Nesterov gives slightly faster convergence
        )
    else:
        raise ValueError(cfg.optim)


def _build_scheduler(opt, cfg: OptimConfig):
    """
    Returns a PyTorch LR-scheduler based on the configuration.
    """
    if cfg.scheduler == "step":
        # Classic ɣ decay every K epochs – stable & predictable.
        return StepLR(opt, step_size=cfg.step_size, gamma=cfg.gamma)

    elif cfg.scheduler == "plateau":
        # Reduce LR when val-metric plateaus – robust for noisy datasets.
        return ReduceLROnPlateau(opt, patience=cfg.patience, factor=cfg.gamma)

    elif cfg.scheduler == "cosine":
        # Smooth annealing to (almost) zero; performs well for long runs.
        return CosineAnnealingLR(opt, T_max=cfg.step_size, eta_min=cfg.lr * cfg.gamma)

    else:

        class _NoSched:
            def step(self, *_):
                pass

        return _NoSched()


def _run_one_epoch(
    net: nn.Module,
    loader: DataLoader,
    opt,  # Optimizer, required if train=True
    device,
    epoch_nbr: int = 0,
) -> float:
    net.train()

    all_lossess = []
    progress_bar = tqdm(loader, desc=f"Epoch {epoch_nbr} [Training]", unit="batch")

    for imgs, labels in progress_bar:  # Iterate over batches
        imgs, labels = imgs.to(device), labels.to(device)

        if opt is None:
            raise ValueError("Optimizer 'opt' must be provided.")

        opt.zero_grad()  # clear gradients every batch to make sure they don't overwritten or accumulate
        logits = net(imgs)
        loss = F.binary_cross_entropy_with_logits(logits, labels)
        loss.backward()
        opt.step()

        all_lossess.append(loss.item())
        progress_bar.set_postfix(loss=loss.item())

    return float(np.mean(all_lossess))


@torch.no_grad()
def _validate(
    net: nn.Module,
    loader: DataLoader,
    device,
    k: int,
    do_plots: bool = True,
    epoch_nbr: int = 0,
) -> Tuple[float, Dict[str, float]]:
    net.eval()
    val_running_loss = 0.0
    all_losses, all_preds_scores, all_gts = [], [], []

    with torch.no_grad():  # Disable gradient computation for validation
        progress_bar_val = tqdm(loader, desc=f"Epoch {epoch_nbr} [Validation]", unit="batch")
        for imgs, labels in progress_bar_val:  # Iterate over unpacked batches
            imgs, labels = imgs.to(device), labels.to(device)

            logits = net(imgs)
            # Compute binary cross-entropy loss
            pos_weight = compute_reweight_vector(
                loader.dataset.get_label_counts()
            )  # Reweighting vector based on label frequencies
            loss = F.binary_cross_entropy_with_logits(
                logits, labels, pos_weight=pos_weight.to(device)
            )
            all_losses.append(loss.item())
            val_running_loss += loss.item()
            progress_bar_val.set_postfix(loss=loss.item())

            # pfredictions and ground truths
            all_preds_scores.append(torch.sigmoid(logits).cpu())
            all_gts.append(labels.cpu())

    y_score = np.vstack([p.numpy() for p in all_preds_scores])
    y_true = np.vstack([gt.numpy() for gt in all_gts])

    metrics = {
        "macro_F1": macro_f1(y_true, y_score),
        "micro_F1": micro_f1(y_true, y_score),
        "precision_at_K": precision_at_k(y_true, y_score, k=k),
        "mAP": mAP(y_true, y_score),
        "ROC_AUC": roc_auc(y_true, y_score),
    }

    # diagnostic plots
    if do_plots:
        try:
            cm_img = save_confusion_matrix(
                y_true, y_score, title=f"macro F1 {metrics['macro_F1']:.3f}"
            )
            roc_img = save_roc_curves(y_true, y_score, title=f"macro F1 {metrics['macro_F1']:.3f}")
            if mlflow.active_run():  # Check if there's an active MLflow run
                mlflow.log_artifact(str(cm_img))
                mlflow.log_artifact(str(roc_img))
            else:
                warnings.warn("No active MLflow run. Skipping artifact logging for plots.")
        except Exception as e:
            warnings.warn(f"Error during diagnostic plot generation or MLflow logging: {e}")

    return float(np.mean(all_losses)), metrics


def compute_reweight_vector(label_counts: torch.Tensor) -> torch.Tensor:
    """
    Compute a reweighting vector based on label frequencies.
    """
    total = label_counts.sum().float()
    freq = label_counts.float() / total

    # inverse frequency – smooth with pos epsilon to avoid exploding on zeros
    epsilon = 1e-6
    inv_freq = 1.0 / (freq + epsilon)

    # scale so the average weight = 1 (purely cosmetic but convenient)
    return inv_freq * (inv_freq.numel() / inv_freq.sum())

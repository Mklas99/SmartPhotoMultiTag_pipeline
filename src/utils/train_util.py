import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW, SGD
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau
from torch.utils.data import DataLoader
import numpy as np
from typing import Tuple, Dict
from tqdm import tqdm
import mlflow

# Import from src path
from src import config
from src.models.metrics import macro_f1, micro_f1, precision_at_k, mAP, roc_auc
from src.utils.plot import save_confusion_matrix, save_roc_curves
from src.utils.average_meter import AverageMeter
from src.config import OptimConfig


def _build_optimizer(net: nn.Module, cfg: OptimConfig):
    params = (p for p in net.parameters() if p.requires_grad)
    if cfg.optim == "adamw": # adaptive learning rate 
        return AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    elif cfg.optim == "sgd":
        return SGD(
            params,
            lr=cfg.lr,
            momentum=cfg.momentum,
            weight_decay=cfg.weight_decay,
            nesterov=True,
        )
    else:  # type: ignore[unreachable]
        raise ValueError(cfg.optim)

def _build_scheduler(opt, cfg: OptimConfig):
    if cfg.scheduler == "step":
        return StepLR(opt, step_size=cfg.step_size, gamma=cfg.gamma)
    elif cfg.scheduler == "plateau":
        return ReduceLROnPlateau(opt, patience=cfg.patience, factor=cfg.gamma)
    else:
        class _NoSched:
            def step(self, *_): pass
        return _NoSched()

def _run_one_epoch(
    net: nn.Module,
    loader: DataLoader,
    opt, # Optimizer, required if train=True
    device,
    train: bool = True,
) -> float:
    import warnings # Ensure warnings is imported
    net.train(train)
    meter = AverageMeter()

    for batch_data in tqdm(loader): # Iterate over batches
        if batch_data is None:
            warnings.warn(
                "Skipping a batch in _run_one_epoch because it's None. "
                "This might be due to all samples in the batch being invalid."
            )
            continue

        imgs, labels = batch_data # Unpack after None check
        imgs, labels = imgs.to(device), labels.to(device)
        logits = net(imgs)
        loss = F.binary_cross_entropy_with_logits(logits, labels)

        if train:
            if opt is None:
                raise ValueError("Optimizer 'opt' must be provided when train=True.")
            opt.zero_grad() # clear gradients every batch to make sure they don't overwritten or accumulate 
            loss.backward()
            opt.step()

        meter.update(loss.item(), imgs.size(0))

    if meter.count == 0 and len(loader) > 0:
        warnings.warn(
            "_run_one_epoch completed but no batches were processed successfully. "
            "Loss will be NaN or 0.0. Check dataset and dataloader."
        )
    return meter.avg


@torch.no_grad()
def _validate(
    net: nn.Module,
    loader: DataLoader,
    device,
    k: int,
) -> Tuple[float, Dict[str, float]]:
    import warnings # Ensure warnings is imported
    net.eval()
    all_losses, all_preds_scores, all_gts = [], [], []

    for batch_data in loader: # Iterate over batches
        if batch_data is None:
            warnings.warn(
                "Skipping a batch in _validate because it's None. "
                "This might be due to all samples in the batch being invalid."
            )
            continue

        imgs, labels = batch_data # Unpack after None check
        imgs, labels = imgs.to(device), labels.to(device)
        logits = net(imgs)
        loss = F.binary_cross_entropy_with_logits(logits, labels)
        all_losses.append(loss.item())

        all_preds_scores.append(torch.sigmoid(logits).cpu())
        all_gts.append(labels.cpu())

    if not all_losses: # Handle case where no valid batches were processed
        warnings.warn("_validate processed no valid batches. Returning default/error metrics.")
        # Define appropriate default metrics for this case
        default_metrics = {
            metric: 0.0 for metric in ["macro_F1", "micro_F1", "precision_at_K", "mAP", "ROC_AUC"]
        }
        # No images logged to mlflow in this case.
        return float('nan'), default_metrics

    y_score = np.vstack([p.numpy() for p in all_preds_scores])
    y_true = np.vstack([gt.numpy() for gt in all_gts])
    
    actual_num_classes = y_true.shape[1] # Use actual number of classes from data for plots

    metrics = {
        "macro_F1": macro_f1(y_true, y_score),
        "micro_F1": micro_f1(y_true, y_score),
        "precision_at_K": precision_at_k(y_true, y_score, k=k),
        "mAP": mAP(y_true, y_score),
        "ROC_AUC": roc_auc(y_true, y_score),
    }

    # diagnostic plots
    try:
        cm_img = save_confusion_matrix(y_true, y_score, num_classes=actual_num_classes)
        roc_img = save_roc_curves(y_true, y_score) # Pass num_classes if your function expects it
        if mlflow.active_run(): # Check if there's an active MLflow run
            mlflow.log_artifact(str(cm_img))
            mlflow.log_artifact(str(roc_img))
        else:
            warnings.warn("No active MLflow run. Skipping artifact logging for plots.")
    except ImportError:
        warnings.warn("mlflow not installed or available. Skipping artifact logging.")
    except Exception as e:
        warnings.warn(f"Error during diagnostic plot generation or MLflow logging: {e}")
        
    return float(np.mean(all_losses)), metrics
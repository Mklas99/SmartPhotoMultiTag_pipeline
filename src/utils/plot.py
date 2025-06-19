"""
Small helpers: seeding, meters, plot saving.
"""

from __future__ import annotations

import random
import string
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import confusion_matrix, precision_recall_curve

from src import config
from src.data.cocodataset import CocoDataset


def _tmp_png(prefix: str) -> Path:
    rnd = "".join(random.choices(string.ascii_letters, k=6))
    (config.RESULTS_DIR / "plots").mkdir(parents=True, exist_ok=True)
    path = config.RESULTS_DIR / "plots" / Path(f"{prefix}_{rnd}.png")
    return path


def save_loss_plot(train_loss: list, val_loss: list, titel: str = "") -> Path:
    plt.figure()
    plt.plot(train_loss, label="train")
    plt.plot(val_loss, label="val")
    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss Plot" + f" - {titel}")

    path = _tmp_png("loss" + titel)
    plt.savefig(path)
    plt.close()
    return path


def save_confusion_matrix(y_true, y_score, title="") -> Path:
    preds = (y_score >= 0.5).astype(int)
    cm = confusion_matrix(y_true.reshape(-1), preds.reshape(-1))
    fig, ax = plt.subplots()
    ax.imshow(cm, cmap="Blues")
    ax.set_title("Confusion Matrix (flattened)")
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    fig.tight_layout()
    plt.title("Confusion Matrix" + f" - {title}")

    path = _tmp_png("cm")
    fig.savefig(path)
    plt.close(fig)
    return path


def save_roc_curves(y_true, y_score, title="") -> Path:
    fig, ax = plt.subplots()
    for c in range(y_true.shape[1]):
        try:
            pr, rec, _ = precision_recall_curve(y_true[:, c], y_score[:, c])
            ax.plot(rec, pr, lw=0.8)
        except ValueError:
            continue
    ax.set_title("PR Curves per class")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    fig.tight_layout()
    plt.title("PR Curves" + f" - {title}")

    path = _tmp_png("pr" + title)
    fig.savefig(path)
    plt.close(fig)
    return path


def save_sample_preds(
    model, val_ds: CocoDataset, device, class_names: List[str], title="", n: int = 6
) -> Path:
    """Save a grid of n validation images + predicted top-5(max) tags."""
    # First, ensure 'n' is not negative. and avlidation dataset is not empty.
    if len(val_ds) == 0:
        print(
            "Warning: Validation dataset is empty. No samples will be selected for plotting."
        )
        idxs = []

    if n < 0:
        print(
            f"Warning: The desired number of samples 'n' is negative ({n}). Will attempt to sample 0 items."
        )
        n = 0
    else:
        n = n

    model.eval()
    idxs = random.sample(range(len(val_ds)), n)
    fig, axes = plt.subplots(2, n // 2, figsize=(14, 6))
    axes = axes.flatten()
    mean = np.array(config.MEAN)
    std = np.array(config.STD)

    for ax, i in zip(axes, idxs):
        img_t, gt = val_ds[i]
        with torch.no_grad():
            pred = torch.sigmoid(model(img_t.unsqueeze(0).to(device)))[0]
        k = min(5, len(pred))
        top5_indices = pred.topk(k).indices.cpu().numpy()
        top5_names = [class_names[j] for j in top5_indices]
        # denorm
        np_img = img_t.permute(1, 2, 0).numpy() * std + mean
        ax.imshow(np.clip(np_img, 0, 1))
        ax.axis("off")
        ax.set_title(f"Top: {', '.join(top5_names)}", fontsize=8)
    fig.suptitle("Sample Predictions" + (f" - {title}" if title else ""), fontsize=14)
    fig.tight_layout()

    path = _tmp_png("sample-pred_" + title)
    fig.savefig(path)
    plt.close(fig)
    return path

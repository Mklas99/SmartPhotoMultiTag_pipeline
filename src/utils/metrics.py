"""
Multi-label metrics handy for training & validation.
"""

from __future__ import annotations

import torch
from sklearn.metrics import (
    f1_score,
    average_precision_score,
    roc_auc_score,
)
import numpy as np


@torch.no_grad()
def macro_f1(y_true, y_pred, thresh=0.5):
    y_pred_bin = (y_pred >= thresh).astype(int)
    return f1_score(y_true, y_pred_bin, average="macro", zero_division=0)


@torch.no_grad()
def micro_f1(y_true, y_pred, thresh=0.5):
    y_pred_bin = (y_pred >= thresh).astype(int)
    return f1_score(y_true, y_pred_bin, average="micro", zero_division=0)


@torch.no_grad()
def precision_at_k(y_true, y_score, k: int):
    """Precision@K averaged over samples."""
    topk = np.argsort(-y_score, axis=1)[:, :k]  # highest scores
    hits = np.take_along_axis(y_true, topk, axis=1)
    return hits.sum() / (y_true.shape[0] * k + 1e-9)


@torch.no_grad()
def mAP(y_true, y_score):
    return average_precision_score(y_true, y_score, average="macro")


@torch.no_grad()
def roc_auc(y_true, y_score):
    # handle classes with only 0s or 1s by suppressing errors
    try:
        return roc_auc_score(y_true, y_score, average="macro")
    except ValueError:
        return np.nan

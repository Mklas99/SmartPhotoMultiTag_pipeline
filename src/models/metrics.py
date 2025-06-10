from sklearn.metrics import f1_score
import numpy as np

def compute_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Micro‑averaged F1 for multi‑label."""
    return f1_score(y_true, y_pred, average="micro", zero_division=0)

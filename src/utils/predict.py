from typing import Tuple
from numpy import ndarray
import torch

def predict(model, val_loader) -> Tuple[ndarray, ndarray]:
    """
    Predicts probabilities for the validation set using the given model.
    
    Args:
        model: The trained model to use for predictions.
        val_loader: DataLoader for the validation dataset.
    
    Returns:
        y_pred_prob: Predicted probabilities for each class.
        y_val: True labels for the validation set.
    """
    model.eval()
    all_probs, all_labels = [], []
    with torch.no_grad():
        for x, y in val_loader:
            probs = torch.sigmoid(model(x))
            all_probs.append(probs.cpu())
            all_labels.append(y.cpu())

    y_pred_prob = torch.cat(all_probs).numpy()   # shape (n_samples, n_labels)
    y_val       = torch.cat(all_labels).numpy()

    return y_pred_prob, y_val
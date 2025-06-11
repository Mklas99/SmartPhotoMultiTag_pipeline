import mlflow, torch, numpy as np
from src.models.basic_model import BasicMLC
from src.utils.metrics import micro_f1
from src.config import CHECKPOINT_DIR, DEFAULT_CLASSES, ModelConfig
from src.data.loader import load_data

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def run_evaluation():
    model = BasicMLC(DEFAULT_CLASSES).to(DEVICE)
    ckpt = CHECKPOINT_DIR / "best_model.pth"
    if not ckpt.exists():
        raise FileNotFoundError("Checkpoint not found; run training first.")
    model.load_state_dict(torch.load(ckpt, map_location=DEVICE))
    model.eval()

    _, val_loader = load_data()
    preds, gts = [], []
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs = imgs.to(DEVICE)
            out = model(imgs).sigmoid().cpu().numpy()
            preds.append((out > 0.5).astype(np.float32))
            gts.append(labels.numpy())

    y_pred = np.vstack(preds)
    y_true = np.vstack(gts)
    f1 = micro_f1(y_true, y_pred)

    mlflow.start_run(run_name="evaluation")
    mlflow.log_metric("f1_score", f1)
    mlflow.end_run()
    print(f"F1(micro) = {f1:.4f}")

if __name__ == "__main__":
    run_evaluation()

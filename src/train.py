import mlflow
import torch
from dataclasses import asdict
from tqdm.auto import tqdm
import os
from src.config import ModelConfig, OptimConfig, TrainConfig, CHECKPOINT_DIR, all_configs
from src.data.loader import load_data
from src.models.model import build_model
from src.utils.seed import set_seed
from src.utils.plot import save_loss_plot

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def run_training(epochs: int | None = None, generate_report: bool = False):
    # ---- configs ----
    mcfg = ModelConfig()
    ocfg = OptimConfig()
    tcfg = TrainConfig()
    if epochs is not None:
        tcfg.epochs = epochs

    set_seed(tcfg.seed)

    # ---- MLflow ----
    mlflow.start_run()
    mlflow.log_params(all_configs())
    if generate_report:
        mlflow.log_param("detailed_report", True)

    # ---- data ----
    train_loader, val_loader = load_data()

    # ---- model, loss, optim ----
    model = build_model(mcfg).to(DEVICE)
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=ocfg.lr, weight_decay=ocfg.weight_decay)

    best_val = float("inf")
    train_losses, val_losses = [], []

    for epoch in range(tcfg.epochs):
        model.train()
        running_loss = 0.0
        for imgs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{tcfg.epochs}"):
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            out = model(imgs)
            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        train_loss = running_loss / len(train_loader)
        train_losses.append(train_loss)
        mlflow.log_metric("train_loss", train_loss, step=epoch)

        # ---- val ----
        model.eval()
        val_running = 0.0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
                out = model(imgs)
                val_running += criterion(out, labels).item()
        val_loss = val_running / len(val_loader)
        val_losses.append(val_loss)
        mlflow.log_metric("val_loss", val_loss, step=epoch)

        # ---- checkpoint ----
        if val_loss < best_val:
            best_val = val_loss
            ckpt = CHECKPOINT_DIR / "best_model.pth"
            torch.save(model.state_dict(), ckpt)
            mlflow.log_artifact(str(ckpt))

    # ---- plot ----
    plot_path = "results/loss_curve.png"
    os.makedirs("results", exist_ok=True)
    save_loss_plot(train_losses, val_losses, plot_path)
    mlflow.log_artifact(plot_path)
    
    # Get a batch of sample data for the input example
    sample_batch, _ = next(iter(train_loader))
    sample_batch = sample_batch.to(DEVICE)

    # ---- final model ----
    # Convert your torch tensor to numpy before logging
    input_example_numpy = sample_batch.detach().cpu().numpy()
    mlflow.pytorch.log_model(
        model, 
        name="model",
        input_example=input_example_numpy,  # Use numpy array instead of torch.Tensor
    )
    # Save the final model for later prediction use
    final_model_path = CHECKPOINT_DIR / "final_model.pth"
    torch.save(model.state_dict(), final_model_path)
    mlflow.log_artifact(str(final_model_path))

    mlflow.end_run()

if __name__ == "__main__":
    import argparse, os
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=None, help="Override epochs")
    parser.add_argument("--report", action="store_true", help="Generate detailed report")
    args = parser.parse_args()
    run_training(args.epochs, generate_report=args.report)

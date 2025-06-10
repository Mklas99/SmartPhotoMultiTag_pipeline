import argparse
from src.train import run_training

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=1, help="Epochs for quick run")
    args = parser.parse_args()
    run_training(args.epochs)

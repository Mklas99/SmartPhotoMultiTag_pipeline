import argparse
import sys
import os

# Add the project root to Python's module search path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '..'))
sys.path.insert(0, project_root)

from src.train import run_training

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=1, help="Epochs for quick run")
    args = parser.parse_args()
    run_training(args.epochs)

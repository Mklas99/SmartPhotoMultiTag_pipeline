import os
import sys

# Add the project root directory to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.train import run_training

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=None, help="Override epochs")
    parser.add_argument("--report", action="store_true", help="Generate detailed report")
    args = parser.parse_args()
    run_training(args.epochs, generate_report=args.report)

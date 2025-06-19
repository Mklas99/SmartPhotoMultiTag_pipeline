import argparse
import os
import subprocess
import sys

# Add the project root directory to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the training and evaluation pipeline.")

    # Existing arguments
    parser.add_argument(
        "--max-samples",
        type=int,
        default=5,  # Default for CI: limit samples for a quick run
        help="Maximum number of data samples to use. Passed to run_train.py.",
    )
    parser.add_argument(
        "--report",
        action="store_true",
        help="Generate detailed report during training. Passed to run_train.py.",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=32,  # Default for CI: small image size
        help="Size of the images. Passed to run_train.py.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,  # Default for CI: small batch size
        help="Batch size for training. Passed to run_train.py.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,  # Default for CI: no parallel data loading
        help="Number of workers for data loading. Passed to run_train.py.",
    )
    parser.add_argument(
        "--backbone",
        type=str,
        default="resnet18",  # Default for CI: lighter backbone
        choices=["resnet18", "resnet50", "efficientnet_b0"],
        help="Model backbone. Passed to run_train.py.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=1,  # Default for CI: minimal epochs for a quick run
        help="Number of training epochs. Passed to run_train.py.",
    )
    parser.add_argument(
        "--early-stop-patience",
        type=int,
        default=2,  # Default for CI: small patience
        help="Patience for early stopping. Passed to run_train.py.",
    )

    args = parser.parse_args()

    # Base command for run_train.py
    cmd = [sys.executable, "scripts/run_train.py"]

    # Add arguments to the command
    cmd.extend(["--epochs", str(args.epochs)])
    cmd.extend(["--max-samples", str(args.max_samples)])
    cmd.extend(["--image-size", str(args.image_size)])
    cmd.extend(["--batch-size", str(args.batch_size)])
    cmd.extend(["--num-workers", str(args.num_workers)])
    cmd.extend(["--backbone", args.backbone])
    cmd.extend(["--early-stop-patience", str(args.early_stop_patience)])

    if args.report:
        cmd.append("--report")

    print(f"Running command: {' '.join(cmd)}")
    subprocess.check_call(cmd)

    print("Pipeline finished.")

import argparse
import os
import sys

# Add the project root directory to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.config import DEFAULT_CLASSES, IMAGE_CNT
from src.data.loader import prepare_dataset

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download and prepare COCO dataset.")
    parser.add_argument(
        "--max-samples",
        type=int,
        default=IMAGE_CNT,
        help="Maximum number of samples per split to download from COCO.",
    )
    parser.add_argument(
        "--classes",
        nargs="+",
        default=DEFAULT_CLASSES,
        help="List of classes to filter from COCO dataset.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    args = parser.parse_args()

    print(
        f"Preparing dataset with max_samples={args.max_samples}, classes={args.classes}, seed={args.seed}"
    )
    prepare_dataset(
        classes=args.classes,
        max_samples=args.max_samples,
        seed=args.seed,
    )
    print("Dataset preparation finished.")

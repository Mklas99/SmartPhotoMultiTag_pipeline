import os
import sys
import subprocess
import argparse

# Add the project root directory to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the training and evaluation pipeline.")
    parser.add_argument(
        "--max-samples", 
        type=int, 
        default=1,  # Default for CI if not overridden by workflow inputs
        help="Number of epochs for training. Corresponds to --epochs in run_train.py."
    )
    parser.add_argument(
        "--report", 
        action="store_true", 
        help="Generate detailed report during training."
    )
    args = parser.parse_args()
    
    # Run training
    cmd = [sys.executable, "scripts/run_train.py", "--epochs", str(args.max_samples)]
    if args.report:
        cmd.append("--report")
    
    print(f"Running command: {' '.join(cmd)}")
    subprocess.check_call(cmd)
    
    print("Pipeline finished.")

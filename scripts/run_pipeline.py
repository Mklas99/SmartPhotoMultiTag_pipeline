import subprocess, sys, os
import sys
import os
# Add the project root to Python's module search path
# Add the project root to Python's module search path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '..'))
sys.path.insert(0, project_root)

# Use absolute path to run_train.py
train_script_path = os.path.join(script_dir, "run_train.py")

subprocess.check_call([sys.executable, "scripts/run_train.py", "--epochs", "50"])
subprocess.check_call([sys.executable, "-m", "src.evaluate"])

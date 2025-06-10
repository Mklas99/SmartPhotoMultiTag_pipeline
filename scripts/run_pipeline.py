import os
import sys
import subprocess

# Add the project root directory to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

if __name__ == "__main__":
    # Forward the REPORT environment variable as a flag if it exists
    report_flag = "--report" if os.environ.get("REPORT") == "true" else ""
    
    # Run training with report flag if needed
    cmd = [sys.executable, "scripts/run_train.py", "--epochs", "50"]
    if report_flag:
        cmd.append(report_flag)
    
    subprocess.check_call(cmd)

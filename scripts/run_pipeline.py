import subprocess, sys
subprocess.check_call([sys.executable, "scripts/run_train.py", "--epochs", "1"])
subprocess.check_call([sys.executable, "-m", "src.evaluate"])

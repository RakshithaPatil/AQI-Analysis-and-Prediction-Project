"""
ml/launch_training.py
Launches train_models.py using the correct Python (3.13 with sklearn).
"""
import subprocess, sys, os
from pathlib import Path

PYTHON313 = r"C:\Users\Rakshitha Patil\AppData\Local\Programs\Python\Python313\python.exe"
ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)
os.environ["PYTHONPATH"] = str(ROOT)

subprocess.run([PYTHON313, "ml/train_models.py"], check=True)

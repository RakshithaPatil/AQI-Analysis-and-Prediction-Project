"""
run_pipeline_launcher.py
Forces Python 3.13 (which has all dependencies) to run the pipeline.
"""
import subprocess, os
from pathlib import Path

PYTHON313 = r"C:\Users\Rakshitha Patil\AppData\Local\Programs\Python\Python313\python.exe"
ROOT = Path(__file__).resolve().parent
os.chdir(ROOT)
os.environ["PYTHONPATH"] = str(ROOT)

subprocess.run([PYTHON313, "run_pipeline.py"], check=True)

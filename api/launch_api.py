"""
api/launch_api.py
Launches the FastAPI service using the correct Python (3.13).
Run:  python api/launch_api.py
"""
import subprocess, sys, os
from pathlib import Path

PYTHON313 = r"C:\Users\Rakshitha Patil\AppData\Local\Programs\Python\Python313\python.exe"
ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)
os.environ["PYTHONPATH"] = str(ROOT)

print("Starting AQI Prediction API on http://localhost:8000")
print("Interactive docs at http://localhost:8000/docs")
subprocess.run([
    PYTHON313, "-m", "uvicorn", "api.main:app",
    "--host", "0.0.0.0",
    "--port", "8000",
    "--reload",
], check=True)

"""
launch.py  —  starts the Streamlit dashboard programmatically
Run:  python launch.py
"""
import subprocess, sys, os

PYTHON313 = r"C:\Users\Rakshitha Patil\AppData\Local\Programs\Python\Python313\python.exe"

os.chdir(r"C:\Users\Rakshitha Patil\air_quality\aqi_prediction")
os.environ["PYTHONPATH"] = r"C:\Users\Rakshitha Patil\air_quality\aqi_prediction"

subprocess.run([
    PYTHON313, "-m", "streamlit", "run",
    "dashboard/app.py",
    "--server.port", "8501",
    "--browser.gatherUsageStats", "false",
], check=True)

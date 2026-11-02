"""
run_tests.py
Runs all tests using the correct Python 3.13 interpreter.
Usage:  python run_tests.py
"""
import subprocess, os, sys
from pathlib import Path

PYTHON313 = r"C:\Users\Rakshitha Patil\AppData\Local\Programs\Python\Python313\python.exe"
ROOT = Path(__file__).resolve().parent
os.chdir(ROOT)
os.environ["PYTHONPATH"] = str(ROOT)

result = subprocess.run([
    PYTHON313, "-m", "pytest",
    "tests/test_transformation.py",
    "tests/test_features.py",
    "tests/test_ingestion.py",
    "tests/test_api.py",
    "-v", "--tb=short"
], cwd=str(ROOT))

sys.exit(result.returncode)

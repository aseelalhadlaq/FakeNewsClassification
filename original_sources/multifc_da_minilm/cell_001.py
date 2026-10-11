# ============================================================
# 0. Install dependencies if needed
# ============================================================
import sys, subprocess, importlib

required_packages = {
    "sentence_transformers": "sentence-transformers",
    "torch": "torch",
    "sklearn": "scikit-learn",
    "pandas": "pandas",
    "numpy": "numpy",
    "matplotlib": "matplotlib",
}

for import_name, pip_name in required_packages.items():
    try:
        importlib.import_module(import_name)
    except Exception:
        print(f"Installing {pip_name}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pip_name])
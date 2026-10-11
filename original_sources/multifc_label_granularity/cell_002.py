# ============================================================
# 0a. Install dependencies if needed
# ============================================================
import sys, subprocess, importlib

_required_packages = {
    "sentence_transformers": "sentence-transformers",
    "torch": "torch",
    "sklearn": "scikit-learn",
    "pandas": "pandas",
    "numpy": "numpy",
    "scipy": "scipy",
}
for _imp, _pip in _required_packages.items():
    try:
        importlib.import_module(_imp)
    except Exception:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", _pip])
        print(f"installed {_pip}")
print("Dependencies ready.")

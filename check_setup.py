"""Step 1 check: verifies the project structure and required libraries."""
import importlib
import os
import sys

print("Python version:", sys.version.split()[0])

# 1. Check that all libraries can be imported
libraries = ["pandas", "numpy", "sklearn", "nltk", "matplotlib", "seaborn", "streamlit", "joblib"]
all_ok = True
for lib in libraries:
    try:
        module = importlib.import_module(lib)
        print(f"[OK]      {lib} {getattr(module, '__version__', '')}")
    except ImportError:
        print(f"[MISSING] {lib}")
        all_ok = False

# 2. Check that all required folders and files exist
required_paths = [
    "data/raw", "data/processed", "notebooks", "models",
    "outputs/figures", "outputs/reports",
    "src/__init__.py", "src/preprocessing.py", "src/train.py",
    "src/evaluate.py", "src/predict.py",
    "app.py", "requirements.txt", "README.md", ".gitignore",
]
for path in required_paths:
    if os.path.exists(path):
        print(f"[OK]      {path}")
    else:
        print(f"[MISSING] {path}")
        all_ok = False

print("\nSETUP TEST PASSED" if all_ok else "\nSETUP TEST FAILED - fix the [MISSING] items above")
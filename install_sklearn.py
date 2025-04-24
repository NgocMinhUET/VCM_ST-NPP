#!/usr/bin/env python
"""
Simple script to install scikit-learn.
Run this if you encounter the 'ModuleNotFoundError: No module named 'sklearn'' error.
"""
import subprocess
import sys

print("Installing scikit-learn...")
try:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "scikit-learn"])
    print("scikit-learn installed successfully!")
    
    # Try importing to verify installation
    import sklearn
    print(f"scikit-learn version: {sklearn.__version__}")
    print("Installation verified!")
except Exception as e:
    print(f"Error installing scikit-learn: {e}")
    print("Please try running 'pip install scikit-learn' manually.") 
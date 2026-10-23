"""
AI-Based Predictive Maintenance and Failure Prevention System for Mechanical Systems.
Focused on Gearbox Vibration Monitoring & Condition Classification.
"""

import os

# Prevent OpenMP multiple runtime conflict between PyTorch and XGBoost on macOS
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

__version__ = "0.1.0"

"""
Stem Estimation Workflow
Mixture-of-Experts CNN for large-scale tree stem enumeration from NAIP + 3DEP LiDAR.
"""

__version__ = "1.0.0"
__author__ = "Refactored from Ardohain, Willsey & Fei (2026)"

from .pipeline import run_pipeline

__all__ = ["run_pipeline"]
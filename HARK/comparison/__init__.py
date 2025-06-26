"""
Model comparison infrastructure for HARK.

This module provides tools for comparing different solution methods for
heterogeneous agent models, including metrics, adapters, and unified interfaces.
"""

from .base import ModelComparison
from .metrics import EconomicMetrics

__all__ = ["ModelComparison", "EconomicMetrics"]

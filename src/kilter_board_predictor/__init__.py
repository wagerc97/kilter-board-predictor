"""
Kilter Board Predictor - A machine learning project for predicting climbing route grades.
"""

__version__ = "0.1.0"
__author__ = "Kilter Board Predictor Team"

from .data_exploration import DataExplorer
from .statistics import BasicStatistics

__all__ = ["DataExplorer", "BasicStatistics"]
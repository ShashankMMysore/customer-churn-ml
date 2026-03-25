"""
Customer Churn ML - E-commerce classification project
"""

__version__ = "1.0.0"
__author__ = "Your Name"

from .data_loader import load_data
from .preprocessor import ChurnPreprocessor
from .model_training import train_models
from .model_evaluation import evaluate_models

__all__ = [
    'load_data',
    'ChurnPreprocessor',
    'train_models',
    'evaluate_models'
]

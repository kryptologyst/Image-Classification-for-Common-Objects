"""
Image Classification for Common Objects

A modern, production-ready image classification system supporting multiple
model architectures with comprehensive evaluation and visualization tools.
"""

__version__ = "1.0.0"
__author__ = "AI Projects Team"
__email__ = "team@aiprojects.com"

from .config import Config
from .logger import setup_logging, get_logger
from .data_loader import DataLoader
from .models import ModelFactory, CNNModelBuilder
from .trainer import Trainer
from .visualizer import Visualizer

__all__ = [
    "Config",
    "setup_logging",
    "get_logger", 
    "DataLoader",
    "ModelFactory",
    "CNNModelBuilder",
    "Trainer",
    "Visualizer"
]

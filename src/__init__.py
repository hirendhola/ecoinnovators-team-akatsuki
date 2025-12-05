"""
Solar Panel Detection System
"""

__version__ = "1.0.0"
__author__ = "Solar Detection Team"

from .pipeline import SolarDetectionPipeline
from .config import Config

__all__ = ['SolarDetectionPipeline', 'Config']
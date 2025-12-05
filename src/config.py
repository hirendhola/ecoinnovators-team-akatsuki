"""
Configuration module for solar panel detection system.
Loads settings from environment variables with sensible defaults.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()


class Config:
    # API Configuration
    GOOGLE_MAPS_API_KEY = os.getenv('GOOGLE_MAPS_API_KEY', '')
    
    # Model Configuration
    MODEL_PATH = os.getenv('MODEL_PATH', 'model/best.pt')
    CONFIDENCE_THRESHOLD = float(os.getenv('CONFIDENCE_THRESHOLD', 0.25))
    
    # Directory Paths
    INPUT_FOLDER = Path(os.getenv('INPUT_FOLDER', 'input_folder'))
    OUTPUT_FOLDER = Path(os.getenv('OUTPUT_FOLDER', 'predictions'))
    INPUT_FILENAME = "input_data.xlsx"
    
    # Image Settings
    MAP_REQUEST_SIZE = "512x512"
    MAP_SCALE = 2
    FINAL_IMAGE_SIZE = 1024
    ZOOM_LEVEL = int(os.getenv('ZOOM_LEVEL', 20))
    
    # Detection Settings
    SLICE_HEIGHT = 640
    SLICE_WIDTH = 640
    OVERLAP_RATIO = 0.2
    
    # Buffer Zones (in square feet)
    BUFFER_RADIUS_1_SQFT = 1200
    BUFFER_RADIUS_2_SQFT = 2400
    
    # Unit Conversions
    SQFT_TO_SQM = 0.092903
    
    # Quality Control Thresholds
    BRIGHTNESS_THRESHOLD_LOW = 30
    BRIGHTNESS_THRESHOLD_HIGH = 225
    CLOUD_THRESHOLD = 0.7
    MIN_IMAGE_VARIANCE = 100
    MIN_OVERLAP_AREA = 10  # minimum pixel overlap to consider detection valid
    
    @classmethod
    def validate(cls):
        """Validate critical configuration values."""
        if not cls.GOOGLE_MAPS_API_KEY or cls.GOOGLE_MAPS_API_KEY == 'your_api_key_here':
            raise ValueError("Google Maps API Key not configured. Please set GOOGLE_MAPS_API_KEY in .env file")
        
        if not Path(cls.MODEL_PATH).exists() and not Path(cls.MODEL_PATH.replace("best.pt", "last.pt")).exists():
            raise FileNotFoundError(f"Model file not found: {cls.MODEL_PATH}")
    
    @classmethod
    def setup_directories(cls):
        """Create necessary directories if they don't exist."""
        cls.INPUT_FOLDER.mkdir(exist_ok=True)
        cls.OUTPUT_FOLDER.mkdir(exist_ok=True)
        (cls.OUTPUT_FOLDER / '.gitkeep').touch()
    
    @classmethod
    def get_model_path(cls):
        """Get the actual model path, checking fallback options."""
        if Path(cls.MODEL_PATH).exists():
            return cls.MODEL_PATH
        
        fallback = cls.MODEL_PATH.replace("best.pt", "last.pt")
        if Path(fallback).exists():
            return fallback
        
        raise FileNotFoundError(f"Model not found at {cls.MODEL_PATH} or {fallback}")
"""
Image quality assessment and preprocessing.
"""

import cv2
import numpy as np
import logging

logger = logging.getLogger(__name__)


class ImageQualityChecker:
    def __init__(self, brightness_low, brightness_high, cloud_threshold, min_variance):
        self.brightness_low = brightness_low
        self.brightness_high = brightness_high
        self.cloud_threshold = cloud_threshold
        self.min_variance = min_variance
    
    def check_quality(self, image_path):
        """
        Assess if image quality is sufficient for detection.
        
        Returns:
            tuple: (is_verifiable: bool, reason: str)
        """
        try:
            img = cv2.imread(str(image_path))
            if img is None:
                return False, "Failed to load image"
            
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Check for darkness (shadows/poor lighting)
            mean_brightness = np.mean(gray)
            if mean_brightness < self.brightness_low:
                return False, "Image too dark (shadows/poor lighting)"
            
            # Check for cloud cover
            bright_pixels = np.sum(gray > self.brightness_high)
            bright_ratio = bright_pixels / gray.size
            
            if bright_ratio > self.cloud_threshold:
                return False, "Heavy cloud cover detected"
            
            # Check image detail/variance
            variance = np.var(gray)
            if variance < self.min_variance:
                return False, "Low image detail (possibly occluded)"
            
            return True, "Good quality"
            
        except Exception as e:
            logger.error(f"Error during quality check: {e}")
            return False, f"Quality check error: {str(e)}"
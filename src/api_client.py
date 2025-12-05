"""
Google Maps API client for fetching satellite imagery.
"""

import requests
import cv2
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class GoogleMapsClient:
    BASE_URL = "https://maps.googleapis.com/maps/api/staticmap"
    
    def __init__(self, api_key, zoom_level, image_size, map_scale):
        self.api_key = api_key
        self.zoom_level = zoom_level
        self.image_size = image_size
        self.map_scale = map_scale
        self.request_size = f"{image_size//map_scale}x{image_size//map_scale}"
    
    def download_satellite_image(self, lat, lon, sample_id, output_folder):
        """
        Download satellite image for given coordinates.
        
        Args:
            lat: Latitude
            lon: Longitude
            sample_id: Unique identifier for this location
            output_folder: Directory to save image
            
        Returns:
            Path to saved image file, or None if failed
        """
        params = {
            "center": f"{lat},{lon}",
            "zoom": self.zoom_level,
            "size": self.request_size,
            "scale": self.map_scale,
            "maptype": "satellite",
            "key": self.api_key
        }
        
        try:
            logger.info(f"Fetching satellite image for sample {sample_id}")
            response = requests.get(self.BASE_URL, params=params, timeout=15)
            
            if response.status_code != 200:
                logger.error(f"API returned status {response.status_code} for sample {sample_id}")
                return None
            
            filename = Path(output_folder) / f"{sample_id}.jpg"
            with open(filename, "wb") as f:
                f.write(response.content)
            
            # Ensure image is correct size
            # call _resize_and_filter if want to check dark area / low light check intentionally 
            self._resize_if_needed(filename)
            
            return str(filename)
            
        except requests.exceptions.Timeout:
            logger.error(f"Request timeout for sample {sample_id}")
            return None
        except Exception as e:
            logger.error(f"Failed to download image for sample {sample_id}: {e}")
            return None
    
    def _resize_if_needed(self, image_path):
        """Resize image to target size if needed."""
        img = cv2.imread(str(image_path))
        if img is None:
            return
        
        h, w = img.shape[:2]
        if h != self.image_size or w != self.image_size:
            img = cv2.resize(img, (self.image_size, self.image_size))
            cv2.imwrite(str(image_path), img)
            
    def _resize_and_filter(self, image_path):
        """Resize image AND apply testing filters."""
        img = cv2.imread(str(image_path))
        if img is None:
            return
        
        logger.warning(f"TESTING: Applying darkness filter to {image_path}")
        img = cv2.convertScaleAbs(img, alpha=0.2, beta=0)

        h, w = img.shape[:2]
        
        if h != self.image_size or w != self.image_size:
            img = cv2.resize(img, (self.image_size, self.image_size))
            
        cv2.imwrite(str(image_path), img)   
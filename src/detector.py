"""
Solar panel detection using SAHI (Slicing Aided Hyper Inference).
"""

import logging
from shapely.geometry import Polygon
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction

logger = logging.getLogger(__name__)


class SolarPanelDetector:
    def __init__(self, model_path, confidence_threshold, slice_size, overlap_ratio):
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.slice_height = slice_size
        self.slice_width = slice_size
        self.overlap_ratio = overlap_ratio
        self.model = None
    
    def initialize(self):
        """Load detection model."""
        device = self._get_device()
        
        logger.info(f"Loading model from {self.model_path}")
        self.model = AutoDetectionModel.from_pretrained(
            model_type='yolov8',
            model_path=self.model_path,
            confidence_threshold=self.confidence_threshold,
            device=device
        )
        logger.info(f"Model loaded successfully on {device}")
    
    def detect(self, image_path):
        """
        Run detection on image to find solar panels.
        
        Args:
            image_path: Path to satellite image
            
        Returns:
            List of (polygon, confidence) tuples
        """
        if self.model is None:
            raise RuntimeError("Model not initialized. Call initialize() first.")
        
        logger.info(f"Running detection on {image_path}")
        
        try:
            result = get_sliced_prediction(
                str(image_path),
                self.model,
                slice_height=self.slice_height,
                slice_width=self.slice_width,
                overlap_height_ratio=self.overlap_ratio,
                overlap_width_ratio=self.overlap_ratio
            )
            
            polygons = []
            for pred in result.object_prediction_list:
                if pred.score.value < self.confidence_threshold:
                    continue
                
                poly = self._extract_polygon(pred)
                if poly is not None:
                    polygons.append((poly, pred.score.value))
            
            logger.info(f"Detected {len(polygons)} solar panels")
            return polygons
            
        except Exception as e:
            logger.error(f"Detection failed: {e}")
            return []
    
    def _extract_polygon(self, prediction):
        """Extract polygon from prediction, trying mask first then bbox."""
        # Try mask-based polygon
        if hasattr(prediction, 'mask') and prediction.mask is not None:
            try:
                mask_points = prediction.mask.to_coco_segmentation()[0]
                coords = [(mask_points[i], mask_points[i+1]) 
                         for i in range(0, len(mask_points), 2)]
                if len(coords) >= 3:
                    poly = Polygon(coords)
                    if poly.is_valid:
                        return poly
            except:
                pass
        
        # Fall back to bounding box
        bbox = prediction.bbox
        x1, y1, x2, y2 = bbox.minx, bbox.miny, bbox.maxx, bbox.maxy
        return Polygon([(x1, y1), (x2, y1), (x2, y2), (x1, y2)])
    
    @staticmethod
    def _get_device():
        """Determine if GPU is available."""
        import os
        if os.system("nvidia-smi > /dev/null 2>&1") == 0:
            return "cuda:0"
        return "cpu"
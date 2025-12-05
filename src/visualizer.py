"""
Visualization of detection results on satellite imagery.
"""

import cv2
import numpy as np
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class DetectionVisualizer:
    def __init__(self):
        self.overlay_alpha = 0.4
        self.original_alpha = 0.6
    
    def draw_results(self, image_path, all_polygons, best_poly, center_px, 
                     buffer_radius_px, buffer_sqft, output_path):
        """
        Create visualization overlay showing detections and buffer zone.
        
        Args:
            image_path: Original satellite image
            all_polygons: All detected panels [(polygon, confidence), ...]
            best_poly: The selected best panel (or None)
            center_px: Center point (x, y)
            buffer_radius_px: Buffer zone radius in pixels
            buffer_sqft: Buffer area in square feet (for annotation)
            output_path: Where to save visualization
        """
        img = cv2.imread(str(image_path))
        if img is None:
            logger.error(f"Cannot load image: {image_path}")
            return
        
        overlay = img.copy()
        
        # Draw buffer zone circle
        self._draw_buffer_zone(overlay, center_px, buffer_radius_px)
        
        # Draw all detected panels in yellow
        self._draw_all_panels(overlay, all_polygons)
        
        # Highlight the selected panel in green
        if best_poly is not None:
            self._draw_selected_panel(overlay, best_poly)
        
        # Blend overlay with original
        result = cv2.addWeighted(overlay, self.overlay_alpha, img, self.original_alpha, 0)
        
        # Add text annotations
        self._add_annotations(result, buffer_sqft, best_poly is not None)
        
        cv2.imwrite(str(output_path), result)
        logger.info(f"Saved visualization: {output_path}")
    
    def _draw_buffer_zone(self, img, center, radius):
        """Draw buffer zone circle."""
        cv2.circle(img, (int(center[0]), int(center[1])), 
                  int(radius), (255, 255, 0), 2)
    
    def _draw_all_panels(self, img, polygons):
        """Draw all detected panels in yellow outline."""
        for poly, _ in polygons:
            coords = np.array(poly.exterior.coords, dtype=np.int32)
            cv2.polylines(img, [coords], True, (0, 255, 255), 1)
    
    def _draw_selected_panel(self, img, poly):
        """Draw the selected best panel with green fill and outline."""
        coords = np.array(poly.exterior.coords, dtype=np.int32)
        cv2.fillPoly(img, [coords], (0, 255, 0))
        cv2.polylines(img, [coords], True, (0, 255, 0), 3)
        
        # Add label
        centroid = poly.centroid
        cv2.putText(img, "TARGET PANEL", 
                   (int(centroid.x) - 50, int(centroid.y)),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    
    def _add_annotations(self, img, buffer_sqft, has_solar):
        """Add text annotations to image."""
        # Buffer info
        cv2.putText(img, f"Buffer: {buffer_sqft} sq.ft", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Detection status
        status_text = "SOLAR CONFIRMED" if has_solar else "NO SOLAR FOUND"
        color = (0, 255, 0) if has_solar else (0, 0, 255)
        cv2.putText(img, status_text, (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
"""
Geometric calculations and spatial operations.
"""

import math
import json
import logging
from shapely.geometry import Point, Polygon

logger = logging.getLogger(__name__)


def get_meters_per_pixel(lat, zoom, scale):
    """
    Calculate ground resolution at given latitude and zoom level.
    
    Args:
        lat: Latitude in degrees
        zoom: Map zoom level
        scale: Map scale factor
        
    Returns:
        Meters per pixel at the given location
    """
    return 156543.03392 * math.cos(math.radians(lat)) / (2 ** zoom) / scale


def buffer_radius_to_pixels(buffer_sqft, sqft_to_sqm, lat, zoom, scale):
    """
    Convert buffer radius from square feet to pixels.
    
    Args:
        buffer_sqft: Buffer area in square feet
        sqft_to_sqm: Conversion factor
        lat: Latitude for resolution calculation
        zoom: Zoom level
        scale: Scale factor
        
    Returns:
        Buffer radius in pixels
    """
    radius_meters = math.sqrt(buffer_sqft * sqft_to_sqm / math.pi)
    meters_per_px = get_meters_per_pixel(lat, zoom, scale)
    return radius_meters / meters_per_px


def find_best_panel(polygons, center_px, buffer_radius_px, min_overlap=10):
    """
    Find panel with maximum overlap within buffer zone.
    
    Args:
        polygons: List of (polygon, confidence) tuples
        center_px: Center point as (x, y) tuple
        buffer_radius_px: Buffer radius in pixels
        min_overlap: Minimum overlap area to consider valid
        
    Returns:
        tuple: (best_polygon, confidence, overlap_area)
    """
    if not polygons:
        return None, 0.0, 0.0
    
    center_point = Point(center_px)
    buffer_circle = center_point.buffer(buffer_radius_px)
    
    best_poly = None
    best_conf = 0.0
    max_overlap = 0.0
    
    for poly, conf in polygons:
        try:
            if buffer_circle.intersects(poly):
                intersection = buffer_circle.intersection(poly)
                overlap_area = intersection.area
                
                if overlap_area > max_overlap and overlap_area > min_overlap:
                    max_overlap = overlap_area
                    best_poly = poly
                    best_conf = conf
        except Exception as e:
            logger.warning(f"Error processing polygon: {e}")
            continue
    
    return best_poly, best_conf, max_overlap


def encode_polygon(poly):
    """
    Encode polygon coordinates to JSON string.
    
    Args:
        poly: Shapely Polygon object
        
    Returns:
        JSON string of coordinates, or empty string if None
    """
    if poly is None:
        return ""
    coords = list(poly.exterior.coords)
    return json.dumps([[round(x, 2), round(y, 2)] for x, y in coords])
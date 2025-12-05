"""
EcoInnovators Ideathon 2026 - Rooftop Solar Detection Pipeline
This script performs automated rooftop solar panel detection using satellite imagery.
"""

import os
import json
import math
import requests
import cv2
import pandas as pd
import numpy as np
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
from shapely.geometry import Polygon, Point
from datetime import datetime
import logging
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

GOOGLE_MAPS_API_KEY = os.getenv('GOOGLE_MAPS_API_KEY', 'AIzaSyBvSZJPZEhV6tb-DD-MKYFBJUeMAMmjxFM')
MODEL_PATH = os.getenv('MODEL_PATH', 'best.pt')

INPUT_FOLDER = os.getenv('INPUT_FOLDER', 'input_folder')
OUTPUT_FOLDER = os.getenv('OUTPUT_FOLDER', 'output_folder')
INPUT_FILENAME = "input_data.xlsx"

MAP_REQUEST_SIZE = "512x512"
MAP_SCALE = 2
FINAL_IMAGE_SIZE = 1024

ZOOM_LEVEL = int(os.getenv('ZOOM_LEVEL', 20))
CONFIDENCE_THRESHOLD = float(os.getenv('CONFIDENCE_THRESHOLD', 0.25))
SQFT_TO_SQM = 0.092903
BUFFER_RADIUS_1_SQFT = 1200
BUFFER_RADIUS_2_SQFT = 2400

SLICE_HEIGHT = 640
SLICE_WIDTH = 640
OVERLAP_RATIO = 0.2

# Image quality thresholds for QC
BRIGHTNESS_THRESHOLD_LOW = 30
BRIGHTNESS_THRESHOLD_HIGH = 225
CLOUD_THRESHOLD = 0.7


def check_api_key():
    """Validate that API key is configured."""
    if not GOOGLE_MAPS_API_KEY or GOOGLE_MAPS_API_KEY == 'your_api_key_here':
        logger.error("="*80)
        logger.error("ERROR: Google Maps API Key not configured!")
        logger.error("Please set GOOGLE_MAPS_API_KEY in .env file")
        logger.error("="*80)
        return False
    return True


def get_meters_per_pixel(lat, zoom, scale):
    """Calculate meters per pixel at given latitude and zoom level."""
    return 156543.03392 * math.cos(math.radians(lat)) / (2 ** zoom) / scale


def check_image_quality(image_path):
    """
    Check if image quality is sufficient for verification.
    Returns: (is_verifiable, reason)
    """
    try:
        img = cv2.imread(image_path)
        if img is None:
            return False, "Failed to load image"
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        mean_brightness = np.mean(gray)
        if mean_brightness < BRIGHTNESS_THRESHOLD_LOW:
            return False, "Image too dark (shadows/poor lighting)"
        
        bright_pixels = np.sum(gray > BRIGHTNESS_THRESHOLD_HIGH)
        total_pixels = gray.size
        bright_ratio = bright_pixels / total_pixels
        
        if bright_ratio > CLOUD_THRESHOLD:
            return False, "Heavy cloud cover detected"
        
        variance = np.var(gray)
        if variance < 100:
            return False, "Low image detail (possibly occluded)"
        
        return True, "Good quality"
        
    except Exception as e:
        logger.error(f"Error checking image quality: {e}")
        return False, f"Quality check error: {str(e)}"


def download_satellite_image(lat, lon, sample_id, folder):
    """Download satellite image from Google Maps Static API."""
    base_url = "https://maps.googleapis.com/maps/api/staticmap"
    params = {
        "center": f"{lat},{lon}",
        "zoom": ZOOM_LEVEL,
        "size": MAP_REQUEST_SIZE,
        "scale": MAP_SCALE,
        "maptype": "satellite",
        "key": GOOGLE_MAPS_API_KEY
    }
    
    try:
        logger.info(f"Downloading image for sample {sample_id}")
        response = requests.get(base_url, params=params, timeout=15)
        
        if response.status_code != 200:
            logger.error(f"API returned status {response.status_code}")
            return None
        
        filename = os.path.join(folder, f"{sample_id}.jpg")
        with open(filename, "wb") as f:
            f.write(response.content)
        
        img = cv2.imread(filename)
        if img is not None:
            h, w, _ = img.shape
            if h != FINAL_IMAGE_SIZE or w != FINAL_IMAGE_SIZE:
                img = cv2.resize(img, (FINAL_IMAGE_SIZE, FINAL_IMAGE_SIZE))
                cv2.imwrite(filename, img)
        
        return filename
        
    except requests.exceptions.Timeout:
        logger.error(f"Timeout downloading image for {sample_id}")
        return None
    except Exception as e:
        logger.error(f"Error downloading {sample_id}: {e}")
        return None


def encode_polygon(poly):
    """Encode polygon coordinates to string format."""
    if poly is None:
        return ""
    coords = list(poly.exterior.coords)
    return json.dumps([[round(x, 2), round(y, 2)] for x, y in coords])


def sahi_inference(detection_model, image_path):
    """Run SAHI inference on image to detect solar panels."""
    logger.info(f"Running SAHI inference on {image_path}")
    
    try:
        result = get_sliced_prediction(
            image_path,
            detection_model,
            slice_height=SLICE_HEIGHT,
            slice_width=SLICE_WIDTH,
            overlap_height_ratio=OVERLAP_RATIO,
            overlap_width_ratio=OVERLAP_RATIO
        )
        
        polygons = []
        for pred in result.object_prediction_list:
            if pred.score.value < CONFIDENCE_THRESHOLD:
                continue
            
            bbox = pred.bbox
            x1, y1, x2, y2 = bbox.minx, bbox.miny, bbox.maxx, bbox.maxy
            
            # Try to use mask if available
            if hasattr(pred, 'mask') and pred.mask is not None:
                try:
                    mask_points = pred.mask.to_coco_segmentation()[0]
                    coords = [(mask_points[i], mask_points[i+1]) 
                             for i in range(0, len(mask_points), 2)]
                    if len(coords) >= 3:
                        poly = Polygon(coords)
                        if poly.is_valid:
                            polygons.append((poly, pred.score.value))
                            continue
                except:
                    pass
            
            poly = Polygon([(x1, y1), (x2, y1), (x2, y2), (x1, y2)])
            polygons.append((poly, pred.score.value))
        
        logger.info(f"Detected {len(polygons)} potential panels")
        return polygons
        
    except Exception as e:
        logger.error(f"SAHI inference error: {e}")
        return []


def find_best_panel(polygons, center_px, buffer_radius_px):
    """
    Find the panel with maximum overlap with buffer zone.
    Returns: (best_polygon, confidence, overlap_area)
    """
    if not polygons:
        return None, 0.0, 0.0
    
    center_point = Point(center_px)
    buffer_circle = center_point.buffer(buffer_radius_px)
    
    best_poly = None
    best_conf = 0.0
    max_overlap_area = 0.0
    
    for poly, conf in polygons:
        try:
            if buffer_circle.intersects(poly):
                intersection = buffer_circle.intersection(poly)
                overlap_area = intersection.area
                
                if overlap_area > max_overlap_area:
                    max_overlap_area = overlap_area
                    best_poly = poly
                    best_conf = conf
        except Exception as e:
            logger.warning(f"Error processing polygon: {e}")
            continue
    
    return best_poly, best_conf, max_overlap_area


def draw_artifacts(image_path, all_polygons, best_poly, center_px, 
                   radius_px, buffer_sqft, output_path):
    """Draw visualization overlay on satellite image."""
    img = cv2.imread(image_path)
    if img is None:
        logger.error(f"Cannot read image: {image_path}")
        return
    
    overlay = img.copy()
    
    cv2.circle(overlay, (int(center_px[0]), int(center_px[1])), 
               int(radius_px), (255, 255, 0), 2)
    
    for poly, conf in all_polygons:
        coords = np.array(poly.exterior.coords, dtype=np.int32)
        cv2.polylines(overlay, [coords], True, (0, 255, 255), 1)
    
    #  best panel in green
    if best_poly is not None:
        coords = np.array(best_poly.exterior.coords, dtype=np.int32)
        cv2.fillPoly(overlay, [coords], (0, 255, 0))
        cv2.polylines(overlay, [coords], True, (0, 255, 0), 3)
        
        centroid = best_poly.centroid
        cv2.putText(overlay, "TARGET PANEL", 
                   (int(centroid.x), int(centroid.y)),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    
    # overlay with original
    result = cv2.addWeighted(overlay, 0.4, img, 0.6, 0)
    
    # text annotations
    cv2.putText(result, f"Buffer: {buffer_sqft} sq.ft", 
               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    status_text = "SOLAR CONFIRMED" if best_poly else "NO SOLAR FOUND"
    color = (0, 255, 0) if best_poly else (0, 0, 255)
    cv2.putText(result, status_text, (10, 60), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    
    cv2.imwrite(output_path, result)
    logger.info(f"Saved overlay: {output_path}")


def validate_input_file(file_path):
    """Validate input Excel file format."""
    try:
        df = pd.read_excel(file_path)
        required_cols = ['sample_id', 'latitude', 'longitude']
        
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            logger.error(f"Missing required columns: {missing_cols}")
            return None
        
        logger.info(f"Input file validated: {len(df)} samples")
        return df
        
    except Exception as e:
        logger.error(f"Error reading input file: {e}")
        return None


def run_pipeline():
    """Main pipeline execution."""
    print("=" * 80)
    print(f"EcoInnovators 2026 - Solar Detection Pipeline")
    print(f"Resolution: {FINAL_IMAGE_SIZE}x{FINAL_IMAGE_SIZE}")
    print("=" * 80)
    
    if not check_api_key():
        print("\n Please configure your Google Maps API key in .env file")
        print("Copy .env.example to .env and add your API key")
        return
    
    os.makedirs(INPUT_FOLDER, exist_ok=True)
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    
    Path(OUTPUT_FOLDER, '.gitkeep').touch()
    
    input_path = os.path.join(INPUT_FOLDER, INPUT_FILENAME)
    
    if not os.path.exists(input_path):
        pd.DataFrame({
            'sample_id': [1001, 1002],
            'latitude': [23.908454, 28.7041],
            'longitude': [71.182617, 77.1025]
        }).to_excel(input_path, index=False)
        logger.info(f"Created sample input: {input_path}")
    
    df = validate_input_file(input_path)
    if df is None:
        logger.error("Invalid input file. Exiting.")
        return
    
    model_to_use = MODEL_PATH
    if not os.path.exists(MODEL_PATH):
        logger.warning(f"Model not found at {MODEL_PATH}")
        alt_path = MODEL_PATH.replace("best.pt", "last.pt")
        if os.path.exists(alt_path):
            model_to_use = alt_path
            logger.info(f"Using fallback: {alt_path}")
        else:
            logger.error("No model file found. Exiting.")
            return
    
    logger.info(f"Initializing SAHI with model: {model_to_use}")
    try:
        device = "cuda:0" if os.system("nvidia-smi > /dev/null 2>&1") == 0 else "cpu"
        detection_model = AutoDetectionModel.from_pretrained(
            model_type='yolov8',
            model_path=model_to_use,
            confidence_threshold=CONFIDENCE_THRESHOLD,
            device=device
        )
        logger.info(f"Model loaded successfully on {device}")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return
    
    results_list = []
    
    for index, row in df.iterrows():
        sample_id = row['sample_id']
        lat = float(row['latitude'])
        lon = float(row['longitude'])
        
        logger.info(f"[{index+1}/{len(df)}] Processing sample {sample_id}")
        
        img_path = download_satellite_image(lat, lon, sample_id, OUTPUT_FOLDER)
        if not img_path:
            logger.warning(f"Skipped sample {sample_id} - download failed")
            continue
        
        is_verifiable, quality_reason = check_image_quality(img_path)
        
        all_polygons = sahi_inference(detection_model, img_path)
        
        center_px = (FINAL_IMAGE_SIZE // 2, FINAL_IMAGE_SIZE // 2)
        meters_per_px = get_meters_per_pixel(lat, ZOOM_LEVEL, MAP_SCALE)
        
        radius_1_m = math.sqrt(BUFFER_RADIUS_1_SQFT * SQFT_TO_SQM / math.pi)
        radius_1_px = radius_1_m / meters_per_px
        
        radius_2_m = math.sqrt(BUFFER_RADIUS_2_SQFT * SQFT_TO_SQM / math.pi)
        radius_2_px = radius_2_m / meters_per_px
        
        final_has_solar = False
        final_buffer_sqft = 2400
        final_area_sqm = 0.0
        final_confidence = 0.0
        chosen_poly = None
        chosen_radius_px = radius_2_px
        
        if len(all_polygons) > 0:
            poly_1200, conf_1200, overlap_1200 = find_best_panel(
                all_polygons, center_px, radius_1_px
            )
            
            if overlap_1200 > 10:  
                final_has_solar = True
                final_buffer_sqft = 1200
                final_area_sqm = poly_1200.area * (meters_per_px ** 2)
                final_confidence = conf_1200
                chosen_poly = poly_1200
                chosen_radius_px = radius_1_px
                logger.info(f"Solar CONFIRMED in 1200 sq.ft buffer")
            else:
                poly_2400, conf_2400, overlap_2400 = find_best_panel(
                    all_polygons, center_px, radius_2_px
                )
                
                if overlap_2400 > 10:
                    final_has_solar = True
                    final_buffer_sqft = 2400
                    final_area_sqm = poly_2400.area * (meters_per_px ** 2)
                    final_confidence = conf_2400
                    chosen_poly = poly_2400
                    chosen_radius_px = radius_2_px
                    logger.info(f"Solar CONFIRMED in 2400 sq.ft buffer")
                else:
                    logger.info("Panels detected but outside buffer zones")
        else:
            logger.info("No panels detected")
        
        if not is_verifiable:
            qc_status = "NOT_VERIFIABLE"
            logger.warning(f"Image quality issue: {quality_reason}")
        else:
            qc_status = "VERIFIABLE"
        
        output_record = {
            "sample_id": int(sample_id),
            "lat": lat,
            "lon": lon,
            "has_solar": final_has_solar,
            "confidence": round(final_confidence, 4),
            "pv_area_sqm_est": round(final_area_sqm, 2),
            "buffer_radius_sqft": final_buffer_sqft,
            "qc_status": qc_status,
            "bbox_or_mask": encode_polygon(chosen_poly),
            "image_metadata": {
                "source": "Google Maps Static API",
                "size": f"{FINAL_IMAGE_SIZE}x{FINAL_IMAGE_SIZE}",
                "meters_per_px": round(meters_per_px, 5),
                "quality_check": quality_reason
            }
        }
        
        json_filename = os.path.join(OUTPUT_FOLDER, f"{sample_id}.json")
        with open(json_filename, 'w') as f:
            json.dump(output_record, f, indent=4)
        
        # Draw overlay
        overlay_filename = os.path.join(OUTPUT_FOLDER, f"{sample_id}_overlay.png")
        draw_artifacts(img_path, all_polygons, chosen_poly, center_px, 
                      chosen_radius_px, final_buffer_sqft, overlay_filename)
        
        results_list.append(output_record)
        print()
    
    combined_json = os.path.join(OUTPUT_FOLDER, "all_predictions.json")
    with open(combined_json, 'w') as f:
        json.dump(results_list, f, indent=4)
    
    print("=" * 80)
    print("PIPELINE COMPLETE")
    print(f"Total Processed: {len(results_list)}")
    print(f"Solar Found: {sum(1 for r in results_list if r['has_solar'])}")
    print(f"Verifiable: {sum(1 for r in results_list if r['qc_status'] == 'VERIFIABLE')}")
    print(f"Not Verifiable: {sum(1 for r in results_list if r['qc_status'] == 'NOT_VERIFIABLE')}")
    print("=" * 80)


if __name__ == "__main__":
    run_pipeline()
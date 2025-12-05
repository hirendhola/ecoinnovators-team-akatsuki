import json
import logging
import pandas as pd
from pathlib import Path

from .config import Config
from .api_client import GoogleMapsClient
from .image_processor import ImageQualityChecker
from .detector import SolarPanelDetector
from .geometry import get_meters_per_pixel, buffer_radius_to_pixels, find_best_panel, encode_polygon
from .visualizer import DetectionVisualizer

logger = logging.getLogger(__name__)


class SolarDetectionPipeline:
    def __init__(self):
        self.config = Config
        
        # Initialize components
        self.maps_client = GoogleMapsClient(
            api_key=Config.GOOGLE_MAPS_API_KEY,
            zoom_level=Config.ZOOM_LEVEL,
            image_size=Config.FINAL_IMAGE_SIZE,
            map_scale=Config.MAP_SCALE
        )
        
        self.quality_checker = ImageQualityChecker(
            brightness_low=Config.BRIGHTNESS_THRESHOLD_LOW,
            brightness_high=Config.BRIGHTNESS_THRESHOLD_HIGH,
            cloud_threshold=Config.CLOUD_THRESHOLD,
            min_variance=Config.MIN_IMAGE_VARIANCE
        )
        
        self.detector = SolarPanelDetector(
            model_path=Config.get_model_path(),
            confidence_threshold=Config.CONFIDENCE_THRESHOLD,
            slice_size=Config.SLICE_HEIGHT,
            overlap_ratio=Config.OVERLAP_RATIO
        )
        
        self.visualizer = DetectionVisualizer()
    
    def run(self):
        """Execute the complete detection pipeline."""
        self._print_header()
        
        Config.validate()
        Config.setup_directories()
        
        input_path = Config.INPUT_FOLDER / Config.INPUT_FILENAME
        df = self._load_or_create_input(input_path)
        
        if df is None:
            logger.error("Failed to load input data")
            return
        
        self.detector.initialize()
        
        results = []
        for index, row in df.iterrows():
            result = self._process_sample(row, index + 1, len(df))
            if result:
                results.append(result)
        
        self._save_combined_results(results)
        self._print_summary(results)
        
    def _process_sample(self, row, current, total):
        """Process a single location sample."""
        sample_id = row['sample_id']
        lat = float(row['latitude'])
        lon = float(row['longitude'])
        
        sample_folder = Config.OUTPUT_FOLDER / str(sample_id)
        sample_folder.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"[{current}/{total}] Processing sample {sample_id}")
        
        image_path = self.maps_client.download_satellite_image(
            lat, lon, sample_id, sample_folder
        )
        
        if not image_path:
            logger.warning(f"Skipped sample {sample_id} - download failed")
            return None
        
        is_verifiable, quality_reason = self.quality_checker.check_quality(image_path)
        
        all_polygons = self.detector.detect(image_path)
        
        center_px = (Config.FINAL_IMAGE_SIZE // 2, Config.FINAL_IMAGE_SIZE // 2)
        meters_per_px = get_meters_per_pixel(lat, Config.ZOOM_LEVEL, Config.MAP_SCALE)
        
        radius_1_px = buffer_radius_to_pixels(
            Config.BUFFER_RADIUS_1_SQFT, Config.SQFT_TO_SQM,
            lat, Config.ZOOM_LEVEL, Config.MAP_SCALE
        )
        
        radius_2_px = buffer_radius_to_pixels(
            Config.BUFFER_RADIUS_2_SQFT, Config.SQFT_TO_SQM,
            lat, Config.ZOOM_LEVEL, Config.MAP_SCALE
        )
        
        result_data = self._find_best_detection(
            all_polygons, center_px, radius_1_px, radius_2_px, meters_per_px
        )
        
        output_record = {
            "sample_id": int(sample_id),
            "lat": lat,
            "lon": lon,
            "has_solar": result_data['has_solar'],
            "confidence": round(result_data['confidence'], 4),
            "pv_area_sqm_est": round(result_data['area_sqm'], 2),
            "buffer_radius_sqft": result_data['buffer_sqft'],
            "qc_status": "VERIFIABLE" if is_verifiable else "NOT_VERIFIABLE",
            "bbox_or_mask": encode_polygon(result_data['polygon']),
            "image_metadata": {
                "source": "Google Maps Static API",
                "size": f"{Config.FINAL_IMAGE_SIZE}x{Config.FINAL_IMAGE_SIZE}",
                "meters_per_px": round(meters_per_px, 5),
                "quality_check": quality_reason
            }
        }
        
        json_path = sample_folder / f"{sample_id}.json"
        with open(json_path, 'w') as f:
            json.dump(output_record, f, indent=4)
        
        overlay_path = sample_folder / f"{sample_id}_overlay.png"
        self.visualizer.draw_results(
            image_path, all_polygons, result_data['polygon'],
            center_px, result_data['radius_px'], result_data['buffer_sqft'],
            overlay_path
        )
        
        return output_record

    def _find_best_detection(self, polygons, center_px, radius_1_px, radius_2_px, meters_per_px):
        """Find the best panel detection within buffer zones."""
        result = {
            'has_solar': False,
            'buffer_sqft': Config.BUFFER_RADIUS_2_SQFT,
            'area_sqm': 0.0,
            'confidence': 0.0,
            'polygon': None,
            'radius_px': radius_2_px
        }
        
        if not polygons:
            logger.info("No panels detected")
            return result
        
        # Try smaller buffer first
        poly_1200, conf_1200, overlap_1200 = find_best_panel(
            polygons, center_px, radius_1_px, Config.MIN_OVERLAP_AREA
        )
        
        if poly_1200 is not None:
            result.update({
                'has_solar': True,
                'buffer_sqft': Config.BUFFER_RADIUS_1_SQFT,
                'area_sqm': poly_1200.area * (meters_per_px ** 2),
                'confidence': conf_1200,
                'polygon': poly_1200,
                'radius_px': radius_1_px
            })
            logger.info("Solar confirmed in 1200 sq.ft buffer")
        else:
            poly_2400, conf_2400, overlap_2400 = find_best_panel(
                polygons, center_px, radius_2_px, Config.MIN_OVERLAP_AREA
            )
            
            if poly_2400 is not None:
                result.update({
                    'has_solar': True,
                    'buffer_sqft': Config.BUFFER_RADIUS_2_SQFT,
                    'area_sqm': poly_2400.area * (meters_per_px ** 2),
                    'confidence': conf_2400,
                    'polygon': poly_2400,
                    'radius_px': radius_2_px
                })
                logger.info("Solar confirmed in 2400 sq.ft buffer")
            else:
                logger.info("Panels detected but outside buffer zones")
        
        return result
    
    def _load_or_create_input(self, input_path):
        """Load input file or create sample if doesn't exist."""
        if not input_path.exists():
            logger.info(f"Creating sample input file: {input_path}")
            df = pd.DataFrame({
                'sample_id': [1001, 1002],
                'latitude': [23.908454, 28.7041],
                'longitude': [71.182617, 77.1025]
            })
            df.to_excel(input_path, index=False)
        
        try:
            df = pd.read_excel(input_path)
            required_cols = ['sample_id', 'latitude', 'longitude']
            
            if not all(col in df.columns for col in required_cols):
                logger.error(f"Input file missing required columns: {required_cols}")
                return None
            
            logger.info(f"Loaded {len(df)} samples from input file")
            return df
            
        except Exception as e:
            logger.error(f"Failed to read input file: {e}")
            return None
    
    def _save_combined_results(self, results):
        """Save all results to a single JSON file."""
        output_path = Config.OUTPUT_FOLDER / "all_predictions.json"
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=4)
        logger.info(f"Saved combined results: {output_path}")
    
    def _print_header(self):
        """Print pipeline startup banner."""
        print("=" * 80)
        print("Solar Panel Detection Pipeline")
        print(f"Resolution: {Config.FINAL_IMAGE_SIZE}x{Config.FINAL_IMAGE_SIZE}")
        print("=" * 80)
    
    def _print_summary(self, results):
        """Print pipeline execution summary."""
        total = len(results)
        solar_found = sum(1 for r in results if r['has_solar'])
        verifiable = sum(1 for r in results if r['qc_status'] == 'VERIFIABLE')
        
        print("\n" + "=" * 80)
        print("PIPELINE COMPLETE")
        print(f"Total Processed: {total}")
        print(f"Solar Found: {solar_found}")
        print(f"Verifiable: {verifiable}")
        print(f"Not Verifiable: {total - verifiable}")
        print("=" * 80)
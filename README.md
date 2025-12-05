# EcoInnovators Ideathon 2026 - Rooftop Solar Detection

AI-powered pipeline for automated verification of rooftop solar panel installations using satellite imagery and deep learning.

## 🎯 Project Overview

This system automatically detects and verifies rooftop solar panel installations from satellite images at given coordinates. It's designed for governance and subsidy verification under India's PM Surya Ghar: Muft Bijli Yojana scheme.

### Key Features
- ✅ Automated satellite image retrieval via Google Maps API
- ✅ SAHI (Slicing Aided Hyper Inference) for accurate detection
- ✅ Dual buffer zone checking (1200 & 2400 sq.ft)
- ✅ Panel area estimation in square meters
- ✅ Quality control & verifiability assessment
- ✅ Audit-ready visualization overlays

## 📁 Repository Structure

```
├── pipeline_code/
│   └── inference.py          # Main inference script
├── environment_details/
│   ├── requirements.txt      # pip dependencies
│   ├── environment.yml       # conda environment
│   └── python_version.txt    # Python version info
├── trained_model/
│   └── best.pt              # YOLOv8 trained model
├── model_card/
│   └── model_card.pdf       # Model documentation
├── prediction_files/
│   └── all_predictions.json # Sample predictions
├── artifacts/
│   └── *_overlay.png        # Visualization outputs
├── model_training_logs/
│   └── training_logs.csv    # Training metrics
├── input_folder/
│   └── input_data.xlsx      # Sample input format
└── README.md
```

## 🚀 Quick Start

### Prerequisites

- Python 3.9 or higher
- CUDA-capable GPU (recommended but not required)
- Google Maps Static API key

### Installation

#### Option 1: Using pip

```bash
# Clone repository
git clone https://github.com/your-username/ecoinnovators-2026.git
cd ecoinnovators-2026

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r environment_details/requirements.txt
```

#### Option 2: Using conda

```bash
# Clone repository
git clone https://github.com/your-username/ecoinnovators-2026.git
cd ecoinnovators-2026

# Create conda environment
conda env create -f environment_details/environment.yml
conda activate ecoinnovators
```

### Configuration

1. **Add your Google Maps API Key:**
   
   Edit `pipeline_code/inference.py` and update:
   ```python
   GOOGLE_MAPS_API_KEY = "YOUR_API_KEY_HERE"
   ```

2. **Ensure model file exists:**
   
   Place your trained model file (`best.pt`) in the `trained_model/` directory.

### Running Inference

1. **Prepare input file:**

   Create an Excel file at `input_folder/input_data.xlsx` with columns:
   - `sample_id`: Unique identifier for each location
   - `latitude`: Latitude coordinate (WGS84)
   - `longitude`: Longitude coordinate (WGS84)

   Example:
   ```
   sample_id | latitude  | longitude
   ----------|-----------|----------
   1001      | 23.908454 | 71.182617
   1002      | 28.704100 | 77.102500
   ```

2. **Run the pipeline:**

   ```bash
   cd pipeline_code
   python inference.py
   ```

3. **Check outputs:**

   Results will be saved in `output_folder/`:
   - `{sample_id}.json` - Individual predictions
   - `{sample_id}_overlay.png` - Visualization with detections
   - `all_predictions.json` - Combined results

## 📊 Output Format

Each sample generates a JSON file with the following structure:

```json
{
    "sample_id": 1001,
    "lat": 23.908454,
    "lon": 71.182617,
    "has_solar": true,
    "confidence": 0.8945,
    "pv_area_sqm_est": 45.23,
    "buffer_radius_sqft": 1200,
    "qc_status": "VERIFIABLE",
    "bbox_or_mask": "[[x1,y1], [x2,y2], ...]",
    "image_metadata": {
        "source": "Google Maps Static API",
        "size": "1024x1024",
        "meters_per_px": 0.29858,
        "quality_check": "Good quality"
    }
}
```

### QC Status Values
- **VERIFIABLE**: Clear image with definitive result
- **NOT_VERIFIABLE**: Poor image quality (clouds, shadows, low resolution)

## 🔧 Configuration Parameters

Key parameters in `inference.py`:

| Parameter | Value | Description |
|-----------|-------|-------------|
| `ZOOM_LEVEL` | 20 | Google Maps zoom level |
| `FINAL_IMAGE_SIZE` | 1024 | Output image dimensions (pixels) |
| `CONFIDENCE_THRESHOLD` | 0.25 | Minimum detection confidence |
| `BUFFER_RADIUS_1_SQFT` | 1200 | First buffer zone (sq.ft) |
| `BUFFER_RADIUS_2_SQFT` | 2400 | Second buffer zone (sq.ft) |
| `SLICE_HEIGHT` | 640 | SAHI slice height |
| `SLICE_WIDTH` | 640 | SAHI slice width |
| `OVERLAP_RATIO` | 0.2 | SAHI overlap ratio |

## 📈 Model Performance

See `model_card/model_card.pdf` for detailed performance metrics including:
- F1 Score on validation set
- RMSE for area estimation
- Performance across different roof types
- Known limitations and failure modes

## 🛠️ Troubleshooting

### Common Issues

1. **Import Error: sahi not found**
   ```bash
   pip install sahi==0.11.14
   ```

2. **CUDA out of memory**
   - Reduce batch size or disable GPU in code
   - Set `device="cpu"` in model initialization

3. **API Key Error**
   - Verify your Google Maps API key is valid
   - Ensure Static Maps API is enabled in Google Cloud Console

4. **Model file not found**
   - Ensure `best.pt` is in `trained_model/` directory
   - Check file path in configuration

## 📝 Training Data Sources

Models were trained using the following datasets:
1. [Alfred Weber Institute Dataset](https://universe.roboflow.com/...) - 5,000 annotated images
2. [LSGI547 Project](https://universe.roboflow.com/...) - 3,200 images
3. [Piscinas Y Tenistable](https://universe.roboflow.com/...) - 1,800 images

All datasets are publicly available under permissive licenses.

## 📄 License

This project is licensed under the MIT License - see LICENSE file for details.

## 👥 Team

EcoInnovators 2026 Team
- Team Member 1
- Team Member 2
- Team Member 3

## 🙏 Acknowledgments

- PM Surya Ghar: Muft Bijli Yojana initiative
- Roboflow community for open datasets
- SAHI library developers
- Ultralytics YOLOv8 team

## 📧 Contact

For questions or issues, please open a GitHub issue or contact: your.email@example.com

---

**Note**: This is a competition submission for EcoInnovators Ideathon 2026. Submission deadline: December 7, 2025, 11:59 PM.
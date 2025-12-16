# External Scripts Review - INIA Testing YOLO Repository

**Source**: https://github.com/maaferna/INIA_Testing_YOLOV8
**Author**: maaferna (Miguel Fernández)
**Date Reviewed**: 2025-12-16
**Purpose**: Extract useful utilities for agricultural weed detection project

---

## Overview

This repository contains custom scripts for YOLOv8 training, validation, and inference with specific focus on:
- **Geospatial data integration** (UTM coordinates, GeoJSON generation)
- **ClearML experiment tracking** integration
- **SAHI (Sliced Inference)** for large images
- **Multi-GPU training orchestration**
- **Automated model selection** (best model finder)
- **Custom visualization** with minimalistic labels

---

## Script Inventory

### 1. `geo_data_utils.py` (6.0 KB) ⭐ **HIGH VALUE**

**Purpose**: Geospatial utilities for converting coordinates and generating geographic metadata.

**Key Functions**:
- `generate_geojson(image_name, metadata_path, output_dir, styled_image_path)`
  - Converts UTM coordinates to WGS84 (lat/lon) using pyproj
  - Creates GeoJSON files with detection metadata
  - Hardcoded for UTM Zone 19S (Chile)
  - Includes model info, confidence, label summaries

- `gather_unique_labels(batch_dir)`
  - Scans metadata JSON files to find all unique class labels

- `generate_batch_summary_csv(batch_dir, output_csv, results)`
  - Creates CSV with UTM coordinates, processing times, and per-class counts
  - Useful for batch processing reports

**Potential Uses**:
- ✅ **Integrate with your Santa Rosa georeferenced datasets**
- ✅ Generate GeoJSON outputs for QGIS/ArcGIS visualization
- ✅ Export detection results with spatial coordinates
- ⚠️ Need to adapt UTM zone (currently hardcoded to Zone 19S)

**Dependencies**: `pyproj`, `json`, `csv`

---

### 2. `utils.py` (43 KB) ⭐⭐⭐ **VERY HIGH VALUE**

**Purpose**: Massive utility module with visualization, image processing, and model management functions.

**Key Functions**:

#### Model Management:
- `download_yolo_model(yolo_version, model_version)` - Auto-download YOLO models if missing
- `get_best_model(project_root_dir, img_size, model_version)` - Find best trained model by mAP50

#### Visualization (Minimalistic Style):
- `apply_minimalistic_label_style_pil(image_path, bboxes, labels, output_path, colors, label_summary)`
  - PIL-based bounding box drawing with **only confidence values** (no class names on boxes)
  - Dynamic font sizing based on image dimensions
  - Semi-transparent label backgrounds
  - **Bottom legend showing class summary** (count per class)

- `apply_minimalistic_label_style_pil_sahi(...)` - SAHI-specific version

- `combine_frame_with_bottom_summary(image, summary_image)` - Append legend to image bottom

- `create_bottom_summary_image(label_summary, colors, labels, image_width)` - Generate legend bar

#### EXIF & Metadata:
- `get_image_metadata(image_path)` - Extract EXIF including GPS, altitude, drone metadata
- `dms_to_decimal(dms, ref)` - Convert GPS DMS to decimal degrees
- `extract_exif_metadata(image_path)` - Comprehensive EXIF extraction

#### Data Processing:
- `generate_json_output(...)` - Create structured JSON metadata for predictions
- `generate_summary_by_class(bboxes, labels)` - Count detections per class
- `numpy_to_native(obj)` - Convert numpy types to native Python for JSON serialization

#### Utilities:
- `convert_path_to_windows_format(path)` - Cross-platform path handling
- `format_time(seconds)` - Human-readable time formatting
- `safe_float(value, default=0.0)` - Safe type conversion

**Potential Uses**:
- ✅ **Adopt minimalistic visualization style** (cleaner than default YOLO boxes)
- ✅ Use best model finder for your multi-experiment setup
- ✅ Extract GPS/EXIF metadata from drone images
- ✅ Generate professional detection summaries with legends
- ✅ JSON metadata generation for reproducibility

**Dependencies**: `PIL`, `cv2`, `numpy`, `matplotlib`, `torch`, `pyproj`, `pandas`

---

### 3. `yolo_training.py` (6.3 KB) ⭐ **MODERATE VALUE**

**Purpose**: Multi-run training orchestrator with ClearML integration.

**Key Features**:
- `train_yolo(config, model_version, batch_size, img_size, devices, num_runs=5, ...)`
  - Trains same model **multiple times** (5 runs default) with different random seeds
  - Tracks best mAP50 across runs
  - **ClearML task creation** for each run
  - Multi-GPU support (non-DDP, single device selection)
  - Saves consolidated JSON summaries per run
  - Automatic timestamp-based organization

**Training Configuration**:
- Uses `cache=False`, `workers=8`, `amp=True`
- Seeds based on `time.time() * 1000` for uniqueness
- Saves: `summary.json`, `results.json`, `training_results_summary_{timestamp}.json`

**Potential Uses**:
- ⚠️ **Partially useful**: You already have `run_experiments.py` for multi-model training
- ✅ Could adopt the **multi-run strategy** (train same config 5x, pick best seed)
- ✅ ClearML integration if you want experiment tracking beyond TensorBoard
- ❌ Less relevant since you use DDP for dual GPU (this uses single device)

**Dependencies**: `ultralytics`, `clearml_utils`, `torch`

---

### 4. `predict_yolo.py` (32 KB) ⭐⭐ **HIGH VALUE**

**Purpose**: Advanced inference script with SAHI support and comprehensive output generation.

**Key Features**:
- `run_inference(img_size, model_version, test_images_dir, ...)`
  - **Standard YOLO inference** with confidence filtering
  - **SAHI sliced inference** support for large images
  - Automatic best model loading
  - Generates:
    - Styled images with minimalistic labels
    - Default YOLO annotated images
    - YOLO format label files (.txt)
    - JSON metadata per image
    - Batch CSV summary
    - GeoJSON files (if UTM metadata available)

- Multiple output directories organized by timestamp
- Extracts EXIF/GPS metadata from images
- Processing time tracking per image
- Comprehensive logging

**SAHI Integration**:
- `AutoDetectionModel` wrapper for YOLO
- Configurable slice dimensions (512x512, 640x640, etc.)
- Overlap ratio control
- Post-processing NMS

**Potential Uses**:
- ✅ **SAHI integration** - Essential for your 1024x1024 high-res images
- ✅ Adopt structured output organization (styled, default, labels, metadata)
- ✅ Batch CSV generation for experiment reports
- ✅ GeoJSON export for georeferenced detections
- ✅ Processing time tracking

**Dependencies**: `ultralytics`, `sahi`, `PIL`, `cv2`, `geo_data_utils`, `utils`

---

### 5. `validation_yolo.py` (18 KB) ⭐ **MODERATE VALUE**

**Purpose**: Single-image and batch validation with detailed metrics extraction.

**Key Features**:
- `run_validation_on_image(image_path, label_path, ...)`
  - Validates **single image** against ground truth
  - Creates temporary dataset structure for YOLO .val()
  - Extracts per-class metrics (precision, recall, mAP50, mAP50-95)
  - Generates Excel summaries

- Uses temporary directories to avoid dataset restructuring
- ClearML integration for validation tracking
- Flexible summary generation with class-wise breakdown

**Metrics Extracted**:
- Overall: Precision, Recall, F1, mAP50, mAP50-95
- Per-class: Precision, Recall, F1, AP50, AP50-95
- Box metrics specifically

**Potential Uses**:
- ⚠️ **Moderate utility**: Your `evaluate_models.py` already does batch evaluation
- ✅ Could use for **quick single-image debugging**
- ✅ Excel export format might be useful for reports
- ✅ Temporary dataset creation trick is clever for ad-hoc validation

**Dependencies**: `ultralytics`, `clearml_utils`, `utils`, `yaml`, `tempfile`

---

### 6. `clearml_utils.py` (8.0 KB) ⭐ **MODERATE VALUE**

**Purpose**: ClearML experiment tracking integration.

**Key Functions**:
- `start_clearml_task(name, description, project_name)`
  - Initializes ClearML task with error handling
  - Logs system info, Python version, package versions

- `log_clearml_result(task, results, epoch)`
  - Logs metrics to ClearML during training
  - Handles scalars, images, plots

**ClearML Features**:
- Remote execution support
- Web-based experiment dashboard
- Model registry
- Hyperparameter tracking

**Potential Uses**:
- ⚠️ **Optional**: Only if you want advanced experiment tracking
- ✅ Useful for team collaboration and remote monitoring
- ✅ Could replace/supplement your current logging
- ❌ Requires ClearML server setup

**Dependencies**: `clearml`

---

### 7. `main.py` (20 KB) ⭐ **LOW VALUE**

**Purpose**: CLI orchestrator for training, validation, and prediction.

**Features**:
- Command-line argument parsing
- Loads JSON config files
- Routes to training, validation, or prediction functions
- Multi-model orchestration

**Potential Uses**:
- ❌ **Not very useful**: You already have well-structured scripts
- ⚠️ Config file approach (JSON) vs your YAML approach
- ✅ Could reference for CLI argument patterns

---

### 8. `config.py` (566 B) - **UTILITY**

**Purpose**: Simple config loader from JSON.

**Potential Uses**:
- ❌ Minimal value - just loads JSON files

---

### 9. `converted_to_utm_from_disk_F.py` (4.5 KB) ⭐ **MODERATE VALUE**

**Purpose**: Batch convert GPS coordinates in EXIF to UTM and save metadata.

**Key Features**:
- Processes entire directories of images
- Extracts GPS from EXIF
- Converts to UTM using pyproj
- Saves metadata JSON files alongside images

**Potential Uses**:
- ✅ **Useful for preprocessing** georeferenced drone imagery
- ✅ Batch UTM conversion for your Santa Rosa datasets
- ⚠️ Hardcoded for specific UTM zone

**Dependencies**: `PIL`, `pyproj`, `json`

---

### 10. `utils_prompts.py` (18 KB) - **LOW VALUE**

**Purpose**: User prompt utilities for interactive CLI (yes/no, selection menus).

**Potential Uses**:
- ❌ Not needed for your automated workflow
- ⚠️ Could use if building interactive tools

---

## Documentation Files Copied

Located in `external_review/maaferna_INIA_docs/`:

1. **clearML-settings.md** - ClearML setup guide
2. **datasets-distribution.md** - Dataset split methodology
3. **experimentation-program.md** - Experiment planning templates
4. **inference-time-documentation.md** - Inference time benchmarks
5. **procedure-calculate-f1.md** - F1 score calculation details
6. **procedure-selection-best.md** - Best model selection criteria
7. **sahi-implementation.md** - SAHI sliced inference guide
8. **script-training.md** - Training script documentation

---

## Configuration Files Copied

Located in `external_review/`:

- `data_416.yaml`, `data_640.yaml`, `data_1024.yaml`, `data_2048.yaml` - Dataset configs for different resolutions
- `temp_data.yaml` - Temporary dataset configuration

---

## Priority Integration Recommendations

### 🔥 **MUST INTEGRATE**:

1. **`geo_data_utils.py`** - GeoJSON generation
   - Adapt UTM zone for your region
   - Integrate with your `evaluate_*.py` scripts
   - Export detections for QGIS visualization

2. **`utils.py` - Minimalistic visualization**
   - Replace default YOLO visualization with cleaner style
   - Confidence-only labels on boxes
   - Bottom legend with class summaries
   - Integrate into your prediction pipeline

3. **`predict_yolo.py` - SAHI integration**
   - Critical for high-resolution (1024px+) inference
   - Sliced inference improves small object detection
   - Structured output organization

### ⚡ **CONSIDER INTEGRATING**:

4. **`utils.py` - Best model finder**
   - Automate model selection in evaluation scripts
   - Replace manual path specification

5. **`utils.py` - EXIF/GPS extraction**
   - Extract drone metadata from UAV imagery
   - Useful for Santa Rosa georeferenced datasets

6. **`yolo_training.py` - Multi-run strategy**
   - Train same model 5x with different seeds
   - Select best performing run
   - Reduce variance in results

### 🤔 **OPTIONAL**:

7. **ClearML integration** - Only if you want advanced tracking
8. **Single-image validation** - For quick debugging

---

## Next Steps

### Immediate Actions:

1. **Review SAHI documentation** (`maaferna_INIA_docs/sahi-implementation.md`)
   - Understand sliced inference parameters
   - Test on your 1024x1024 images

2. **Adapt `geo_data_utils.py`**
   - Change UTM zone from 19S to your region's zone
   - Test GeoJSON generation with Santa Rosa data

3. **Test minimalistic visualization**
   - Run `apply_minimalistic_label_style_pil()` on sample images
   - Compare with current visualization

4. **Create integration plan**
   - Decide which functions to add to your `scripts/`
   - Consider creating `scripts/geo_utils.py` and `scripts/visualization.py`

### Integration Strategy:

```
ultralytics/
├── scripts/
│   ├── [EXISTING] run_experiments.py
│   ├── [EXISTING] evaluate_models.py
│   ├── [NEW] geo_utils.py          # From geo_data_utils.py
│   ├── [NEW] visualization.py       # From utils.py (viz functions)
│   ├── [NEW] sahi_predict.py        # From predict_yolo.py (SAHI parts)
│   └── [ENHANCED] evaluate_*.py     # Add GeoJSON export
```

---

## Technical Notes

### Dependencies to Install:

```bash
pip install sahi pyproj clearml
```

### UTM Zone for Chile (Santa Rosa):

- **Zone 19S** (used in these scripts) is correct for central Chile
- Verify your exact location: https://mangomap.com/robertyoung/maps/69585/what-utm-zone-am-i-in-

### SAHI Configuration:

- **Slice size**: 512-640px recommended for 1024px images
- **Overlap ratio**: 0.2-0.3 typical
- **Postprocessing**: NMS with IoU threshold ~0.5

---

## Contact

**Original Repository**: https://github.com/maaferna/INIA_Testing_YOLOV8
**Author**: maaferna (Miguel Fernández - likely INIA researcher)
**Review Date**: December 16, 2025
**Reviewed By**: Claude Code AI Assistant for lfleon9b

---

## License Note

⚠️ **Important**: Verify the license of the original repository before integrating code. If MIT/Apache/BSD licensed, integration is straightforward. If GPL, be aware of copyleft implications.

Check: https://github.com/maaferna/INIA_Testing_YOLOV8/blob/master/LICENSE

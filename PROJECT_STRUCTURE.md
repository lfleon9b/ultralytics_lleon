# 📁 Project Structure - Ultralytics Agricultural Weed Detection

**Last Updated**: December 16, 2025

This document describes the organization of the Ultralytics agricultural weed detection project.

---

## 🗂️ Root Directory Overview

```
ultralytics/                          # Project root
├── .claude.md                        # 📋 AI assistant project documentation
├── PROJECT_LOG.md                    # 📅 Session log with detailed history
├── PROJECT_STRUCTURE.md              # 📁 This file - directory organization
├── README.md                         # 🌾 Main README (agricultural focus)
├── CONTRIBUTING.md                   # 🤝 Contribution guidelines
├── LICENSE                           # ⚖️ AGPL-3.0 license
├── CITATION.cff                      # 📚 Citation information
├── pyproject.toml                    # 📦 Python project configuration
├── mkdocs.yml                        # 📖 Documentation site config
├── yolo11{n,s,m,l,x}.pt             # 🤖 Pre-trained YOLO11 models (24-114 MB)
│
├── configs/                          # ⚙️ Dataset configurations
├── datasets/                         # 🌱 Agricultural datasets
├── scripts/                          # 🐍 Custom training & evaluation scripts
├── experiments/                      # 📊 Training results & model weights
├── documents/                        # 📄 Experiment reports (Spanish)
├── external_review/                  # 🔍 External scripts analysis
│
├── ultralytics/                      # 🏗️ Core YOLO framework (DO NOT MODIFY)
├── docs/                             # 📖 Ultralytics documentation
├── examples/                         # 💡 Usage examples
├── tests/                            # ✅ Test suite
├── docker/                           # 🐳 Docker configurations
├── figures/                          # 📈 Visualization outputs
└── .github/                          # 🔧 GitHub workflows & templates
```

---

## 🌾 Agricultural Extensions (Our Custom Code)

### 1. `configs/` - Dataset Configurations
**Purpose**: YAML files defining dataset paths, class names, and metadata.

```
configs/
├── lentils_v1.yaml                   # Legacy 4-class configuration
│   # Classes: AMBEL, LENCU, POLAV, POLPE
│   # Original multi-model benchmark dataset
│
├── merge_varios_cultivos.yaml        # 6-class multi-crop unified model
│   # Classes: AMBEL, LENCU, LOLSS, POLAV, POLPE, RAPRA
│   # 5,765 train / 181 val / 140 test images
│   # mAP50: 81.9% (YOLO11l)
│
├── sr_dauca.yaml                     # Single-class DAUCA specialist
│   # Classes: DAUCA (Daucus carota - Wild carrot)
│   # 70 train / 3 val / 1 test images
│   # mAP50: 86.4%, Precision: 84.6% (YOLO11l)
│
└── ultralytics.code-workspace        # VS Code workspace settings
```

**File Format**:
```yaml
train: /path/to/dataset/train/images
val: /path/to/dataset/valid/images
test: /path/to/dataset/test/images
nc: 6                                  # Number of classes
names: ['AMBEL', 'LENCU', 'LOLSS', 'POLAV', 'POLPE', 'RAPRA']
roboflow: fia2024/mergevarios          # Roboflow project reference
```

---

### 2. `scripts/` - Custom Training & Evaluation Scripts
**Purpose**: Automation scripts for experiments, evaluation, and monitoring.

```
scripts/
├── run_experiments.py                # 🚀 Multi-model orchestrator
│   # Trains all 5 YOLO11 variants (n, s, m, l, x) sequentially
│   # Optimized batch sizes per model for dual RTX 4090
│   # DLM2 methodology: ALL augmentation disabled
│   # Usage: python scripts/run_experiments.py
│
├── train_sr_dauca.py                 # 🎯 DAUCA specialist training
│   # Single-model training for sr_dauca dataset
│   # YOLO11l with DLM2 parameters
│   # Usage: python scripts/train_sr_dauca.py
│
├── train_merge_varios.py             # 🌾 Multi-crop training
│   # Single-model training for merge_varios_cultivos
│   # YOLO11l with DLM2 parameters
│   # Usage: python scripts/train_merge_varios.py
│
├── evaluate_models.py                # 📊 Batch evaluation
│   # Evaluates all completed experiments on test set
│   # Generates per-class metrics (precision, recall, F1, mAP50)
│   # Outputs: CSV files with class-wise breakdown
│   # Usage: python scripts/evaluate_models.py
│
├── evaluate_sr_dauca.py              # 🔍 DAUCA evaluation
│   # Detailed evaluation on val and test splits
│   # Generates confusion matrix, per-class metrics
│   # Usage: python scripts/evaluate_sr_dauca.py
│
├── evaluate_merge_varios.py          # 🔍 Multi-crop evaluation
│   # Similar to evaluate_sr_dauca for merge_varios dataset
│   # Usage: python scripts/evaluate_merge_varios.py
│
├── compare_results.py                # 📈 Cross-experiment comparison
│   # Compiles metrics across all experiments
│   # Generates comparison tables (CSV)
│   # Usage: python scripts/compare_results.py
│
├── plot_results.py                   # 📊 Visualization
│   # Plots training curves (loss, mAP, precision, recall)
│   # Usage: python scripts/plot_results.py <experiment_path>
│
├── live_plot.py                      # 📉 Real-time monitoring
│   # Live training curve plotting during experiments
│   # Monitors results.csv for updates
│   # Usage: python scripts/live_plot.py <experiment_path>
│
└── live_dashboard.py                 # 🖥️ GPU monitoring
    # Real-time GPU utilization dashboard
    # Uses pynvml for GPU stats (temperature, memory, utilization)
    # Usage: python scripts/live_dashboard.py
```

**Key Features**:
- **DLM2 Compliance**: All training scripts disable internal augmentation
- **Multi-GPU Support**: DDP with device selection `0,1`
- **Batch Size Optimization**: Tailored per model for 2×RTX 4090 (48GB total)
- **Comprehensive Logging**: JSON summaries, CSV metrics, TensorBoard

---

### 3. `datasets/` - Agricultural Datasets
**Purpose**: Image datasets with YOLO format annotations (train/val/test splits).

```
datasets/
├── lentils_v1/                       # Legacy 4-class dataset
│   ├── train/
│   │   ├── images/                   # Training images
│   │   └── labels/                   # YOLO format .txt labels
│   ├── valid/                        # Validation split
│   ├── test/                         # Test split
│   ├── data.yaml                     # Dataset config (auto-generated)
│   └── README.roboflow.txt           # Roboflow export info
│
├── merge_varios_cultivos/            # Multi-crop unified dataset
│   ├── train/                        # 5,765 images
│   ├── valid/                        # 181 images
│   ├── test/                         # 140 images
│   ├── data.yaml
│   └── README.roboflow.txt
│
├── sr_dauca/                         # DAUCA specialist dataset
│   ├── train/                        # 70 images (SMALL!)
│   ├── valid/                        # 3 images
│   ├── test/                         # 1 image
│   ├── data.yaml
│   └── README.roboflow.txt
│
└── sr_dauca_extra/                   # Additional DAUCA samples
    ├── StaRosa_DAUCA/                # Raw drone imagery
    └── StaRosa_DAUCA-20251211T.../   # Extracted patches
```

**Species Mapping**:
| Code | Scientific Name | Common Name | Type |
|------|----------------|-------------|------|
| AMBEL | *Ambrosia artemisiifolia* | Ragweed | Weed |
| LENCU | *Lens culinaris* | Lentil | Crop |
| LOLSS | *Lolium* spp. | Ryegrass | Weed |
| POLAV | *Polygonum aviculare* | Knotweed | Weed |
| POLPE | *Polygonum persicaria* | Smartweed | Weed |
| RAPRA | *Raphanus raphanistrum* | Wild radish | Weed |
| DAUCA | *Daucus carota* | Wild carrot | Weed |

**Label Format** (YOLO):
```
<class_id> <x_center> <y_center> <width> <height>
```
All coordinates normalized to [0, 1].

---

### 4. `experiments/` - Training Results & Weights
**Purpose**: Outputs from training runs, organized by dataset and model configuration.

```
experiments/
├── comparison_report.csv             # Cross-experiment comparison
│
├── lentils_v1_dlm2/                  # DLM2 methodology runs
│   ├── live_dashboard.png            # GPU monitoring screenshot
│   ├── test_metrics_by_class.csv     # Per-class test metrics
│   ├── yolo11n_img1024_ep50_noaug/   # Nano model results
│   ├── yolo11s_img1024_ep50_noaug/   # Small model results
│   ├── yolo11m_img1024_ep50_noaug/   # Medium model results
│   ├── yolo11l_img1024_ep50_noaug/   # Large model results
│   └── yolo11x_img1024_ep50_noaug/   # Extra-large model results
│
├── merge_varios_cultivos/
│   ├── experiment_log.md             # Experiment notes & findings
│   ├── final_evaluation_report.csv   # Test set metrics
│   ├── eval_val/                     # Validation evaluation outputs
│   ├── eval_test/                    # Test evaluation outputs
│   ├── live_dashboard.png
│   └── yolo11l_img1024_ep50_noaug/
│       ├── weights/
│       │   ├── best.pt               # Best model (by mAP50)
│       │   └── last.pt               # Last epoch
│       ├── results.csv               # Training metrics per epoch
│       ├── results.png               # Training curves
│       ├── confusion_matrix.png      # Confusion matrix
│       ├── F1_curve.png              # F1 vs confidence
│       ├── P_curve.png               # Precision vs confidence
│       ├── R_curve.png               # Recall vs confidence
│       ├── PR_curve.png              # Precision-Recall curve
│       └── args.yaml                 # Training arguments used
│
└── sr_dauca/
    ├── experiment_log.md
    ├── final_evaluation_report.csv
    ├── eval_val/                     # Validation evaluation
    ├── eval_test/                    # Test evaluation
    ├── eval_val_recheck/             # Re-validation runs
    ├── live_dashboard.png
    └── yolo11l_img1024_ep50_noaug/
        └── [same structure as above]
```

**Important**:
- ⚠️ **DO NOT COMMIT** `experiments/` to git (too large, excluded in .gitignore)
- Model weights: `best.pt` (20-100 MB), `last.pt`
- Training history: `results.csv` tracks loss, mAP, precision, recall per epoch

---

### 5. `documents/` - Experiment Reports
**Purpose**: Detailed Spanish-language reports summarizing experimental results.

```
documents/
├── resumen_experiencias_multicultivo_2025.md
│   # Multi-crop detection summary (December 2025)
│   # YOLO11l results: 81.9% mAP50, per-class breakdown
│   # Methodology: DLM2, dataset characteristics
│   # Limitations: POLAV class imbalance (57% mAP50)
│   # Recommendations for improvement
│
└── resumen_experiencias_dauca_2025.md
    # DAUCA specialist summary (December 2025)
    # YOLO11l results: 86.4% mAP50, 84.6% precision
    # Geographic context: Santa Rosa region
    # Integration with Sentinel-2 NDVI validation
    # Spatial autocorrelation analysis (Moran's I = 0.667)
    # Recommendations: Expand dataset beyond 70 images
```

**Report Contents**:
- Methodology description (DLM2, hardware, hyperparameters)
- Dataset characteristics (size, distribution, imbalance)
- Training configuration (batch size, epochs, augmentation = disabled)
- Results tables (precision, recall, F1, mAP50, mAP50-95)
- Per-class performance breakdown
- Confusion matrices and visualizations
- Limitations and challenges
- Recommendations for future work

---

### 6. `external_review/` - External Scripts Analysis
**Purpose**: Collection and review of useful scripts from community repositories.

```
external_review/
├── INVENTORY.md                      # 📋 Comprehensive script analysis
│   # 420+ lines of detailed review
│   # Script-by-script breakdown
│   # Priority rankings and integration recommendations
│   # Code examples and use cases
│
├── maaferna_INIA_scripts/            # Python scripts (172 KB)
│   ├── geo_data_utils.py             # ⭐⭐⭐ UTM↔GPS, GeoJSON generation
│   ├── utils.py                      # ⭐⭐⭐ Visualization, EXIF, model finder
│   ├── predict_yolo.py               # ⭐⭐⭐ SAHI sliced inference
│   ├── validation_yolo.py            # ⭐⭐ Single-image validation
│   ├── yolo_training.py              # ⭐⭐ Multi-run training
│   ├── clearml_utils.py              # ⭐ Experiment tracking
│   ├── converted_to_utm_from_disk_F.py  # ⭐ GPS→UTM batch conversion
│   ├── main.py                       # CLI orchestrator
│   ├── utils_prompts.py              # Interactive prompts
│   └── config.py                     # Config loader
│
├── maaferna_INIA_docs/               # Documentation (8 files)
│   ├── sahi-implementation.md        # SAHI sliced inference guide
│   ├── clearML-settings.md           # ClearML setup
│   ├── datasets-distribution.md      # Dataset split methodology
│   ├── experimentation-program.md    # Experiment planning
│   ├── inference-time-documentation.md  # Benchmarks
│   ├── procedure-calculate-f1.md     # F1 calculation
│   ├── procedure-selection-best.md   # Best model selection
│   └── script-training.md            # Training docs
│
├── maaferna_INIA_README.md           # Original repo README
├── data_416.yaml                     # Sample configs
├── data_640.yaml
├── data_1024.yaml
└── data_2048.yaml
```

**Key Findings**:
1. **SAHI Integration** - Critical for 1024px+ high-res images
2. **Geospatial Tools** - GeoJSON export, UTM conversion
3. **Enhanced Visualization** - Minimalistic labels (confidence only + legend)
4. **Best Model Finder** - Automated model selection by mAP50
5. **Multi-run Strategy** - Train 5x with different seeds, pick best

**Status**: 📋 Reviewed, awaiting integration (Session 3)

---

## 🏗️ Core Ultralytics Framework (Upstream)

### 7. `ultralytics/` - Main Framework
**Purpose**: Core YOLO implementation (DO NOT MODIFY - maintain upstream compatibility)

```
ultralytics/
├── __init__.py                       # Package initialization
├── cfg/                              # Default configurations
│   ├── __init__.py                   # Config system
│   ├── datasets/                     # Dataset configs (COCO, VOC, etc.)
│   ├── models/                       # Model architectures (YAML)
│   │   ├── 11/                       # YOLO11 variants
│   │   ├── v8/                       # YOLO v8 variants
│   │   ├── v9/                       # YOLO v9 variants
│   │   └── ...
│   └── trackers/                     # Tracking configs
│
├── data/                             # Data loading & augmentation
│   ├── augment.py                    # Augmentation transforms
│   ├── base.py                       # Base dataset class
│   ├── build.py                      # Dataset builders
│   ├── loaders.py                    # Data loaders
│   └── utils.py                      # Data utilities
│
├── engine/                           # Training & inference engines
│   ├── trainer.py                    # Main training loop
│   ├── validator.py                  # Validation logic
│   ├── predictor.py                  # Inference engine
│   ├── exporter.py                   # Model export (ONNX, TF, etc.)
│   ├── tuner.py                      # Hyperparameter tuning
│   └── results.py                    # Result handling
│
├── models/                           # Model implementations
│   ├── yolo/                         # YOLO models
│   │   ├── detect/                   # Detection models
│   │   ├── segment/                  # Segmentation models
│   │   ├── classify/                 # Classification models
│   │   ├── pose/                     # Pose estimation
│   │   └── obb/                      # Oriented bounding boxes
│   ├── sam/                          # Segment Anything Model (SAM)
│   ├── fastsam/                      # Fast SAM
│   ├── rtdetr/                       # RT-DETR
│   └── nas/                          # Neural Architecture Search
│
├── nn/                               # Neural network modules
│   ├── modules/                      # Building blocks
│   │   ├── block.py                  # Convolution blocks
│   │   ├── conv.py                   # Convolution layers
│   │   ├── head.py                   # Detection heads
│   │   └── transformer.py            # Transformer blocks
│   ├── tasks.py                      # Model task definitions
│   ├── autobackend.py                # Multi-backend support
│   └── text_model.py                 # Text models
│
├── utils/                            # Utility functions
│   ├── callbacks/                    # Training callbacks
│   ├── export/                       # Export utilities
│   ├── benchmarks.py                 # Benchmarking
│   ├── checks.py                     # System checks
│   ├── downloads.py                  # Model/data downloads
│   ├── files.py                      # File operations
│   ├── logger.py                     # Logging
│   ├── metrics.py                    # Evaluation metrics
│   ├── ops.py                        # Operations (NMS, etc.)
│   ├── plotting.py                   # Visualization
│   ├── torch_utils.py                # PyTorch utilities
│   └── ...
│
├── solutions/                        # Pre-built solutions
│   ├── object_counter.py             # Object counting
│   ├── heatmaps.py                   # Heatmap generation
│   ├── distance_calculation.py       # Distance measurement
│   ├── speed_estimation.py           # Speed tracking
│   └── ...
│
├── trackers/                         # Object tracking
│   ├── bot_sort.py                   # BoT-SORT tracker
│   ├── byte_tracker.py               # ByteTrack
│   └── utils/                        # Tracking utilities
│
└── hub/                              # Ultralytics HUB integration
    ├── __init__.py
    ├── auth.py                       # Authentication
    └── session.py                    # Cloud sessions
```

**Important**:
- ✅ **USE** these modules in your scripts (import from ultralytics)
- ❌ **DO NOT MODIFY** core framework files
- 🔄 **PULL UPDATES** regularly: `git fetch upstream && git merge upstream/main`

---

### 8. `docs/` - Documentation
**Purpose**: Ultralytics documentation site (MkDocs-based).

```
docs/
├── en/                               # English documentation
│   ├── datasets/                     # Dataset guides
│   ├── guides/                       # How-to guides
│   ├── models/                       # Model documentation
│   ├── modes/                        # Operation modes (train, val, predict)
│   ├── tasks/                        # Task types (detect, segment, classify)
│   ├── integrations/                 # Third-party integrations
│   └── reference/                    # API reference
├── macros/                           # Documentation macros
└── overrides/                        # Custom theme elements
```

**Access**: https://docs.ultralytics.com/

---

### 9. `examples/` - Usage Examples
**Purpose**: Example scripts demonstrating various use cases.

```
examples/
├── tutorial.ipynb                    # Quickstart tutorial
├── heatmaps.ipynb                    # Heatmap visualization
├── object_counting.ipynb             # Object counting
├── object_tracking.ipynb             # Multi-object tracking
├── hub.ipynb                         # Ultralytics HUB usage
│
├── YOLOv8-ONNXRuntime/               # ONNX Runtime inference
├── YOLOv8-OpenCV-ONNX-Python/        # OpenCV + ONNX
├── YOLOv8-CPP-Inference/             # C++ inference
├── YOLOv8-SAHI-Inference-Video/      # SAHI sliced inference example
├── RTDETR-ONNXRuntime-Python/        # RT-DETR inference
└── ...                               # Many more examples
```

**Relevant Example**: `YOLOv8-SAHI-Inference-Video/` - Study for Session 3 SAHI integration!

---

### 10. `tests/` - Test Suite
**Purpose**: Automated tests for framework validation.

```
tests/
├── test_python.py                    # Python API tests
├── test_cli.py                       # CLI tests
├── test_exports.py                   # Export functionality tests
├── test_solutions.py                 # Solutions tests
└── ...
```

**Usage**: `pytest tests/` (run all tests)

---

### 11. `docker/` - Docker Configurations
**Purpose**: Containerization for reproducible environments.

```
docker/
├── Dockerfile                        # Standard CUDA image
├── Dockerfile-cpu                    # CPU-only image
├── Dockerfile-arm64                  # ARM architecture (Apple Silicon)
├── Dockerfile-jetson-jetpack{4,5,6}  # NVIDIA Jetson
├── Dockerfile-conda                  # Conda environment
└── ...
```

**Usage**: `docker build -f docker/Dockerfile -t ultralytics .`

---

## 📊 Additional Directories

### 12. `figures/` - Visualization Outputs
**Purpose**: Generated plots, charts, and images (auto-created during experiments).

```
figures/
├── training_curves/                  # Loss/mAP plots
├── confusion_matrices/               # Confusion matrices
├── detection_samples/                # Sample predictions
└── comparison_plots/                 # Cross-experiment comparisons
```

**Status**: May not exist initially, created by scripts.

---

### 13. `runs/` - YOLO CLI Outputs
**Purpose**: Default output directory for YOLO command-line runs.

```
runs/
├── detect/                           # Detection runs
│   ├── train/                        # Training outputs
│   ├── val/                          # Validation outputs
│   └── predict/                      # Prediction outputs
├── segment/                          # Segmentation runs
├── classify/                         # Classification runs
└── pose/                             # Pose estimation runs
```

**Note**:
- ⚠️ Excluded from git (.gitignore)
- Use `experiments/` for organized long-term storage instead

---

## 🔧 Configuration Files

### Root Level Configuration Files

| File | Purpose |
|------|---------|
| `pyproject.toml` | Python project metadata, dependencies, build system |
| `mkdocs.yml` | Documentation site configuration (MkDocs) |
| `.gitignore` | Git exclusion patterns (experiments/, runs/, *.pt, __pycache__, etc.) |
| `.pre-commit-config.yaml` | Pre-commit hooks for code quality |
| `CITATION.cff` | Citation information for academic use |
| `LICENSE` | AGPL-3.0 license text |
| `CONTRIBUTING.md` | Contribution guidelines for upstream |
| `SECURITY.md` | Security policy |

---

## 📦 Model Weights (Root Directory)

```
/home/malezainia1/dev/ultralytics/
├── yolo11n.pt                        # Nano (2.6M params, 24 MB)
├── yolo11s.pt                        # Small (9.4M params, 38 MB)
├── yolo11m.pt                        # Medium (20.1M params, 81 MB)
├── yolo11l.pt                        # Large (25.3M params, 102 MB)
└── yolo11x.pt                        # Extra-Large (56.9M params, 114 MB)
```

**Usage**: Loaded automatically by Ultralytics if present, otherwise downloaded from GitHub releases.

---

## 🚫 Excluded from Git (.gitignore)

```
# Python
__pycache__/
*.pyc
*.pyo
*.egg-info/

# Experiments & Runs (too large)
experiments/
runs/

# Model Weights (download on-demand)
*.pt
*.pth
*.weights

# Datasets (download from Roboflow)
datasets/*/train/
datasets/*/valid/
datasets/*/test/

# Logs & Temporary Files
*.log
.DS_Store
Thumbs.db

# IDE & Environment
.vscode/
.idea/
*.swp
.env
```

---

## 🔗 Git Configuration

### Remotes

```bash
origin    https://github.com/lfleon9b/ultralytics_lleon.git  (your fork)
upstream  https://github.com/ultralytics/ultralytics.git     (original)
```

### Workflow

```bash
# Work on your fork
git add <files>
git commit -m "feat: description"
git push origin main

# Get upstream updates
git fetch upstream
git merge upstream/main
git push origin main
```

---

## 📈 Project Growth Metrics

| Metric | Count | Notes |
|--------|-------|-------|
| **Custom Scripts** | 10 | In `scripts/` |
| **Dataset Configs** | 3 | In `configs/` |
| **Datasets** | 3-4 | In `datasets/` (5,765+ images total) |
| **Experiments** | 15+ | In `experiments/` (5 models × 3 datasets) |
| **Model Weights** | 5 | YOLO11 n/s/m/l/x (24-114 MB each) |
| **Documentation** | 5 | .claude.md, PROJECT_LOG.md, PROJECT_STRUCTURE.md, 2 Spanish reports |
| **External Scripts Reviewed** | 10 | In `external_review/` (172 KB) |
| **Total Lines of Custom Code** | ~2,000+ | Scripts + configs |
| **Total Lines of Documentation** | ~1,500+ | All .md files |

---

## 🎯 Directory Usage Guidelines

### DO:
✅ Add new scripts to `scripts/`
✅ Add new dataset configs to `configs/`
✅ Document experiments in `documents/`
✅ Commit small files (<1MB): configs, scripts, docs
✅ Update `.claude.md`, `PROJECT_LOG.md` regularly

### DON'T:
❌ Modify `ultralytics/` core framework
❌ Commit `experiments/` (too large)
❌ Commit model weights `*.pt` (use Git LFS or exclude)
❌ Commit datasets (use Roboflow/external hosting)
❌ Hardcode absolute paths (use relative or config-based)

---

## 🔄 Keeping Upstream in Sync

**Recommended Frequency**: Weekly (or when new features released)

```bash
# Check for updates
git fetch upstream

# View changes
git log upstream/main --oneline -10

# Merge (no conflicts expected in custom dirs)
git merge upstream/main

# Resolve any conflicts (rare if you don't modify ultralytics/)
# ...

# Push to your fork
git push origin main
```

**Recent Upstream Updates**:
- SAM3 model integration (Dec 2025)
- SystemLogger improvements
- Documentation fixes
- Export enhancements

---

## 📞 Quick Reference

### File Locations Cheat Sheet

| What | Where |
|------|-------|
| Dataset configs | `configs/*.yaml` |
| Training scripts | `scripts/train_*.py` |
| Evaluation scripts | `scripts/evaluate_*.py` |
| Experiment results | `experiments/{dataset}/{model_config}/` |
| Best model weights | `experiments/{dataset}/{model_config}/weights/best.pt` |
| Training curves | `experiments/{dataset}/{model_config}/results.png` |
| Per-class metrics | `experiments/{dataset}/final_evaluation_report.csv` |
| Experiment reports | `documents/*.md` |
| Project documentation | `.claude.md`, `PROJECT_LOG.md`, `PROJECT_STRUCTURE.md` |
| External scripts | `external_review/maaferna_INIA_scripts/*.py` |
| Pre-trained models | Root directory: `yolo11{n,s,m,l,x}.pt` |

---

**Last Updated**: December 16, 2025
**Next Review**: After Session 3 (SAHI integration)

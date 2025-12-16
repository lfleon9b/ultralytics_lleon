# 🌾 Ultralytics Agricultural Weed Detection - Project Log

**Project Name**: Ultralytics YOLO for Precision Agriculture
**Geographic Focus**: Santa Rosa Region, Chile
**Primary Goal**: Multi-crop weed detection for precision spot-spraying
**Target Impact**: 35-50% reduction in agrochemical usage

---

## 📅 Session Log

### Session 1 - Initial Setup & Experiments
**Date**: December 11-12, 2025
**Status**: ✅ Completed

#### Achievements:
1. **Repository Setup**
   - Forked Ultralytics YOLO framework
   - Configured dual RTX 4090 GPU environment with DDP support
   - Established DLM2 methodology (pre-augmented datasets, no internal augmentation)

2. **Dataset Configuration**
   - Created `configs/lentils_v1.yaml` (4 classes: AMBEL, LENCU, POLAV, POLPE)
   - Created `configs/sr_dauca.yaml` (1 class: DAUCA - Daucus carota specialist)
   - Created `configs/merge_varios_cultivos.yaml` (6 classes: multi-crop unified model)

3. **Custom Script Development**
   - `scripts/run_experiments.py` - Multi-model orchestrator (YOLO11 n/s/m/l/x)
   - `scripts/train_sr_dauca.py` - DAUCA specialist training
   - `scripts/train_merge_varios.py` - Multi-crop training
   - `scripts/evaluate_models.py` - Per-class metrics extraction
   - `scripts/evaluate_sr_dauca.py` - DAUCA validation & test evaluation
   - `scripts/evaluate_merge_varios.py` - Multi-crop evaluation
   - `scripts/compare_results.py` - Cross-experiment comparison
   - `scripts/plot_results.py` - Training curve visualization
   - `scripts/live_plot.py` - Real-time training monitoring
   - `scripts/live_dashboard.py` - GPU monitoring dashboard (pynvml)

4. **Experimental Results**
   - **Multi-crop model** (merge_varios_cultivos, YOLO11l, 50 epochs):
     - mAP50: 81.9% (validation), 79.9% (test)
     - Precision: 79.9% (global)
     - Best class: LENCU (F1=0.90, mAP50=94.3%)
     - Challenging class: POLAV (mAP50=57.0% - dataset imbalance issue)

   - **DAUCA specialist** (sr_dauca, YOLO11l, 50 epochs):
     - Precision: 84.6% (high selectivity for spot-spraying)
     - Recall: 79.1% (high detection rate)
     - mAP50: 86.4% (robust performance)
     - Note: Small dataset (70 train images) - high variance expected

5. **Documentation**
   - Created experiment reports in `documents/` (Spanish)
   - Documented DLM2 methodology compliance
   - Multi-scale integration notes (UAS 1.5mm/px + Sentinel-2 10m/px)

#### Git Commits:
```
97c369ec3 - docs: Add summary report for Daucus carota detection (Santa Rosa)
6c739728a - docs: Add summary report for multi-crop detection experiences (Dec 2025)
9df36dd64 - docs: Update sr_dauca experiment log with validation metrics
4f189e208 - feat: Add experiment results for merge_varios and sr_dauca datasets (YOLO11l, 50 epochs)
```

---

### Session 2 - Documentation & External Resources
**Date**: December 16, 2025
**Status**: ✅ Completed

#### Achievements:

1. **Project Exploration & Summary** ✅
   - Comprehensive codebase analysis using Explore agent
   - Identified key project components:
     - Core Ultralytics framework (13 subdirectories)
     - Custom agricultural extensions (configs, scripts, datasets)
     - 3 active datasets (5,765+ training images total)
     - Multi-GPU DDP training infrastructure
   - Analyzed recent experimental results and methodology

2. **Documentation Creation** ✅
   - **`.claude.md`** - Comprehensive project guide (13 sections, ~400 lines)
     - Project identity and agricultural focus
     - Critical DLM2 methodology conventions
     - Dataset descriptions with performance metrics
     - Common tasks and workflows
     - Development guidelines (DOs/DON'Ts)
     - Integration context (multi-scale detection pipeline)
     - Quick reference tables

   - Purpose: Enable AI assistant to understand project context in future sessions
   - Coverage: Technical specs, methodology, results, file locations, best practices

3. **GitHub Repository Setup** ✅
   - Configured git remotes:
     - `origin` → https://github.com/lfleon9b/ultralytics_lleon (your fork)
     - `upstream` → https://github.com/ultralytics/ultralytics (original)
   - Updated `README.md` with agricultural focus section:
     - Project overview and key metrics
     - Current models and results
     - Custom extensions description
     - Instructions for staying updated with upstream
   - Fork visibility: Public (GitHub policy - forks of public repos must be public)

4. **Initial Commit to Fork** ✅
   - Committed files:
     - `.claude.md` (project documentation)
     - `README.md` (updated with agricultural focus)
     - `configs/` (3 dataset YAML files + workspace config)
     - `scripts/` (10 custom Python scripts)
   - Merged upstream changes (SAM3 model integration, 283 files updated)
   - Pushed to GitHub: https://github.com/lfleon9b/ultralytics_lleon

   **Commit message**:
   ```
   feat: Add agricultural weed detection extensions for precision farming

   This commit adds specialized configurations and scripts for agricultural
   weed detection applications, extending Ultralytics YOLO for precision
   farming use cases.

   Changes:
   - Add .claude.md with comprehensive project documentation
   - Update README.md with agricultural focus and fork information
   - Add dataset configs for multi-crop and specialist weed models (3 configs)
   - Add custom training scripts for agricultural experiments (10 scripts)
   - Include evaluation, monitoring, and comparison utilities

   Key features:
   - Multi-crop weed detection (6 species, 81.9% mAP50)
   - DAUCA specialist model (84.6% precision)
   - DLM2 methodology compliance (pre-augmented datasets)
   - Dual RTX 4090 GPU optimization with DDP support
   - Integration with satellite imagery validation

   Geographic focus: Santa Rosa region, Chile
   Target: 35-50% reduction in agrochemical usage through spot-spraying
   ```

5. **External Resources Review** ✅
   - Cloned INIA_Testing_YOLOV8 repository (maaferna)
   - Collected 10 custom Python scripts (172 KB total)
   - Copied 8 documentation files
   - Organized in `external_review/` folder structure:
     ```
     external_review/
     ├── INVENTORY.md (comprehensive analysis, 420 lines)
     ├── maaferna_INIA_scripts/ (10 Python files)
     ├── maaferna_INIA_docs/ (8 markdown files)
     ├── maaferna_INIA_README.md
     └── *.yaml configs
     ```

6. **Detailed Script Inventory** ✅
   - Created `external_review/INVENTORY.md` with:
     - Script-by-script analysis (10 scripts reviewed)
     - Function-level breakdown of key utilities
     - Priority rankings (⭐⭐⭐ High Value → ⭐ Low Value)
     - Integration recommendations
     - Dependencies mapping
     - Code examples and use cases

   **High-Value Scripts Identified**:
   - `geo_data_utils.py` - UTM↔GPS conversion, GeoJSON generation
   - `utils.py` - Minimalistic visualization, EXIF extraction, best model finder
   - `predict_yolo.py` - **SAHI sliced inference** integration (critical for 1024px images!)
   - `validation_yolo.py` - Single-image validation utilities
   - `yolo_training.py` - Multi-run training strategy (5 seeds, pick best)

7. **Learning Session: Technical Terminology** ✅
   - Explained "refactoring" concept with examples
   - Clarified code reuse terminology:
     - Modularity / Modular programming
     - Utility/Helper modules
     - Extensions / Plugins
     - DRY principle (Don't Repeat Yourself)
   - Discussed project organization patterns

#### Git Activity:
```bash
# Remote configuration
git remote add origin https://github.com/lfleon9b/ultralytics_lleon.git
git remote rename origin upstream
git remote rename fork origin

# Staging and commit
git add .claude.md configs/ scripts/ README.md
git commit -m "feat: Add agricultural weed detection extensions..."

# Merge and push
git pull origin main --no-rebase  # Merged 283 upstream files
git push origin main              # Pushed commit 381ca46ce
```

#### Metrics:
- **Files created**: 2 major documentation files (.claude.md, INVENTORY.md)
- **Files modified**: 1 (README.md)
- **Files committed**: 16 total
- **External scripts reviewed**: 10 (172 KB)
- **Documentation reviewed**: 8 files
- **Lines of documentation written**: ~820 lines

---

## 🎯 Next Steps (Session 3 & Beyond)

### Session 3 - SAHI Integration & Geospatial Features
**Planned Date**: December 17, 2025
**Status**: 📋 Planned

#### Priority 1: SAHI Sliced Inference 🔥
**Why**: Critical for improving small object detection in 1024×1024px high-resolution images.

**Tasks**:
1. [ ] Install dependencies
   ```bash
   pip install sahi pyproj
   ```

2. [ ] Review SAHI documentation
   - Read `external_review/maaferna_INIA_docs/sahi-implementation.md`
   - Understand slice_height, slice_width, overlap_ratio parameters
   - Study post-processing options (NMS, match_metric, match_threshold)

3. [ ] Test SAHI on sample images
   - Select 5-10 representative images from sr_dauca/merge_varios
   - Run standard YOLO inference vs SAHI sliced inference
   - Compare:
     - Detection counts (especially small objects)
     - mAP50 scores
     - Inference time
     - Visual quality

4. [ ] Create `scripts/sahi_predict.py`
   - Extract SAHI logic from `external_review/maaferna_INIA_scripts/predict_yolo.py`
   - Adapt to your project structure
   - Add command-line arguments:
     - `--slice-size` (default: 640)
     - `--overlap-ratio` (default: 0.2)
     - `--model-type` (standard, sahi, both)
   - Integrate with your existing output structure

5. [ ] Benchmark on validation set
   - Run on full sr_dauca validation set (3 images)
   - Run on full merge_varios validation set (181 images)
   - Generate comparison report

**Expected Outcomes**:
- 5-15% improvement in small object detection
- Documented slice configuration for production
- Updated `.claude.md` with SAHI workflow

---

#### Priority 2: Geospatial Integration 🗺️
**Why**: Enable QGIS/ArcGIS visualization and spatial analysis of detections.

**Tasks**:
1. [ ] Adapt `geo_data_utils.py` for Chile
   - Change UTM zone configuration (currently hardcoded to Zone 19S)
   - Verify zone for Santa Rosa: https://mangomap.com/robertyoung/maps/69585/what-utm-zone-am-i-in-
   - Test with known coordinates

2. [ ] Extract GPS from drone imagery
   - Use `utils.py` functions: `get_image_metadata()`, `extract_exif_metadata()`
   - Test on sr_dauca dataset (if GPS available)
   - Create batch processing script: `scripts/extract_gps.py`

3. [ ] Integrate GeoJSON generation
   - Modify `evaluate_sr_dauca.py` and `evaluate_merge_varios.py`
   - Add `--export-geojson` flag
   - Generate GeoJSON for each image with:
     - Detection count per class
     - Model confidence
     - GPS/UTM coordinates
     - Timestamp

4. [ ] Create spatial analysis notebook
   - Load GeoJSON in Python (geopandas)
   - Visualize detection density heatmap
   - Calculate Moran's I (spatial autocorrelation)
   - Generate prescription maps for VRA (Variable Rate Application)

5. [ ] Test in QGIS
   - Import GeoJSON layer
   - Overlay on satellite imagery (Sentinel-2)
   - Validate detection locations
   - Export styled maps for documentation

**Expected Outcomes**:
- GeoJSON exports for all experiments
- QGIS-ready detection layers
- Spatial analysis report
- Updated `.claude.md` with geospatial workflow

---

#### Priority 3: Enhanced Visualization 🎨
**Why**: Cleaner, more professional detection visualizations for reports and presentations.

**Tasks**:
1. [ ] Create `scripts/visualization.py`
   - Extract functions from `external_review/maaferna_INIA_scripts/utils.py`:
     - `apply_minimalistic_label_style_pil()` - Confidence-only labels
     - `create_bottom_summary_image()` - Legend generation
     - `combine_frame_with_bottom_summary()` - Image + legend
   - Adapt color schemes for your class names

2. [ ] Test minimalistic style
   - Run on 10 sample predictions
   - Compare with default YOLO visualization
   - Get feedback on label placement, font size, colors

3. [ ] Integrate into prediction pipeline
   - Update `scripts/evaluate_models.py`
   - Add `--style` argument (default, minimalistic, both)
   - Generate both styles during evaluation

4. [ ] Create comparison gallery
   - Script to generate side-by-side comparison images
   - Save to `figures/visualization_comparison/`
   - Update experiment reports with new visualizations

**Expected Outcomes**:
- Professional detection visualizations
- Customizable styling options
- Gallery of results for documentation

---

### Session 4 - Model Optimization & Ablation Studies
**Planned Date**: December 18-19, 2025
**Status**: 📋 Planned

#### Tasks:
1. [ ] Address class imbalance (POLAV: 57% mAP50)
   - Analyze class distribution in training set
   - Consider:
     - Data augmentation specifically for POLAV
     - Class weights in loss function
     - Oversampling minority class
     - Collect more POLAV samples
   - Re-train and evaluate

2. [ ] Expand sr_dauca dataset (currently 70 images)
   - Collect additional drone imagery
   - Augment existing samples (if DLM2 allows external augmentation)
   - Split strategy: maintain test set, expand train/val

3. [ ] Ablation studies
   - Compare YOLO11n vs YOLO11s vs YOLO11m vs YOLO11l (already done)
   - Test different image sizes: 640 vs 1024 vs 1280
   - Evaluate batch size impact on convergence
   - Test epochs: 50 vs 100 vs 150

4. [ ] Model pruning & quantization
   - Explore INT8 quantization for faster inference
   - Test on edge devices (Jetson, Raspberry Pi with Coral TPU)
   - Benchmark inference time vs accuracy tradeoff

**Expected Outcomes**:
- Improved POLAV detection (target: >70% mAP50)
- Larger sr_dauca dataset (target: 200+ images)
- Optimal hyperparameter configuration
- Edge deployment feasibility report

---

### Session 5 - Production Pipeline & Deployment
**Planned Date**: December 20+, 2025
**Status**: 📋 Planned

#### Tasks:
1. [ ] Multi-run training strategy
   - Adapt `yolo_training.py` multi-seed approach
   - Train each model 3-5 times with different seeds
   - Select best performing run
   - Document variance across runs

2. [ ] Inference API development
   - Create FastAPI endpoint for real-time inference
   - Support:
     - Single image upload
     - Batch processing
     - SAHI toggle
     - Confidence threshold adjustment
   - Return: JSON + annotated image + GeoJSON

3. [ ] Integration with farm management system
   - Export prescription maps (VRA zones)
   - Generate spot-spray coordinates
   - Calculate agrochemical savings estimates

4. [ ] Field validation
   - Deploy model on field laptop/tablet
   - Real-time inference during spraying operations
   - Collect ground truth for model validation
   - Measure actual chemical reduction

5. [ ] Documentation & training
   - User manual for operators
   - Troubleshooting guide
   - Model update procedure
   - Data collection protocols

**Expected Outcomes**:
- Production-ready inference pipeline
- Field deployment package
- Operator training materials
- Impact assessment framework

---

### Session 6+ - Continuous Improvement
**Status**: 🔄 Ongoing

#### Long-term Goals:
1. [ ] Multi-season data collection
   - Expand datasets across growing seasons
   - Account for seasonal variations
   - Build robust all-weather models

2. [ ] Additional crop types
   - Expand beyond current 6+1 species
   - Include crop-specific weed profiles
   - Regional adaptation (beyond Santa Rosa)

3. [ ] Integration with other sensors
   - Multispectral imagery (NDVI, NDRE)
   - Hyperspectral cameras
   - LiDAR for 3D structure

4. [ ] Advanced analytics
   - Temporal analysis (weed growth tracking)
   - Predictive modeling (infestation forecasting)
   - Economic impact quantification

5. [ ] Research publication
   - Manuscript preparation
   - Methodology documentation
   - Open-source dataset release (if permitted)

---

## 📊 Current Project Status

### Models
| Model | Dataset | Classes | mAP50 (val) | Precision | Recall | Status |
|-------|---------|---------|-------------|-----------|--------|--------|
| YOLO11l | merge_varios | 6 | 81.9% | 79.9% | - | ✅ Production |
| YOLO11l | sr_dauca | 1 | 86.4% | 84.6% | 79.1% | ✅ Production |
| YOLO11{n,s,m,x} | lentils_v1 | 4 | - | - | - | 📊 Benchmark |

### Datasets
| Dataset | Train | Val | Test | Classes | Resolution | Augmented |
|---------|-------|-----|------|---------|------------|-----------|
| merge_varios | 5,765 | 181 | 140 | 6 | 1024×1024 | ✅ External |
| sr_dauca | 70 | 3 | 1 | 1 | 1024×1024 | ✅ External |
| lentils_v1 | - | - | - | 4 | 1024×1024 | ✅ External |

### Infrastructure
- **Hardware**: Dual NVIDIA RTX 4090 (24GB VRAM each)
- **Training**: DDP multi-GPU, optimized batch sizes per model
- **Methodology**: DLM2 (external pre-augmentation, zero internal augmentation)
- **Framework**: Ultralytics YOLO v8.3.235
- **Python**: >=3.8
- **PyTorch**: >=1.8

---

## 📁 Project Structure

See `PROJECT_STRUCTURE.md` for detailed directory tree and file descriptions.

**Key Locations**:
- **Configs**: `configs/*.yaml` (dataset definitions)
- **Scripts**: `scripts/*.py` (training, evaluation, monitoring)
- **Models**: Root directory `yolo11{n,s,m,l,x}.pt` (24-114 MB)
- **Experiments**: `experiments/{dataset}/{model_config}/` (results, weights)
- **Documentation**: `.claude.md`, `PROJECT_LOG.md`, `documents/*.md`
- **External Review**: `external_review/` (INIA scripts analysis)

---

## 🔗 Important Links

- **GitHub Fork**: https://github.com/lfleon9b/ultralytics_lleon
- **Upstream Ultralytics**: https://github.com/ultralytics/ultralytics
- **External Resources**: https://github.com/maaferna/INIA_Testing_YOLOV8
- **Ultralytics Docs**: https://docs.ultralytics.com/
- **SAHI Documentation**: https://github.com/obss/sahi

---

## 📝 Notes & Lessons Learned

### Methodology Insights:
1. **DLM2 compliance is critical** - Disabling internal augmentation preserves radiometric fidelity for agricultural applications
2. **Class imbalance significantly impacts performance** - POLAV suffers from insufficient training samples
3. **Small datasets require careful validation** - sr_dauca (70 images) shows high performance but limited generalization
4. **Multi-scale validation is valuable** - UAS detections validated against Sentinel-2 NDVI anomalies

### Technical Learnings:
1. **Batch size optimization by model** - Essential for dual GPU efficiency
   - YOLO11x: 4-8 batch size
   - YOLO11l: 8-10 batch size
   - YOLO11m: 16 batch size
   - YOLO11{s,n}: 24 batch size

2. **Git workflow for forks**:
   - `origin` → your fork (push here)
   - `upstream` → original repo (pull updates)
   - Regular merges: `git fetch upstream && git merge upstream/main`

3. **Documentation is key** - `.claude.md` enables context preservation across sessions

### Community Resources:
1. **External scripts** provide valuable utilities (SAHI, geospatial, visualization)
2. **Code reuse** accelerates development vs writing from scratch
3. **Review before integration** - Understand dependencies and adaptation needs

---

## 👥 Contributors

- **Project Lead**: Leonardo Leon (lfleon9b)
- **Technical Support**: Claude Code AI Assistant
- **External Resources**: Miguel Fernández (maaferna) - INIA Testing YOLO
- **Framework**: Ultralytics Team

---

## 📄 License

**AGPL-3.0** - Inherited from Ultralytics YOLO
See: https://ultralytics.com/license

For commercial use, consider Ultralytics Enterprise License.

---

**Last Updated**: December 16, 2025
**Next Review**: December 17, 2025 (Session 3)

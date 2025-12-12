# Experiment Log: Lentils Detection (V1)

**Dataset:** `datasets/lentils_v1`  
**Image Size:** 1024x1024  
**Hardware:** 2x NVIDIA RTX 4090 (24GB VRAM each)  
**Strategy:** Distributed Data Parallel (DDP) across 2 GPUs (`device=0,1`)

## Batch Size Optimization Strategy

Due to the large input size (`1024px`) and the high VRAM requirements of larger YOLO11 models, batch sizes were dynamically adjusted to prevent `CUDA OutOfMemory` errors while maintaining high GPU utilization (target: 80-90%).

**Calculation Logic:**
- **Base Consumption:** 1024x1024 images require ~4x more activation memory than standard 640x640.
- **Constraints:** Total VRAM per GPU is 24GB. System overhead is ~1GB. Safe limit is ~21-22GB.
- **Split:** Total batch size is divided by 2 (e.g., Batch 32 = 16 images per GPU).

| Model Variant | Params (M) | Batch per GPU | **Total Batch** | Estimated VRAM | Reasoning |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **yolo11n** (Nano) | ~2.6 | 32 | **64** | Low | Limited by CPU data loading speed rather than VRAM. |
| **yolo11s** (Small) | ~9.4 | 24 | **48** | Medium | Balanced throughput and memory usage. |
| **yolo11m** (Medium) | ~20.1 | 16 | **32** | High | Fits comfortably. Standard setting for 4090s. |
| **yolo11l** (Large) | ~25.3 | 8 | **16** | High | Reduced to prevent OOM. 16 images/GPU would likely crash. |
| **yolo11x** (XLarge) | ~56.9 | 4 | **8** | High | Massive model. Strictly limited to 4 images/GPU to ensure stability. |

## Experiment Results

| Date | Experiment Name | Model | Batch | Epochs | mAP50 | mAP50-95 | Loss (Final) | Notes |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| | `yolo11n_img1024_ep100` | Nano | 64 | 100 | | | | |
| | `yolo11s_img1024_ep100` | Small | 48 | 100 | | | | |
| | `yolo11m_img1024_ep100` | Medium | 32 | 100 | | | | |
| | `yolo11l_img1024_ep100` | Large | 16 | 100 | | | | |
| | `yolo11x_img1024_ep100` | XLarge | 8 | 100 | | | | |

## Observations

*   **Nano/Small**: Expected to train very fast but might struggle with small/dense objects at 1024px.
*   **Medium**: Usually the "sweet spot" for accuracy vs speed.
*   **Large/XLarge**: Should provide best accuracy but training is significantly slower. Watch for overfitting if dataset is small.

## Commands Used

**Run All Experiments:**
python scripts/run_experiments.py**Live Monitoring:**
python scripts/live_dashboard.py

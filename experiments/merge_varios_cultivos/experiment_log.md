# Experiment Log: Multi-Crop Weed Detection (Merge Varios)

**Date:** 2025-12-11  
**Dataset:** `datasets/merge_varios_cultivos` (5765 Train / 181 Val / 140 Test)  
**Classes:** 6 (AMBEL, LENCU, LOLSS, POLAV, POLPE, RAPRA)  
**Image Size:** 1024x1024  
**Hardware:** 2x NVIDIA RTX 4090 (DDP `device=0,1`)

## Methodology (DLM2 Aligned)

*   **Model**: YOLO11l (Large)
*   **Epochs**: 50
*   **Batch Size**: 10 (per GPU split)
*   **Augmentation**: **Disabled** (Internal YOLO augmentation turned off to rely on pre-augmented dataset).
*   **Optimizer**: Auto (AdamW)

## Results (Test Set)

| Class | mAP50-95 | mAP50 | Precision | Recall | F1 Score |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **ALL** | **0.4657** | **0.8190** | **0.7987** | **0.7797** | **0.7884** |
| AMBEL | 0.5368 | 0.8912 | 0.8374 | 0.8413 | 0.8393 |
| LENCU | 0.6111 | 0.9431 | 0.8695 | 0.9324 | 0.8999 |
| LOLSS | 0.3938 | 0.7756 | 0.7719 | 0.7328 | 0.7519 |
| POLAV | 0.2383 | 0.5700 | 0.6128 | 0.5360 | 0.5718 |
| POLPE | 0.4511 | 0.8576 | 0.8524 | 0.7996 | 0.8252 |
| RAPRA | 0.5633 | 0.8762 | 0.8484 | 0.8358 | 0.8420 |

## Observations

*   **Performance Jump**: Compared to the single-crop dataset (`lentils_v1`), the overall mAP50 improved significantly (~0.73 -> 0.82) and mAP50-95 (~0.41 -> 0.47). This confirms that the larger dataset (5k images) provides better generalization.
*   **Lentils (LENCU)**: The model is extremely reliable at detecting the crop itself (F1 ~0.90), which is crucial for variable rate application maps (to avoid spraying the crop).
*   **Weed Detection**:
    *   **AMBEL** and **RAPRA** are detected with high confidence (>0.87 mAP50).
    *   **POLAV** remains the most challenging weed (0.57 mAP50), likely due to morphological similarity or size/occlusion issues.
*   **Stability**: Training with `batch=10` on YOLO11l was stable and completed in ~1.6 hours.

## Artifacts
*   **Weights**: `experiments/merge_varios_cultivos/yolo11l_img1024_ep50_noaug/weights/best.pt`
*   **Full Report**: `experiments/merge_varios_cultivos/final_evaluation_report.csv`


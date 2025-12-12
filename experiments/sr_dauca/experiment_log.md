# Experiment Log: Santa Rosa Dauca (sr_dauca)

**Date:** 2025-12-11  
**Dataset:** `datasets/sr_dauca` (70 Train / 3 Val / 1 Test)  
**Classes:** 1 (DAUCA)  
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
| **ALL** | **0.3676** | **0.8635** | **0.8459** | **0.7914** | **0.8178** |
| DAUCA | 0.3676 | 0.8635 | 0.8459 | 0.7914 | 0.8178 |

## Observations

*   **Dataset Limitation**: The dataset is very small (70 train images) with an extremely limited test set (only 1 image).
*   **High Precision**: Despite the small size, the model achieved 85% Precision on the test sample, suggesting it learned the specific features of `DAUCA` well.
*   **Rapid Convergence**: Training completed in < 2 minutes due to dataset size.
*   **Recommendation**: To ensure robustness, we strongly recommend expanding the test set to at least 10-20 images to validate these metrics across different conditions.

## Artifacts
*   **Weights**: `experiments/sr_dauca/yolo11l_img1024_ep50_noaug/weights/best.pt`
*   **Full Report**: `experiments/sr_dauca/final_evaluation_report.csv`


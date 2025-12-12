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

## Results

### Test Set (1 Image)
| Class | mAP50-95 | mAP50 | Precision | Recall | F1 Score |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **ALL** | **0.3676** | **0.8635** | **0.8459** | **0.7914** | **0.8178** |
| DAUCA | 0.3676 | 0.8635 | 0.8459 | 0.7914 | 0.8178 |

### Validation Set (3 Images)
| Class | mAP50-95 | mAP50 | Precision | Recall |
| :--- | :--- | :--- | :--- | :--- |
| **ALL** | **0.3678** | **0.8050** | **0.8021** | **0.7327** |

## Observations

*   **Dataset Limitation**: The dataset is extremely small (70 train images) with limited evaluation data (3 val, 1 test). Metrics are indicative but may not be statistically robust.
*   **Performance**:
    *   **Precision (80-85%)**: High precision indicates the model is conservative and accurate when it detects a weed.
    *   **mAP50 (80-86%)**: Strong detection capability for `DAUCA` at standard IoU thresholds.
    *   **Consistency**: Validation and Test metrics are very close, suggesting the model has learned generalizable features even from few samples.
*   **Rapid Convergence**: Training completed in < 2 minutes due to dataset size.
*   **Recommendation**: To ensure robustness for field deployment, expanding the test set to at least 10-20 images is strongly recommended.

## Artifacts
*   **Weights**: `experiments/sr_dauca/yolo11l_img1024_ep50_noaug/weights/best.pt`
*   **Full Report**: `experiments/sr_dauca/final_evaluation_report.csv`

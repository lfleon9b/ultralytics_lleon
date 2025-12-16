import os
import pandas as pd
from ultralytics import YOLO
from pathlib import Path

# Configuration
PROJECT_DIR = "experiments/merge_varios_cultivos"
MODEL_PATH = os.path.join(PROJECT_DIR, "yolo11l_img1024_ep50_noaug/weights/best.pt")
DATA_CONFIG = "configs/merge_varios_cultivos.yaml"
IMGSZ = 1024
DEVICE = "0"  # Use single GPU for stable inference

def run_evaluation(split_name):
    print(f"\n>>> Starting evaluation on {split_name.upper()} set...")
    
    model = YOLO(MODEL_PATH)
    
    metrics = model.val(
        data=DATA_CONFIG,
        split=split_name,
        imgsz=IMGSZ,
        device=DEVICE,
        batch=16,
        save_json=True,
        plots=True,
        project=PROJECT_DIR,
        name=f"eval_{split_name}",
        exist_ok=True
    )
    
    results = []
    
    # General stats
    results.append({
        "Split": split_name,
        "Class": "ALL",
        "mAP50-95": round(metrics.box.map, 4),
        "mAP50": round(metrics.box.map50, 4),
        "Precision": round(metrics.box.mp, 4),
        "Recall": round(metrics.box.mr, 4),
        "F1": round(metrics.box.f1.mean(), 4)
    })
    
    # Per-class stats
    for class_idx, class_name in metrics.names.items():
        try:
            p = metrics.box.p[class_idx]
            r = metrics.box.r[class_idx]
            ap50 = metrics.box.ap50[class_idx]
            ap = metrics.box.ap[class_idx]
            f1 = 2 * (p * r) / (p + r + 1e-16)
            
            results.append({
                "Split": split_name,
                "Class": class_name,
                "mAP50-95": round(ap, 4),
                "mAP50": round(ap50, 4),
                "Precision": round(p, 4),
                "Recall": round(r, 4),
                "F1": round(f1, 4)
            })
        except:
            pass
            
    return results

if __name__ == "__main__":
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model not found at {MODEL_PATH}")
        exit(1)

    all_data = []
    
    # Evaluate on Validation Set
    val_res = run_evaluation("val")
    all_data.extend(val_res)
    
    # Evaluate on Test Set
    test_res = run_evaluation("test")
    all_data.extend(test_res)
    
    # Save Report
    df = pd.DataFrame(all_data)
    report_path = os.path.join(PROJECT_DIR, "final_evaluation_report.csv")
    df.to_csv(report_path, index=False)
    
    print("\n" + "="*60)
    print(f"Evaluation Finished. Report saved to: {report_path}")
    print("="*60)
    print(df.to_string())


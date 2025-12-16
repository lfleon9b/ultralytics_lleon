import os
import glob
import pandas as pd
import json
from ultralytics import YOLO
from pathlib import Path

# Configuration
EXPERIMENTS_DIR = "experiments/lentils_v1_dlm2"
DATA_CONFIG = "configs/lentils_v1.yaml"
OUTPUT_REPORT = "test_metrics_by_class.csv"
IMGSZ = 1024
DEVICE = "0,1"  # Use multiple GPUs for validation too

def evaluate_models():
    # Find all completed experiments (must have weights/best.pt)
    experiment_paths = glob.glob(os.path.join(EXPERIMENTS_DIR, "*", "weights", "best.pt"))
    
    if not experiment_paths:
        print(f"No completed models found in {EXPERIMENTS_DIR}")
        return

    all_results = []

    print(f"Found {len(experiment_paths)} models to evaluate on TEST set...")

    for model_path in sorted(experiment_paths):
        # Extract experiment name
        exp_name = Path(model_path).parent.parent.name
        print(f"\n>>> Evaluating: {exp_name}")
        
        try:
            # Load model
            model = YOLO(model_path)
            
            # Run Validation on TEST split
            # Note: We use split='test'. If your data.yaml doesn't have 'test:', it falls back to val.
            # Your config has: test: test/images
            metrics = model.val(
                data=DATA_CONFIG,
                split='test',
                imgsz=IMGSZ,
                device=DEVICE,
                batch=16, # Safe batch size for validation
                save_json=True,
                plots=True
            )
            
            # --- Extract General Metrics ---
            # map50-95, map50, precision, recall
            res_general = {
                "Model": exp_name,
                "Class": "ALL",
                "mAP50-95": round(metrics.box.map, 4),
                "mAP50": round(metrics.box.map50, 4),
                "Precision": round(metrics.box.mp, 4),
                "Recall": round(metrics.box.mr, 4),
                "F1": round(metrics.box.f1.mean(), 4) # Approximate mean F1
            }
            all_results.append(res_general)
            
            # --- Extract Per-Class Metrics ---
            # metrics.box.maps is an array of mAP50-95 per class
            # metrics.names is a dict of class names {0: 'AMBEL', 1: 'LENCU'...}
            
            # We need to dig deeper for P/R per class if available in the object attributes
            # Usually metrics.box.p and metrics.box.r are arrays matching class indices
            
            for class_idx, class_name in metrics.names.items():
                try:
                    p = metrics.box.p[class_idx]
                    r = metrics.box.r[class_idx]
                    ap50 = metrics.box.ap50[class_idx]
                    ap = metrics.box.ap[class_idx] # mAP50-95
                    
                    # Calculate F1 for this class
                    # F1 = 2 * (P * R) / (P + R)
                    f1 = 2 * (p * r) / (p + r + 1e-16)

                    res_class = {
                        "Model": exp_name,
                        "Class": class_name,
                        "mAP50-95": round(ap, 4),
                        "mAP50": round(ap50, 4),
                        "Precision": round(p, 4),
                        "Recall": round(r, 4),
                        "F1": round(f1, 4)
                    }
                    all_results.append(res_class)
                except Exception as e:
                    print(f"Could not extract metrics for class {class_name}: {e}")

        except Exception as e:
            print(f"!!! Error evaluating {exp_name}: {e}")

    # Create DataFrame and Save
    if all_results:
        df = pd.DataFrame(all_results)
        
        # Sort nicely
        df = df.sort_values(by=["Model", "Class"])
        
        # Move 'ALL' rows to the top of each model group? Or leave as is.
        # Let's save it.
        save_path = os.path.join(EXPERIMENTS_DIR, OUTPUT_REPORT)
        df.to_csv(save_path, index=False)
        
        print("\n" + "="*50)
        print(f"Evaluation Complete. Report saved to: {save_path}")
        print("="*50)
        print(df.to_string())
    else:
        print("No results generated.")

if __name__ == "__main__":
    evaluate_models()


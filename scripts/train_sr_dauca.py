import subprocess
import os

# Configuration
PROJECT_NAME = "experiments/sr_dauca"
DATA_CONFIG = "configs/sr_dauca.yaml"
EPOCHS = 50
IMGSZ = 1024
DEVICE = "0,1" 

# Setup for YOLO11l (Large)
MODEL = "yolo11l.pt"
BATCH_SIZE = 10 # Optimized for L on 2x4090

def run_training():
    model_name = MODEL.replace(".pt", "")
    run_name = f"{model_name}_img{IMGSZ}_ep{EPOCHS}_noaug"
    
    print(f"\n>>> Starting training: {run_name} on {DATA_CONFIG}")
    
    cmd = [
        "yolo", "detect", "train",
        f"project={PROJECT_NAME}",
        f"name={run_name}",
        f"data={DATA_CONFIG}",
        f"model={MODEL}",
        f"epochs={EPOCHS}",
        f"imgsz={IMGSZ}",
        f"device={DEVICE}",
        f"batch={BATCH_SIZE}",
        "exist_ok=True",
        # DLM2 Methodology: Disable internal augmentation
        "augment=False",
        "mosaic=0.0",
        "mixup=0.0",
        "hsv_h=0.0",
        "hsv_s=0.0",
        "hsv_v=0.0",
        "degrees=0.0",
        "translate=0.0",
        "scale=0.0",
        "shear=0.0",
        "perspective=0.0",
        "flipud=0.0",
        "fliplr=0.0",
        "erasing=0.0",
        "copy_paste=0.0"
    ]
    
    print("Command:", " ".join(cmd))
    
    try:
        subprocess.run(cmd, check=True)
        print(f">>> Success! Results saved to {PROJECT_NAME}/{run_name}")
    except subprocess.CalledProcessError as e:
        print(f"!!! Training failed: {e}")

if __name__ == "__main__":
    os.makedirs(PROJECT_NAME, exist_ok=True)
    run_training()

import subprocess
import os

# Configuration
PROJECT_NAME = "experiments/lentils_v1_dlm2" # New project folder for DLM2 aligned runs
DATA_CONFIG = "configs/lentils_v1.yaml"
EPOCHS = 50 # Reduced to 50 to match DLM2 methodology
IMGSZ = 1024
DEVICE = "0,1"  # Use "0,1" for both GPUs, or "0" for single

# Model-specific batch sizes based on DLM2 Appendix and optimization
# Document values: X=8, L=10, M=14, S=14, N=14 (conservative)
# We will use slightly optimized values for 4090s while keeping X safe.
# REDUCED FURTHER to prevent "ConnectionResetError" / OOM / DDP crashes
BATCH_SIZES = {
    "yolo11n.pt": 24, # Reduced from 32
    "yolo11s.pt": 24, # Reduced from 32
    "yolo11m.pt": 16, # Kept at 16 (Safe)
    "yolo11l.pt": 8,  # Reduced from 10 (Safe)
    "yolo11x.pt": 4   # Reduced from 8 (Very Safe for 1024px)
}

# List of models to test
models = [
    "yolo11n.pt",
    "yolo11s.pt", 
    "yolo11m.pt",
    "yolo11l.pt",
    "yolo11x.pt"
]

def run_experiment(model_weight):
    model_name = model_weight.replace(".pt", "")
    run_name = f"{model_name}_img{IMGSZ}_ep{EPOCHS}_noaug"
    
    # Determine batch size dynamically
    current_batch = BATCH_SIZES.get(model_weight, 16)
    
    print(f"\n>>> Starting experiment: {run_name} with batch={current_batch}")
    
    cmd = [
        "yolo", "detect", "train",
        f"project={PROJECT_NAME}",
        f"name={run_name}",
        f"data={DATA_CONFIG}",
        f"model={model_weight}",
        f"epochs={EPOCHS}",
        f"imgsz={IMGSZ}",
        f"device={DEVICE}",
        f"batch={current_batch}",
        "exist_ok=True",
        # DLM2 Methodology: Disable internal augmentation (pre-augmented dataset)
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
    
    # Join command for display
    print("Command:", " ".join(cmd))
    
    # Run command
    try:
        subprocess.run(cmd, check=True)
        print(f">>> Finished experiment: {run_name}")
    except subprocess.CalledProcessError as e:
        print(f"!!! Error running {run_name}: {e}")

if __name__ == "__main__":
    # Ensure directories exist
    os.makedirs(PROJECT_NAME, exist_ok=True)
    
    for model in models:
        run_experiment(model)

    print("\nAll experiments completed.")

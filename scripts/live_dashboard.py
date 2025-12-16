import time
import pandas as pd
import matplotlib.pyplot as plt
import os
import pynvml
from pathlib import Path
from datetime import datetime

# Configuration
EXPERIMENTS_DIR = "/home/malezainia1/dev/ultralytics/experiments/sr_dauca"
REFRESH_INTERVAL = 5  # Seconds (faster for GPU stats)
OUTPUT_FILE = "live_dashboard.png"

# Initialize NVML for GPU stats
try:
    pynvml.nvmlInit()
    GPU_COUNT = pynvml.nvmlDeviceGetCount()
    HAS_GPU = True
except Exception as e:
    print(f"Warning: Could not initialize NVML for GPU monitoring: {e}")
    HAS_GPU = False
    GPU_COUNT = 0

# Data storage for GPU history
gpu_history = {i: {'time': [], 'util': [], 'mem': []} for i in range(GPU_COUNT)}
start_time = time.time()

def get_gpu_stats():
    if not HAS_GPU:
        return {}
    
    stats = {}
    current_t = time.time() - start_time
    
    for i in range(GPU_COUNT):
        handle = pynvml.nvmlDeviceGetHandleByIndex(i)
        util = pynvml.nvmlDeviceGetUtilizationRates(handle).gpu
        mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        mem_used_gb = mem_info.used / 1024**3
        
        # Append to history
        gpu_history[i]['time'].append(current_t)
        gpu_history[i]['util'].append(util)
        gpu_history[i]['mem'].append(mem_used_gb)
        
        # Keep only last 10 minutes of data (approx 120 points at 5s interval)
        if len(gpu_history[i]['time']) > 120:
            gpu_history[i]['time'].pop(0)
            gpu_history[i]['util'].pop(0)
            gpu_history[i]['mem'].pop(0)
            
    return stats

def get_training_data(root_dir):
    experiments_path = Path(root_dir)
    results_files = list(experiments_path.rglob("results.csv"))
    data = {}
    
    for csv_file in results_files:
        try:
            exp_name = csv_file.parent.name
            if exp_name == 'train':
                 exp_name = csv_file.parent.parent.name
            
            df = pd.read_csv(csv_file)
            df.columns = [c.strip() for c in df.columns]
            data[exp_name] = df
        except Exception:
            pass
    return data

def update_dashboard(root_dir):
    # 1. Collect Data
    training_data = get_training_data(root_dir)
    get_gpu_stats()
    
    # 2. Setup Plot Grid (2x2)
    # Top Row: Training Metrics
    # Bottom Row: GPU Stats
    plt.clf()
    fig = plt.gcf()
    
    ax_map = fig.add_subplot(221)
    ax_loss = fig.add_subplot(222)
    ax_gpu_util = fig.add_subplot(223)
    ax_gpu_mem = fig.add_subplot(224)

    # --- Plot 1: mAP50 ---
    if training_data:
        for exp_name, df in training_data.items():
            if 'metrics/mAP50(B)' in df.columns:
                ax_map.plot(df['epoch'], df['metrics/mAP50(B)'], label=exp_name, linewidth=2)
    
    ax_map.set_title("Validation mAP50")
    ax_map.set_xlabel("Epoch")
    ax_map.set_ylabel("mAP50")
    ax_map.grid(True, linestyle='--', alpha=0.5)
    ax_map.legend(fontsize='x-small')

    # --- Plot 2: Loss ---
    if training_data:
        for exp_name, df in training_data.items():
            loss_cols = ['train/box_loss', 'train/cls_loss', 'train/dfl_loss']
            available_loss = [c for c in loss_cols if c in df.columns]
            if available_loss:
                df['total_loss'] = df[available_loss].sum(axis=1)
                ax_loss.plot(df['epoch'], df['total_loss'], label=exp_name, linewidth=2)
    
    ax_loss.set_title("Training Loss")
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("Loss")
    ax_loss.grid(True, linestyle='--', alpha=0.5)

    # --- Plot 3: GPU Utilization ---
    if HAS_GPU:
        for i in range(GPU_COUNT):
            if gpu_history[i]['time']:
                ax_gpu_util.plot(gpu_history[i]['time'], gpu_history[i]['util'], label=f"GPU {i}")
        
        ax_gpu_util.set_title("GPU Utilization (%) - Last 10m")
        ax_gpu_util.set_xlabel("Time (s)")
        ax_gpu_util.set_ylabel("Util %")
        ax_gpu_util.set_ylim(0, 105)
        ax_gpu_util.grid(True, linestyle='--', alpha=0.5)
        ax_gpu_util.legend(fontsize='x-small')

    # --- Plot 4: GPU Memory ---
    if HAS_GPU:
        for i in range(GPU_COUNT):
            if gpu_history[i]['time']:
                ax_gpu_mem.plot(gpu_history[i]['time'], gpu_history[i]['mem'], label=f"GPU {i}")
        
        ax_gpu_mem.set_title("GPU Memory Usage (GB) - Last 10m")
        ax_gpu_mem.set_xlabel("Time (s)")
        ax_gpu_mem.set_ylabel("GB Used")
        ax_gpu_mem.grid(True, linestyle='--', alpha=0.5)
        ax_gpu_mem.legend(fontsize='x-small')

    plt.tight_layout()
    plt.draw()
    
    # Save to file
    save_path = Path(root_dir) / OUTPUT_FILE
    plt.savefig(save_path)
    # print(f"Updated dashboard saved to {save_path}")

def main():
    print(f"Monitoring {EXPERIMENTS_DIR}...")
    print(f"Detected {GPU_COUNT} GPUs.")
    print(f"Dashboard updating every {REFRESH_INTERVAL}s -> {EXPERIMENTS_DIR}/{OUTPUT_FILE}")
    
    plt.ion()
    plt.figure(figsize=(16, 10))

    try:
        while True:
            update_dashboard(EXPERIMENTS_DIR)
            plt.pause(REFRESH_INTERVAL)
    except KeyboardInterrupt:
        print("\nStopped monitoring.")
    finally:
        pynvml.nvmlShutdown()

if __name__ == "__main__":
    if not os.path.exists(EXPERIMENTS_DIR):
        print(f"Waiting for directory {EXPERIMENTS_DIR}...")
        while not os.path.exists(EXPERIMENTS_DIR):
            time.sleep(5)
            
    main()


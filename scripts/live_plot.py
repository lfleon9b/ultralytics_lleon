import time
import pandas as pd
import matplotlib.pyplot as plt
import os
from pathlib import Path

# Configuration
EXPERIMENTS_DIR = "/home/malezainia1/dev/ultralytics/experiments/lentils_v1"
REFRESH_INTERVAL = 10  # Seconds
OUTPUT_FILE = "live_comparison.png"

def get_data(root_dir):
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

def update_plot(root_dir):
    data = get_data(root_dir)
    if not data:
        print("No data found yet...")
        return

    plt.clf() # Clear the figure
    
    # Setup subplots
    fig = plt.gcf()
    if not fig.get_axes():
         # Initialize layout if cleared
         ax1 = fig.add_subplot(121)
         ax2 = fig.add_subplot(122)
    else:
        ax1, ax2 = fig.axes

    ax1.cla()
    ax2.cla()

    # Plot data
    for exp_name, df in data.items():
        # mAP50
        if 'metrics/mAP50(B)' in df.columns:
            ax1.plot(df['epoch'], df['metrics/mAP50(B)'], label=exp_name, linewidth=2)
        
        # Loss
        loss_cols = ['train/box_loss', 'train/cls_loss', 'train/dfl_loss']
        available_loss = [c for c in loss_cols if c in df.columns]
        if available_loss:
            df['total_loss'] = df[available_loss].sum(axis=1)
            ax2.plot(df['epoch'], df['total_loss'], label=exp_name, linewidth=2)

    # Styling
    ax1.set_title("Real-time: mAP50")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("mAP50")
    ax1.grid(True, linestyle='--', alpha=0.5)
    ax1.legend(loc='lower right', fontsize='small')

    ax2.set_title("Real-time: Training Loss")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Loss")
    ax2.grid(True, linestyle='--', alpha=0.5)
    ax2.legend(loc='upper right', fontsize='small')

    plt.tight_layout()
    plt.draw()
    
    # Also save to file for remote viewing (VS Code image preview auto-reloads)
    save_path = Path(root_dir) / OUTPUT_FILE
    plt.savefig(save_path)
    print(f"Updated plot saved to {save_path}")

def main():
    print(f"Monitoring {EXPERIMENTS_DIR}...")
    print(f"Plots will update every {REFRESH_INTERVAL} seconds.")
    
    # Turn on interactive mode
    plt.ion()
    plt.figure(figsize=(15, 6))

    try:
        while True:
            update_plot(EXPERIMENTS_DIR)
            plt.pause(REFRESH_INTERVAL)
    except KeyboardInterrupt:
        print("\nStopped monitoring.")

if __name__ == "__main__":
    # Check if directory exists
    if not os.path.exists(EXPERIMENTS_DIR):
        print(f"Waiting for directory {EXPERIMENTS_DIR} to be created...")
        while not os.path.exists(EXPERIMENTS_DIR):
            time.sleep(5)
            
    main()


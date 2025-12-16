import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import glob
import os

def plot_comparisons(experiments_root, output_file="comparison_plots.png"):
    """
    Plots mAP50 and Loss curves for all experiments found in experiments_root.
    """
    experiments_path = Path(experiments_root)
    results_files = list(experiments_path.rglob("results.csv"))

    if not results_files:
        print("No results.csv files found to plot.")
        return

    # Create a figure with two subplots: mAP50 and Loss
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Track if we actually plotted anything
    plotted = False

    for csv_file in results_files:
        try:
            # Determine experiment name from parent folder
            exp_name = csv_file.parent.name
            if exp_name == 'train':
                 exp_name = csv_file.parent.parent.name

            # Read CSV
            df = pd.read_csv(csv_file)
            df.columns = [c.strip() for c in df.columns] # Clean whitespace

            # Check for required columns
            # mAP50 is typically 'metrics/mAP50(B)'
            # Loss sum is typically sum of 'train/box_loss', 'train/cls_loss', 'train/dfl_loss'
            
            map_col = 'metrics/mAP50(B)'
            loss_cols = ['train/box_loss', 'train/cls_loss', 'train/dfl_loss']
            
            if map_col in df.columns:
                ax1.plot(df['epoch'], df[map_col], label=exp_name, linewidth=2)
                plotted = True
            
            # Calculate total training loss for the loss plot
            available_loss_cols = [c for c in loss_cols if c in df.columns]
            if available_loss_cols:
                df['total_loss'] = df[available_loss_cols].sum(axis=1)
                ax2.plot(df['epoch'], df['total_loss'], label=exp_name, linewidth=2)
                plotted = True

        except Exception as e:
            print(f"Could not process {csv_file}: {e}")

    if not plotted:
        print("Could not extract data for plotting.")
        return

    # Formatting Plot 1: mAP50
    ax1.set_title("Validation mAP50 over Epochs")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("mAP50")
    ax1.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax1.legend()

    # Formatting Plot 2: Total Loss
    ax2.set_title("Training Loss (Box + Cls + DFL) over Epochs")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Total Loss")
    ax2.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax2.legend()

    plt.tight_layout()
    
    # Save the plot
    output_path = experiments_path / output_file
    plt.savefig(output_path, dpi=300)
    print(f"Comparison plot saved to: {output_path}")

if __name__ == "__main__":
    # Adjust path if needed
    experiments_dir = "/home/malezainia1/dev/ultralytics/experiments/lentils_v1"
    
    # Ensure the directory exists (it might not if no experiments run yet)
    if os.path.exists(experiments_dir):
        plot_comparisons(experiments_dir)
    else:
        print(f"Directory {experiments_dir} does not exist yet. Run experiments first.")


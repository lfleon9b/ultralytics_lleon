import os
import glob
import pandas as pd
from pathlib import Path

def get_latest_results(experiment_dir):
    """Finds the results.csv file in the experiment directory."""
    # The standard YOLO structure is project/name/results.csv
    # But sometimes it might be project/name/train/results.csv depending on how it was run
    # We will search recursively
    results_files = list(Path(experiment_dir).rglob("results.csv"))
    if not results_files:
        return None
    # If multiple, take the one in the root of the specific run or the most recent one?
    # Usually there is only one per 'name' unless resume was used.
    # Let's take the first one found for now.
    return results_files[0]

def parse_results(csv_path):
    """Parses the last row of the results.csv to get final metrics."""
    try:
        df = pd.read_csv(csv_path)
        df.columns = [c.strip() for c in df.columns] # Clean whitespace
        if df.empty:
            return {}
        
        # Get the row with the best mAP50-95, or just the last epoch? 
        # Usually best.pt is saved based on fitness (weighted combination of metrics).
        # Let's just grab the last epoch for simplicity of "final result" or max mAP.
        # Ultralytics results.csv usually has columns: 
        # epoch, train/box_loss, ..., metrics/precision(B), metrics/recall(B), metrics/mAP50(B), metrics/mAP50-95(B), ...
        
        # We will extract the max value for mAP50-95 and corresponding other metrics, 
        # or the last epoch. Let's take the row with max mAP50-95.
        best_idx = df['metrics/mAP50-95(B)'].idxmax()
        best_row = df.iloc[best_idx]
        
        metrics = {
            'epoch': int(best_row['epoch']),
            'precision': best_row.get('metrics/precision(B)', 0),
            'recall': best_row.get('metrics/recall(B)', 0),
            'mAP50': best_row.get('metrics/mAP50(B)', 0),
            'mAP50-95': best_row.get('metrics/mAP50-95(B)', 0),
        }
        return metrics
    except Exception as e:
        print(f"Error parsing {csv_path}: {e}")
        return {}

def compile_comparison(experiments_root_dir):
    """Compiles metrics from all experiments in the directory."""
    experiments_root = Path(experiments_root_dir)
    if not experiments_root.exists():
        print(f"Directory {experiments_root} does not exist.")
        return

    all_data = []
    
    # Iterate over directories in experiments_root
    # Assuming structure: experiments/dataset_version/experiment_name
    # Or just experiments/experiment_name if flat. 
    # Based on our previous plan: experiments/lentils_v1/yolo11n...
    
    # Let's search for all directories that look like run directories
    # We can assume any subdirectory in experiments_root might be a project containing runs, 
    # or a run itself.
    
    # Strategy: Walk through and look for results.csv, then infer experiment name from parent folder
    for csv_file in experiments_root.rglob("results.csv"):
        experiment_name = csv_file.parent.name
        # If the parent is 'train' (default yolo name), go one level up
        if experiment_name == 'train': 
             experiment_name = csv_file.parent.parent.name
        
        project_name = csv_file.parent.parent.name
        if project_name == 'train': # handle nested case if needed
             project_name = csv_file.parent.parent.parent.name

        metrics = parse_results(csv_file)
        if metrics:
            metrics['experiment'] = experiment_name
            # metrics['project'] = project_name
            all_data.append(metrics)

    if not all_data:
        print("No results found.")
        return

    # Create DataFrame
    comparison_df = pd.DataFrame(all_data)
    
    # Reorder columns
    cols = ['experiment', 'mAP50-95', 'mAP50', 'precision', 'recall', 'epoch']
    # Add any other columns found
    remaining_cols = [c for c in comparison_df.columns if c not in cols]
    comparison_df = comparison_df[cols + remaining_cols]
    
    # Sort by mAP50-95 descending
    comparison_df = comparison_df.sort_values(by='mAP50-95', ascending=False)
    
    print("\n=== Experiment Comparison ===")
    print(comparison_df.to_string(index=False))
    
    # Save to CSV
    output_path = experiments_root / "comparison_report.csv"
    comparison_df.to_csv(output_path, index=False)
    print(f"\nReport saved to {output_path}")

if __name__ == "__main__":
    # Adjust this path to your experiments folder
    experiments_dir = "/home/malezainia1/dev/ultralytics/experiments"
    compile_comparison(experiments_dir)


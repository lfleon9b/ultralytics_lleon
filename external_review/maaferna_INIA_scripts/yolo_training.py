import json
import os
import logging
from datetime import datetime
import time
import random
import torch
from ultralytics import YOLO
from clearml_utils import start_clearml_task, log_clearml_result




def train_yolo(config, model_version, batch_size, img_size, devices, num_runs=5, base_seed=42, epochs=50, yolo_version=8):
    """
    Train a specific YOLO model version multiple times using both GPUs without DDP.

    Args:
    config (dict): Configuration dictionary
    model_version (str): YOLO model version ('n', 's', 'm', 'l', 'x')
    batch_size (int): Batch size for training
    img_size (int): Image size for training
    num_runs (int): Number of training runs
    base_seed (int): Base random seed
    epochs (int): Number of training epochs
    devices (list): List of GPUs to use (should be ['0', '1'] for dual-GPU setup)

    Returns:
    dict: Best results and corresponding seed
    """
    base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'training_results', yolo_version)
    img_size_dir = os.path.join(base_dir, f'img_size_{img_size}')
    model_dir = os.path.join(img_size_dir, f'yolo{yolo_version}_{model_version}')
    os.makedirs(model_dir, exist_ok=True)

    best_map50 = 0
    best_results = None
    best_seed = None
    all_results = []

    for run in range(1, num_runs + 1):
        run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = os.path.join(model_dir, f'{run_timestamp}_run_{run}')
        os.makedirs(run_dir, exist_ok=True)

        # Generate a unique task name for each run
        task_name = f"YOLOv{yolo_version}_{model_version}_training_{img_size}x{img_size}_run_{run}_{run_timestamp}"

        # Use the start_clearml_task function to create a new ClearML task for each run
        task = start_clearml_task(
            name=task_name,
            description=f"Training YOLOv{yolo_version} {model_version} with image size {img_size} using multiple GPUs",
            project_name="YOLOv8 Research"
        )
        
        time.sleep(2)  # Brief delay to ensure task synchronization with ClearML
        if not task:
            raise ValueError(f"Failed to create ClearML task: {task_name}")
        
        try:
            run_seed = int(time.time() * 1000) % (2**32 - 1)
            random.seed(run_seed)
            torch.manual_seed(run_seed)
            logging.info(f"Starting run {run} for YOLOv{yolo_version}_{model_version} with seed {run_seed}")

            model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'models', f'yolo{yolo_version}{model_version}.pt')
            model = YOLO(model_path)

            # Set up multi-GPU training without DDP
            device_list = [int(d) for d in devices]
            device = f'cuda:{device_list[0]}' if len(device_list) == 1 else 'cuda'
            if len(device_list) > 1:
                os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, device_list))

            workers = 8

            # Run model training
            results = model.train(
                data=config["yaml_file"],
                epochs=epochs,
                imgsz=img_size,
                batch=batch_size,
                device=device,
                seed=run_seed,
                project=run_dir,
                name="results",  # Use unique run name
                cache=False,
                workers=workers,
                pretrained=True,
                optimizer='auto',
                verbose=True,
                save_json=True,
                amp=True
            )

            metrics = results.results_dict
            precision = metrics.get('metrics/precision(B)', 0)
            recall = metrics.get('metrics/recall(B)', 0)
            current_map50 = metrics.get('metrics/mAP50(B)', 0)
            current_map50_95 = metrics.get('metrics/mAP50-95(B)', 0)
            f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

            # Save results in the run-specific folder
            json_summary = {
                "yaml_path": config["yaml_file"],
                "confidence_threshold": config.get("confidence_threshold", 0.3),
                "img_size": img_size,
                "training_image_size": config.get("training_image_size", img_size),
                "device": device,
                "model_name": model_version,
                "class_names": config.get("class_names", []),
                "timestamp": run_timestamp,
                "epochs": epochs,
                "run": run,
                "seed": run_seed,
                "batch": batch_size,
                "clearml_task_id": task.id,
                "workers": workers,
                "max_det": config.get("max_det", 300),
                "f1_score": f1_score,
                "precision": precision,
                "recall": recall,
                "current_map50": current_map50,
                "current_map50_95": current_map50_95
            }

            # Save the JSON summary
            json_summary_path = os.path.join(run_dir, 'summary.json')
            with open(json_summary_path, 'w') as json_file:
                json.dump(json_summary, json_file, indent=4)

            # Save results in the JSON format for the run
            results_json_path = os.path.join(run_dir, 'results.json')
            with open(results_json_path, 'w') as results_file:
                json.dump(metrics, results_file, indent=4)

            all_results.append(json_summary)
            logging.info(f"Run {run} completed. mAP@0.5: {current_map50}")

            if current_map50 > best_map50:
                best_map50 = current_map50
                best_results = metrics
                best_seed = run_seed

        except Exception as e:
            logging.error(f"Error in run {run}: {str(e)}")

        finally:
            if task:
                task.close()

    # Save consolidated summary with timestamp
    consolidated_summary_filename = f'training_results_summary_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    consolidated_summary_path = os.path.join(model_dir, consolidated_summary_filename)
    with open(consolidated_summary_path, 'w') as consolidated_file:
        json.dump(all_results, consolidated_file, indent=4)

    logging.info(f"Best mAP@0.5: {best_map50} (seed {best_seed})")
    
    return {"results": best_results, "seed": best_seed}

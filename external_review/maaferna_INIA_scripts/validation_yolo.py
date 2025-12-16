from datetime import datetime
from io import StringIO
import io
import os
import logging
import shutil
import sys
import random
import time

import tempfile
import numpy as np
import torch
from ultralytics import YOLO
import yaml
import json

from clearml_utils import log_clearml_result, start_clearml_task
from utils import  create_excel_from_json, create_flexible_summary, safe_float


logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)


def run_validation_on_image(image_path, label_path, devices, best_model_path, confidence_threshold, img_size, project_root_dir, model_name, labels_dict, accountoutput_dir=None):
    """
    Run validation on a single image, comparing model predictions with the ground truth labels.
    
    Args:
    - image_path (str): Full path to the image.
    - label_path (str): Full path to the label file (ground truth) in .txt format.
    - devices (list or str): List of device IDs for GPUs or 'cpu' for CPU fallback.
    - best_model_path (str): Path to the YOLOv8 model weights.
    - confidence_threshold (float): Confidence threshold for filtering predictions.
    - img_size (int): Size of the image for model inference.
    - project_root_dir (str): Root directory of the project.
    - model_name (str): Name of the model (e.g., 'x', 'l', 'm', etc.)
    - labels_dict (dict): Dictionary mapping class indices to class names.
    - output_dir (str): Directory to save the validation results.
    
    Returns:
    - json_summary (dict): Dictionary containing the validation results and metrics.
    """
    try:
        # Load YOLOv8 model
        model = YOLO(best_model_path)
        
        # Determine the device
        if isinstance(devices, list):
            selected_device = f'cuda:{devices[0]}'
        else:
            selected_device = 'cpu'

        # Convert labels_dict to a list of class names
        class_names = [labels_dict[i] for i in sorted(labels_dict.keys())]

        # Create a temporary directory to act as a mini dataset
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_images_dir = os.path.join(temp_dir, 'images')
            temp_labels_dir = os.path.join(temp_dir, 'labels')
            os.makedirs(temp_images_dir)
            os.makedirs(temp_labels_dir)

            # Copy the image and label to the temporary directory
            temp_image_path = os.path.join(temp_images_dir, os.path.basename(image_path))
            temp_label_path = os.path.join(temp_labels_dir, os.path.basename(label_path))
            shutil.copy(image_path, temp_image_path)
            shutil.copy(label_path, temp_label_path)

            # Prepare a temporary YAML file
            temp_yaml_path = os.path.join(temp_dir, 'data.yaml')
            yaml_data = {
                'path': temp_dir,
                'train': 'images',  # Not used, but required in structure
                'val': 'images',
                'test': 'images',
                'names': class_names
            }
            with open(temp_yaml_path, 'w') as yaml_file:
                yaml.dump(yaml_data, yaml_file)

            # Create output directory
            if output_dir is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_dir = os.path.join(
                    project_root_dir,
                    'valid',
                    'valid_single_image',
                    f'img_size_{img_size}',
                    f'yolov8_{model_name}',
                    f'detect_{timestamp}'
                )
            os.makedirs(output_dir, exist_ok=True)

            # Run validation
            results = model.val(
                data=temp_yaml_path,
                imgsz=img_size,
                conf=confidence_threshold,
                device=selected_device,
                save_txt=True,
                save_conf=True,
                verbose=True,
                project=output_dir,
                save_json=True,
                name=''
            )

        # Extract detailed metrics
        box_metrics = results.box

        # Create flexible summary
        instance_counts = {class_name: 0 for class_name in class_names}
        with open(label_path, 'r') as f:
            for line in f:
                class_id = int(line.split()[0])
                if class_id < len(class_names):
                    class_name = class_names[class_id]
                    instance_counts[class_name] += 1

        flexible_summary = create_flexible_summary(results, class_names, instance_counts)

        # Handle fitness value
        fitness = results.fitness if hasattr(results, 'fitness') else None
        if isinstance(fitness, np.float64):
            fitness = float(fitness)

        # Create JSON summary object
        json_summary = {
            "image_path": image_path,
            "label_path": label_path,
            "best_model_path": best_model_path,
            "confidence_threshold": confidence_threshold,
            "img_size": img_size,
            "device": selected_device,
            "model_name": model_name,
            "class_names": class_names,
            "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
            "metrics": {
                "precision": [safe_float(p) for p in box_metrics.p.tolist()] if hasattr(box_metrics, 'p') else [None] * len(class_names),
                "recall": [safe_float(r) for r in box_metrics.r.tolist()] if hasattr(box_metrics, 'r') else [None] * len(class_names),
                "mAP50": safe_float(box_metrics.map50),
                "mAP50-95": safe_float(box_metrics.map),
                "fitness": fitness,
                "instance_counts": instance_counts
            },
            "speed": {
                "preprocess": safe_float(results.speed['preprocess']) if 'preprocess' in results.speed else None,
                "inference": safe_float(results.speed['inference']) if 'inference' in results.speed else None,
                "postprocess": safe_float(results.speed['postprocess']) if 'postprocess' in results.speed else None
            },
            "flexible_summary": flexible_summary
        }

        # Save JSON summary to file
        json_summary_path = os.path.join(output_dir, f'validation_summary_{os.path.basename(image_path)}.json')
        with open(json_summary_path, 'w') as json_file:
            json.dump(json_summary, json_file, indent=4)

        logger.info(f"Summary saved in: {json_summary_path}")
        logger.info("Flexible Summary:\n" + flexible_summary)

        return json_summary

    except Exception as e:
        logger.error(f"Error during validation for image {image_path}: {str(e)}")
        return None



def run_validation_on_dataset(images_dir, labels_dir, devices, best_model_path, confidence_threshold, img_size, project_root_dir, model_name, yaml_path, batch_size, training_image_size, num_runs=1,account_type=None):
    """
    Run validation on a dataset multiple times with YOLOv8, log each run to ClearML, and measure processing time per image.
    
    Parameters:
        images_dir (str): Path to the directory containing validation images.
        labels_dir (str): Path to the directory containing labels for validation images.
        devices (list or str): List of devices to run the validation on (e.g., ['cuda:0'] or 'cpu').
        best_model_path (str): Path to the trained model file (e.g., 'best.pt') for validation.
        confidence_threshold (float): Confidence threshold for YOLOv8 inference.
        img_size (int): Input image size for YOLOv8.
        project_root_dir (str): Root directory for the project, used for organizing output folders.
        model_name (str): Model version name, used for output naming (e.g., 'n', 's', 'm', 'l', 'x').
        yaml_path (str): Path to the data YAML file for YOLOv8.
        num_runs (int): Number of validation runs to perform with random seeds for variability.
        account_type (str, optional): ClearML account type to use for logging.

    This function creates a separate ClearML task for each validation run, logs mAP metrics, mean precision, mean recall, and calculates 
    the average processing time per image to provide a performance benchmark. Results are saved to JSON for analysis.
    """
    
   # Get the directory of the best_model_path
    weights_folder = os.path.dirname(best_model_path)  # This points to the weights folder


    # Navigate to the parent directory where args.yaml is located
    args_folder = os.path.dirname(weights_folder)  # Move up to the run directory


    # Construct the path to args.yaml
    args_yaml_path = os.path.join(args_folder, 'args.yaml')

    

    def get_seed_from_args_yaml(args_yaml_path):

        with open(args_yaml_path, 'r') as file:

            args = yaml.safe_load(file)

            return args.get('seed', None)  # Replace 'seed' with the actual key if different


    def warmup_model(model, device, img_size, num_warmup=10):
        """
            Primera ejecución de un modelo en GPU implica:
            - Compilación JIT (Just-In-Time) de kernels CUDA
            - Inicialización de buffers
            - Asignación de memoria
            - Optimizaciones iniciales
            Esto hace que las primeras inferencias sean más lentas
        """
        model.to(device)
        # Crear una imagen sintética con valores en el rango [0,255] y luego normalizar
        dummy_input = torch.ones(1, 3, img_size, img_size).to(device) * 127.5  # Valor medio de imagen
        dummy_input = dummy_input / 255.0  # Normalizar a [0,1]
        
        with torch.no_grad():
            for i in range(num_warmup):
                _ = model(dummy_input)
                if i == 0:  # Solo log para la primera iteración
                    #logging.info(f"Warm-up iteration {i+1}/{num_warmup} completed")
                    pass
        
        torch.cuda.synchronize()
        logging.info("Warm-up completed successfully")

    def clear_gpu_cache():
        """
        Necesaria porque:
        - La GPU mantiene datos en caché
        - Puede afectar mediciones posteriores
        - Asegura condiciones similares entre runs
        """
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(
        project_root_dir,
        'valid',
        'valid_dataset',
        f'img_size_{img_size}',
        f'yolov8_{model_name}',
        f'detect_{timestamp}'
    )
    os.makedirs(output_dir, exist_ok=True)
    print("Images dir",images_dir)
    temp_yaml_path = os.path.join(images_dir, 'temp_data.yaml')
    with open(temp_yaml_path, 'w') as yaml_file:
        yaml.dump({
            'train': '',
            'val': images_dir,
            'nc': 4,
            'names': ['AMBEL', 'LENCU', 'POLAV', 'POLPE']
        }, yaml_file)

    # num_images = len([file for file in os.listdir(images_dir) if file.endswith(('.jpg', '.jpeg', '.png'))])

    logging.info("Starting warm-up runs...")
    for warm_up_run in range(2):
        try:
            clear_gpu_cache()
            model = YOLO(best_model_path)
            selected_device = f'cuda:{devices[0]}' if isinstance(devices, list) else 'cpu'
            warmup_model(model, selected_device, img_size)
            
            # Ejecutar validación sin guardar resultados
            _ = model.val(
                data=temp_yaml_path,
                batch=batch_size,
                imgsz=img_size,
                conf=confidence_threshold,
                device=selected_device,
                verbose=False,  # Reducir output
                project=output_dir,
                name=f'warmup_run_{warm_up_run}',
                task='detect'
            )
            logging.info(f"Warm-up run {warm_up_run + 1}/2 completed")
            
        except Exception as e:
            logging.error(f"Error during warm-up run {warm_up_run + 1}: {str(e)}")
    
    # Lista para almacenar tiempos de todos los runs
    all_run_times = []

    # Get fixed seed from model
    run_seed = get_seed_from_args_yaml(args_yaml_path)

    for run in range(1, num_runs + 1):
        try:
            # Limpiar caché antes de cada run
            clear_gpu_cache()
            
            task = start_clearml_task(
                name=f"Yolov8_{model_name}_validation_{img_size}x{img_size}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_run_{run}",
                description="Running YOLOv8 validation on dataset",
                project_name="Yolov8 Research Valid (WarmUP)"
            )
            task_id = task.id

            model = YOLO(best_model_path)
            selected_device = f'cuda:{devices[0]}' if isinstance(devices, list) else 'cpu'

            # Realizar warm-up antes de la validación
            logging.info(f"Performing warm-up for run {run}")
            warmup_model(model, selected_device, img_size)


            logging.info(f"Starting validation run {run}/{num_runs} with seed {run_seed}")

            # Se habilito la opción de seleccionar la cantidad de detenciones por images, sin embargo, la images de entrenamiento no se encuentran altamente pobladas
            # Por lo tanto, no afecta las métricas.
            max_det=300 

            results = model.val(
                data=temp_yaml_path,
                batch=batch_size,
                imgsz=img_size,
                conf=confidence_threshold,
                device=selected_device,
                save_txt=True,
                save_conf=True,
                verbose=True,
                max_det=max_det,
                project=output_dir,
                save_json=True,
                name=f'run_{run}',
                task='detect'
            )

            # Cálculo de tiempos con sincronización CUDA
            torch.cuda.synchronize()
            inference_time = results.speed['inference']
            postprocess_time = results.speed['postprocess']
            mean_time_per_image = inference_time + postprocess_time
            all_run_times.append(mean_time_per_image)

            metrics = results.box
            mAP50 = float(metrics.map50)
            mAP50_95 = float(metrics.map)
            precisions = [float(p) for p in metrics.p]
            recalls = [float(r) for r in metrics.r]
            mean_precision = sum(precisions) / len(precisions) if precisions else 0
            mean_recall = sum(recalls) / len(recalls) if recalls else 0
            class_names = list(results.names.values())

            # Calcular estadísticas directamente sin necesidad de excluir runs
            current_mean_time = sum(all_run_times) / len(all_run_times)
            current_std_time = np.std(all_run_times) if len(all_run_times) > 1 else 0

            # Logging de métricas incluyendo estadísticas de tiempo
            task.get_logger().report_scalar("mAP50", "value", mAP50, iteration=run)
            task.get_logger().report_scalar("mAP50-95", "value", mAP50_95, iteration=run)
            task.get_logger().report_scalar("Mean Processing Time per Image (ms)", "value", mean_time_per_image, iteration=run)
            task.get_logger().report_scalar("Mean Processing Time Across Runs (ms)", "value", current_mean_time, iteration=run)
            task.get_logger().report_scalar("Std Dev Processing Time (ms)", "value", current_std_time, iteration=run)

            json_summary = {
                "yaml_path": yaml_path,
                "best_model_path": best_model_path,
                "confidence_threshold": confidence_threshold,
                "img_size": img_size,
                "training_image_size": training_image_size, 
                "device": selected_device,
                "model_name": model_name,
                "class_names": class_names,
                "timestamp": timestamp,
                "run": run,
                "seed": run_seed,
                "batch": batch_size,
                "clearml_task_id": task_id,
                "max_det":max_det,
                "metrics": {
                    "precision": precisions,
                    "recall": recalls,
                    "mean_precision": mean_precision,
                    "mean_recall": mean_recall,
                    "mAP50": mAP50,
                    "mAP50-95": mAP50_95,
                    "mean_time_per_image_ms": mean_time_per_image,
                    "mean_time_across_runs_ms": current_mean_time,
                    "std_dev_time_ms": current_std_time,
                    "pre_measurement_warmup_runs": 2  # Documentar que hubo warm-up

                }
            }

            json_summary_path = os.path.join(output_dir, f'validation_results_summary_run_{run}.json')
            with open(json_summary_path, 'w') as json_file:
                json.dump(json_summary, json_file, indent=4)
            task.upload_artifact(f"validation_results_summary_run_{run}", json_summary_path)

            logging.info(f"Run {run} completed - mAP50: {mAP50}, Time: {mean_time_per_image:.4f}ms (Mean across runs: {current_mean_time:.4f}ms ± {current_std_time:.4f}ms)")

        except Exception as e:
            logging.error(f"Error during validation: {str(e)}")
        finally:
            log_clearml_result(task, f"Validation completed for {num_runs} runs with model {model_name}.")

    create_excel_from_json(output_dir)
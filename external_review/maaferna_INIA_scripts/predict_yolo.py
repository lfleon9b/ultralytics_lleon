from datetime import datetime
import json
import os
import logging
import shutil
import traceback
from PIL import Image  # Import the PIL library
from glob import glob
import glob
import time
from datetime import timedelta

import cv2
from ultralytics import YOLO, solutions
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
from sahi.utils.file import save_json

import pandas as pd
import numpy as np
import torch

from geo_data_utils import generate_batch_summary_csv, generate_geojson
from utils import apply_minimalistic_label_style_pil, apply_minimalistic_label_style_pil_sahi, combine_frame_with_bottom_summary, create_bottom_summary_image 
from utils import draw_styled_boxes_and_summary, generate_json_output, get_image_metadata, generate_summary_by_class, format_time
from utils import generate_summary_by_class_sahi_specific, get_best_model, numpy_to_native, prediction_result_to_dict

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def run_inference(img_size, model_version, test_images_dir, project_root_dir, confidence_threshold, labels_dict, colors, img_size_training_model):
    """
    Run inference on test images using the best trained YOLO model with a confidence threshold.
    Save the predicted labels and images, including metadata and class summaries.

    Args:
    img_size (int): Image size used for inference.
    model_version (str): YOLO model version used for inference ('n', 's', 'm', 'l', 'x').
    test_images_dir (str): Path to the test images directory.
    project_root_dir (str): Root directory of the project where models and training results are stored.
    confidence_threshold (float): Minimum confidence score to consider a detection valid.
    labels_dict (dict): Dictionary of labels and their corresponding class names.
    colors (list): List of RGB color tuples corresponding to labels.

    Returns:
    None
    """
  # Set up logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # Define valid image extensions
    valid_image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif')

    # Create main output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    main_output_dir = os.path.join(project_root_dir, 'outputs', 'predictions', f'img_size_{img_size}', f'yolov8_{model_version}', f'predict_{timestamp}')
    os.makedirs(main_output_dir, exist_ok=True)

    # Create subdirectories
    styled_images_dir = os.path.join(main_output_dir, 'styled-images')
    default_images_dir = os.path.join(main_output_dir, 'default-images')
    labels_dir = os.path.join(main_output_dir, 'labels')
    metadata_dir = os.path.join(main_output_dir, 'JSON_metadata')

    for directory in [styled_images_dir, default_images_dir, labels_dir, metadata_dir]:
        os.makedirs(directory, exist_ok=True)

    # Find the best model
    best_model_path = get_best_model(project_root_dir, img_size, model_version)
    if not best_model_path:
        logging.error("No best model found for prediction.")
        return

    logging.info(f"Best model found: {best_model_path}")

    # Load the trained model
    try:
        model = YOLO(best_model_path)
        logging.info("YOLO model loaded successfully.")
    except Exception as e:
        logging.error(f"Failed to load the YOLO model from {best_model_path}: {str(e)}")
        return

    # Load hyperparameters and training information
    try:
        model_data = torch.load(best_model_path, map_location=torch.device('cpu'))
        model_hyperparams = model_data['model'].yaml
        training_image_size = model_data['train_args']['imgsz']
        logging.info("Model hyperparameters and training information loaded successfully.")
    except Exception as e:
        logging.error(f"Error extracting hyperparameters: {str(e)}")
        model_hyperparams = {}
        training_image_size = 'Unknown'

    # Process each file in the directory
    for filename in os.listdir(test_images_dir):
        file_path = os.path.join(test_images_dir, filename)
        
        # Skip non-image files
        if not filename.lower().endswith(valid_image_extensions):
            logging.info(f"Skipping non-image file: {filename}")
            continue

        logging.info(f"Processing image: {filename}")

        try:
            # Run prediction on single image
            results = model.predict(
                source=file_path,
                imgsz=img_size,
                device='cuda',
                conf=confidence_threshold,
                save=True,
                save_txt=True,
                project=main_output_dir,
                name='temp_prediction'
            )

            # Process the result
            for result in results:
                bboxes = []
                # Extract bounding boxes and label indexes
                for box in result.boxes:
                    x_min, y_min, x_max, y_max = box.xyxy[0].tolist()
                    confidence = box.conf.tolist()[0]
                    label_idx = int(box.cls.tolist()[0])
                    bboxes.append([label_idx, x_min, y_min, x_max, y_max, confidence])

                # Generate summary and get metadata
                label_summary = generate_summary_by_class(bboxes, labels_dict)
                image_metadata = get_image_metadata(file_path)

                # Generate styled image
                base_name = os.path.splitext(filename)[0]
                styled_image_name = f"{base_name}-styled.jpg"
                styled_output_path = os.path.join(styled_images_dir, styled_image_name)
                apply_minimalistic_label_style_pil(file_path, bboxes, list(labels_dict.values()), styled_output_path, colors, label_summary)

                # Move default image
                default_image_path = os.path.join(main_output_dir, 'temp_prediction', f"{filename}")
                if os.path.exists(default_image_path):
                    shutil.move(default_image_path, os.path.join(default_images_dir, filename))

                # Move label file
                label_file_path = os.path.join(main_output_dir, 'temp_prediction', 'labels', f"{base_name}.txt")
                if os.path.exists(label_file_path):
                    shutil.move(label_file_path, os.path.join(labels_dir, f"{base_name}.txt"))

                # Prepare model information
                with Image.open(file_path) as img:
                    img_size = img.size
                model_info = {
                    "model_version": f'yolov8{model_version}.pt',
                    "confidence_threshold": confidence_threshold,
                    "img_size": img_size,
                    "img_size_training_model": img_size_training_model,
                    "hyperparameters": model_hyperparams,
                    "training_image_size": training_image_size,
                    'source_path': test_images_dir,
                    'model_architecture': str(model.model),
                    'device': str(model.device)
                }

                # Generate JSON metadata
                json_file_name = f"{base_name}_image_prediction_metadata.json"
                json_output_path = os.path.join(metadata_dir, json_file_name)
                generate_json_output(image_metadata, bboxes, model_info, label_summary, json_output_path)

                logging.info(f"Processed image: {filename}")

            # Remove temporary prediction folder after processing each image
            temp_prediction_dir = os.path.join(main_output_dir, 'temp_prediction')
            if os.path.exists(temp_prediction_dir):
                shutil.rmtree(temp_prediction_dir)

        except Exception as e:
            logging.error(f"An error occurred during prediction of {filename}: {str(e)}")
            traceback.print_exc()

    logging.info(f"Inference process completed. Results saved in: {main_output_dir}")
    logging.info(f"Styled images: {styled_images_dir}")
    logging.info(f"Default images: {default_images_dir}")
    logging.info(f"Labels: {labels_dir}")
    logging.info(f"JSON metadata: {metadata_dir}")


def run_inference_single_image(single_image_path, model_version, best_model_path, output_dir, confidence_threshold, labels_dict, colors, img_size_training_model):
    """
    Run inference for a single image, and save both the default and styled predictions.
    Additionally, generate a summary of the detected objects by class.

    Args:
    single_image_path (str): Path to the input image.
    model_version (str): The YOLO model version used.
    best_model_path (str): Path to the best-trained YOLO model.
    output_dir (str): Directory to save the prediction results.
    confidence_threshold (float): Confidence threshold for predictions.

    Returns:
    None
    """
    # Load the YOLO model
    model = YOLO(best_model_path)

    # Predict the image
    results = model.predict(
        source=single_image_path,
        conf=confidence_threshold,  # Apply confidence threshold
        save_txt=True,  # Save predicted labels as .txt files
        project=output_dir,  # Output directory
        name='predictions'  # Subfolder name for predictions
    )

    # Extract bounding boxes and labels from results
    bboxes = []  # List to store bounding boxes
    for result in results:
        for box in result.boxes:
            xyxy = box.xyxy[0].cpu().numpy().tolist()  # Convert to Python list (will convert to floats)
            x_min, y_min, x_max, y_max = xyxy  # Unpack the coordinates
            confidence = float(box.conf[0].cpu().numpy())  # Convert to native Python float
            label_idx = int(box.cls[0].cpu().numpy())  # Get class index as integer
            bboxes.append([label_idx, x_min, y_min, x_max, y_max, confidence])

    # Generate a summary by class
    label_summary = generate_summary_by_class(bboxes, labels_dict)

    # Display the summary
    print("Summary of detected objects by class:")
    for label, count in label_summary.items():
        print(f"{label}: {count}")

    # Define the label list
    label_list = [labels_dict[i] for i in sorted(labels_dict.keys())]

    # Ensure the base name is assigned properly
    base_name = os.path.splitext(os.path.basename(single_image_path))[0]

    # Use the most recent predictions folder
    predictions_folders = sorted(glob.glob(os.path.join(output_dir, 'predictions*')), key=os.path.getmtime, reverse=True)
    if not predictions_folders:
        print("No predictions folder found.")
        return
    last_predictions_folder = predictions_folders[0]

    # Generate the styled image and include label summary in the legend
    styled_output_path = os.path.join(last_predictions_folder, f"{base_name}-styled.jpg")
    apply_minimalistic_label_style_pil(single_image_path, bboxes, label_list, styled_output_path, colors, label_summary)

    # Copy the default YOLO prediction image to the correct folder
    default_image_path = os.path.join(output_dir, f"predictions", f"{base_name}.jpg")
    default_output_path = os.path.join(last_predictions_folder, f"{base_name}-default.jpg")
    try:
        shutil.copyfile(default_image_path, default_output_path)
        print(f"Default YOLO prediction image saved to {default_output_path}")
    except Exception as e:
        print(f"Error saving default YOLO prediction image: {str(e)}")

    # Get metadata from the image
    image_metadata = get_image_metadata(single_image_path)

    # Get image dimensions
    image = Image.open(single_image_path)
    img_size = image.size  # Image dimensions (width, height)

    # Load hyperparameters and training information from the best.pt file
    try:
        model_data = torch.load(best_model_path, map_location=torch.device('cpu'))  # Load the model file

        # Extract hyperparameters from the model's dictionary
        model_hyperparams = model_data['model'].yaml  # Access the hyperparameters directly

        # Extract image size and training details from the model's dictionary
        training_image_size = model_data['train_args']['imgsz']  # Access the image size directly

    except Exception as e:
        print(f"Error extracting hyperparameters: {str(e)}")
        model_hyperparams = {}
        training_image_size = 'Unknown'

    # Prepare model information
    model_info = {
        "model_version": f'yolov8{model_version}.pt',
        "confidence_threshold": confidence_threshold,
        "img_size": img_size,  # Use the actual image size
        "img_size_training_model": img_size_training_model,
        "hyperparameters": model_hyperparams,
        "training_image_size": training_image_size,
        'source_path': single_image_path,
        'model_architecture': str(model.model),
        'device': str(model.device)
    }

    # Generate the JSON metadata directory inside the most recent predictions folder
    metadata_dir = os.path.join(last_predictions_folder, 'JSON_metadata')
    os.makedirs(metadata_dir, exist_ok=True)

    # Create a unique JSON filename based on the image name to avoid overwriting
    json_file_name = f"{base_name}_image_prediction_metadata.json"
    json_output_path = os.path.join(metadata_dir, json_file_name)

    # Generate the JSON file with label summary, hyperparameters, and training image size
    generate_json_output(image_metadata, bboxes, model_info, label_summary, json_output_path)

    print(f"JSON metadata saved to: {json_output_path}")


def run_inference_video(video_path, model_version, model_path, output_dir, confidence_threshold, labels_dict, colors):
    """
    Run YOLOv8 inference on a video file and apply styled bounding boxes, and place label summary at the bottom.
    Also generate a subtitle (.srt) file with object counts per frame.

    Args:
    video_path (str): Path to the video file.
    model_version (str): YOLO model version.
    model_path (str): Path to the trained YOLO model.
    output_dir (str): Directory to save the results.
    confidence_threshold (float): Confidence threshold for predictions.
    labels_dict (dict): Dictionary mapping class indices to class names.
    colors (list): List of RGB color tuples corresponding to labels.

    Returns: None
    """

    # Load the YOLO model
    model = YOLO(model_path)

    # Open the video file
    video = cv2.VideoCapture(video_path)

    # Retrieve video properties
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(video.get(cv2.CAP_PROP_FPS))
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

    # Output paths for customized and default videos
    output_video_path = os.path.join(output_dir, f"customized_{os.path.basename(video_path)}")
    default_video_path = os.path.join(output_dir, f"default_{os.path.basename(video_path)}")
    srt_output_path = os.path.join(output_dir, f"{os.path.splitext(os.path.basename(video_path))[0]}.srt")

    # VideoWriter for output videos
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height + 50))  # Extra space for bottom summary
    out_default = cv2.VideoWriter(default_video_path, fourcc, fps, (width, height))

    # Setup ObjectCounter
    region_points = [(0, height), (width, height), (width, 0), (0, 0)]
    counter = solutions.ObjectCounter(view_img=False, reg_pts=region_points, names=labels_dict, draw_tracks=True, line_thickness=2)

    frame_count = 0
    error_count = 0
    label_counts_overall = {key: 0 for key in labels_dict.keys()}

    # Open .srt file for writing
    with open(srt_output_path, 'w') as srt_file:
        while video.isOpened():
            ret, frame = video.read()
            if not ret:
                print("Video frame is empty or video processing has been successfully completed.")
                break

            try:
                # Run YOLO inference
                results = model.track(frame, persist=True, conf=confidence_threshold, device='cuda')

                frame_count += 1
                if frame_count % 100 == 0:
                    print(f"Processed {frame_count} frames")

                for result in results:
                    try:
                        # Count objects in the frame
                        frame = counter.start_counting(frame, result)

                        # Extract bounding boxes and label counts
                        styled_frame, label_counts = draw_styled_boxes_and_summary(frame, [result], labels_dict, colors)

                        # Update overall label counts
                        for label_idx, count in label_counts.items():
                            label_counts_overall[label_idx] += count

                        # Write to .srt file (generate timestamp and object count data)
                        start_time = frame_count / fps
                        end_time = (frame_count + 1) / fps

                        timestamp = f"{format_time(start_time)} --> {format_time(end_time)}"
                        object_summary = ', '.join([f"{labels_dict[label_idx]}: {count}" for label_idx, count in label_counts.items()])

                        srt_file.write(f"{frame_count}\n")
                        srt_file.write(f"{timestamp}\n")
                        srt_file.write(f"{object_summary}\n\n")

                        # Create bottom summary for the current frame
                        summary_image = create_bottom_summary_image(label_counts, labels_dict, width, colors)

                        # Combine the styled frame and the summary at the bottom
                        combined_frame = combine_frame_with_bottom_summary(Image.fromarray(styled_frame), summary_image)
                        out.write(combined_frame)

                        # Save the default frame (without styling)
                        out_default.write(frame)

                    except Exception as e:
                        print(f"Error processing result for frame {frame_count}: {str(e)}")
                        continue

            except Exception as e:
                error_count += 1
                print(f"An error occurred while processing frame {frame_count}: {str(e)}")
                if error_count > 50:
                    print("Too many errors encountered. Stopping processing.")
                    break

    # Release resources
    video.release()
    out.release()
    print(f"Processed video saved to: {output_video_path}")
    print(f"Total frames processed: {frame_count}")
    print(f"Total errors encountered: {error_count}")
    print(f"Final label counts: {label_counts_overall}")
    print(f"SRT file generated: {srt_output_path}")


def run_inference_sahi_single_image(image_path, device, slice_size, overlap_ratio, best_model_path, confidence_threshold, img_size, batch_output_dir, model_version, labels_dict, colors, generate_slices, images_dir, batch_mode=False):
    """
    Run SAHI (Slicing Aided Hyper Inference) on a single image using a YOLOv8 model with image slicing.
    Generate predictions and save both the full-image predictions and, if requested, individual sliced-region predictions.

    Args:
    image_path (str): Path to the input image.
    device (str): Device to run inference on ('cpu' or 'cuda').
    slice_size (int): The height and width of each image slice used for inference.
    overlap_ratio (float): Overlap ratio between adjacent slices for better prediction accuracy.
    best_model_path (str): Path to the best-trained YOLOv8 model file.
    confidence_threshold (float): Confidence threshold for predictions.
    img_size (int): Image size used for slicing and inference.
    batch_output_dir (str): The parent directory for the entire batch of predictions (common timestamp folder).
    model_version (str): YOLOv8 model version used for inference (e.g., 'n', 's', 'm', 'l', 'x').
    labels_dict (dict): Dictionary mapping class indices to class names.
    colors (list): List of RGB color tuples corresponding to labels for visual styling.
    generate_slices (bool): Whether or not to generate individual sliced-region predictions.
    images_dir (str): The directory containing the batch of images.
    batch_mode (bool): Whether the inference is run in batch mode or single image mode.

    Returns:
    dict: A summary of the inference results for CSV generation.
    """
    try:
        # Start time tracking for the image
        start_time = time.perf_counter()
        
        # Define image name and output paths based on the new structure
        image_name = os.path.splitext(os.path.basename(image_path))[0]

        if batch_mode:
            # Batch mode: use the batch_output_dir passed in (already created for all images)
            output_dir = os.path.join(batch_output_dir, image_name)
        else:
            # Single image mode: create the specific directory structure for this single image
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = os.path.join(
                batch_output_dir,  # this is the base dir for single image
                f'{image_name}',
                f'predict_{timestamp}'
            )

        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Image results will be saved to: {output_dir}")

        # Create subdirectories for styled images and metadata
        styled_images_dir = os.path.join(output_dir, 'styled-images')
        metadata_dir = os.path.join(output_dir, 'JSON_metadata')
        os.makedirs(styled_images_dir, exist_ok=True)
        os.makedirs(metadata_dir, exist_ok=True)

        # Load YOLOv8 model
        model = AutoDetectionModel.from_pretrained(
            model_type='yolov8',
            model_path=best_model_path,
            confidence_threshold=confidence_threshold,
            device=device
        )

        # Run SAHI inference (with slicing)
        result = get_sliced_prediction(
            image_path,
            model,
            slice_height=slice_size,
            slice_width=slice_size,
            overlap_height_ratio=overlap_ratio,
            overlap_width_ratio=overlap_ratio
        )

        # Save full image prediction visualization
        result.export_visuals(export_dir=output_dir, file_name="full_image_prediction")
        logger.info(f"Saved full image prediction visualization")

        # Convert the prediction result to a dictionary and save as JSON
        result_dict = prediction_result_to_dict(result, image_path)
        json_file_name = f"{image_name}_image_metadata.json"
        json_output_path = os.path.join(metadata_dir, json_file_name)
        with open(json_output_path, 'w') as f:
            json.dump(result_dict, f, indent=4)
        logger.info(f"Saved prediction results as JSON at {json_output_path}")

        full_image = Image.open(image_path)
        bboxes = []  # List to store all bounding boxes for the full image

        # Process bounding boxes for all slices but skip file generation if `generate_slices` is False
        for idx, pred in enumerate(result.object_prediction_list):
            bbox = pred.bbox.to_xyxy()
            x_min, y_min, x_max, y_max = [numpy_to_native(coord) for coord in bbox]
            confidence = numpy_to_native(pred.score.value)
            class_name = pred.category.name
            label_idx = list(labels_dict.values()).index(class_name)

            # Append to bboxes list for full image processing
            bboxes.append([label_idx, x_min, y_min, x_max, y_max, confidence])

            if generate_slices:
                slices_dir = os.path.join(output_dir, 'slices')
                os.makedirs(slices_dir, exist_ok=True)
                region_image = full_image.crop((x_min, y_min, x_max, y_max))
                region_image_path = os.path.join(slices_dir, f"{image_name}_region_{idx}.jpg")
                region_image.save(region_image_path)
                logger.info(f"Saved region image: {region_image_path}")

                region_prediction = {
                    "bbox": [x_min, y_min, x_max, y_max],
                    "category": class_name,
                    "score": confidence
                }
                try:
                    region_prediction_path = os.path.join(slices_dir, f"{image_name}_region_{idx}_prediction.json")
                    with open(region_prediction_path, 'w') as f:
                        json.dump(region_prediction, f, indent=4)
                    logger.info(f"Saved prediction for region {idx}")
                except Exception as e:
                    logger.error(f"Error saving slice prediction JSON: {str(e)}")
                    continue

        # Generate styled image and save
        styled_image_name = f"{image_name}-styled.jpg"
        styled_output_path = os.path.join(styled_images_dir, styled_image_name)
        label_summary = generate_summary_by_class_sahi_specific(result.object_prediction_list, labels_dict)
        apply_minimalistic_label_style_pil_sahi(image_path, result.object_prediction_list, list(labels_dict.values()), styled_output_path, colors, label_summary)
        logger.info(f"Saved styled image: {styled_output_path}")

                    # Load model metadata and hyperparameters
        image_metadata = get_image_metadata(image_path)
        model_data = torch.load(best_model_path, map_location=torch.device('cpu'))
        model_hyperparams = model_data['model'].yaml
        training_image_size = model_data['train_args']['imgsz']


        model_info = {
            "model_version": f'yolov8{model_version}.pt',
            "confidence_threshold": confidence_threshold,
            "img_size": full_image.size,
            "img_size_training_model": img_size,
            "hyperparameters": model_hyperparams,
            "training_image_size": training_image_size,
            'source_path': os.path.dirname(image_path),
            'device': device
        }


        # Save metadata and generate the output JSON
        generate_json_output(image_metadata, bboxes, model_info, label_summary, json_output_path)

        with open(json_output_path, 'r') as f:
            json_data = json.load(f)

       # Check if UTM_Coordinates are in the root of the JSON
        if 'UTM_Coordinates' not in json_data:
            print(f"Debug: UTM Coordinates are missing for {image_name}")
            print(f"Debug: Full JSON data for {image_name}: {json.dumps(json_data, indent=2)}")
            utm_easting = ''
            utm_northing = ''
            altitude = ''
        else:
            # Extract UTM Coordinates from the root of the JSON
            utm_coordinates = json_data['UTM_Coordinates']
            utm_easting = utm_coordinates.get('UTM_Easting', '')
            utm_northing = utm_coordinates.get('UTM_Northing', '')
            altitude = utm_coordinates.get('Altitude', '')

        # Stop time tracking after inference is completed
        end_time = time.perf_counter()

        # Calculate elapsed time
        elapsed_time = end_time - start_time

        # Format the elapsed time to show seconds with milliseconds
        formatted_time = "{:02}:{:02}:{:06.3f}".format(int(elapsed_time // 3600), int((elapsed_time % 3600) // 60), elapsed_time % 60)

        results = {
            'image_name': image_name,
            'utm_easting': utm_easting,
            'utm_northing': utm_northing,
            'altitude': altitude,
            'image_path': image_path,
            'styled_image_path': styled_output_path,
            'label_summary': label_summary,
            'processing_time': formatted_time  # Track the time taken for this image
        }

        logger.info(f"Saved JSON metadata: {json_output_path}")

        # Generate GeoJSON file
        generate_geojson(image_name, json_output_path, output_dir, styled_output_path)

        logger.info(f"SAHI inference completed on image: {image_path}")
        logger.info(f"Results saved in: {output_dir}")
        logger.info(f"Processing time for {image_name}: {formatted_time}")

        return results

    except Exception as e:
        logger.error(f"An error occurred during inference: {str(e)}", exc_info=True)
        raise



def run_inference_sahi_directory(device, slice_size, overlap_ratio, best_model_path, confidence_threshold, img_size, images_dir, project_root_dir, model_version, labels_dict, colors, generate_slices):
    """
    Run SAHI inference on a directory of images using a YOLOv8 model with slicing and save results for each image.
    """
    logging.info(f"Running SAHI inference on directory: {images_dir}")

    if not best_model_path:
        logging.error(f"No trained model found for img_size={img_size} and model_version={model_version}")
        return

    logging.info(f"YOLOv8 model loaded successfully for SAHI inference: {best_model_path}")

    valid_image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif')

    # Create the batch output directory (shared for all images in this run)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    batch_output_dir = os.path.join(
        project_root_dir, 'outputs', 'SAHI', 'Batch_images', 
        f'img_size_{img_size}', f'yolov8_{model_version}', os.path.basename(images_dir), f'predict_{timestamp}'
    )
    os.makedirs(batch_output_dir, exist_ok=True)
    logging.info(f"Batch output will be saved to: {batch_output_dir}")

    all_results = []
    start_time = time.time()  # Start time for tracking the batch processing
    image_count = 0

    for filename in os.listdir(images_dir):
        file_path = os.path.join(images_dir, filename)

        if not filename.lower().endswith(valid_image_extensions):
            logging.info(f"Skipping non-image file: {filename}")
            continue

        logging.info(f"Processing image: {filename}")
        image_count += 1

        try:
            # Pass batch_mode=True to simplify the output path
            result = run_inference_sahi_single_image(
                image_path=file_path,
                device=device,
                slice_size=slice_size,
                overlap_ratio=overlap_ratio,
                best_model_path=best_model_path,
                confidence_threshold=confidence_threshold,
                img_size=img_size,
                batch_output_dir=batch_output_dir,  # Use the shared batch output dir
                model_version=model_version,
                labels_dict=labels_dict,
                colors=colors,
                generate_slices=generate_slices,
                images_dir=images_dir,
                batch_mode=True
            )
            all_results.append(result)

        except Exception as e:
            logging.error(f"An error occurred during SAHI inference on {filename}: {str(e)}")

    # After processing all images, calculate total time and generate a single batch summary CSV
    total_time = time.time() - start_time  # Calculate the total time for batch processing
    time_taken = str(timedelta(seconds=int(total_time)))  # Format the time as HH:MM:SS

    # Save the batch processing time and image count in a summary JSON
    batch_summary = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {
                    "model_version": f"yolov8_{model_version}.pt" if model_version else "Unknown",
                    "best_model_path": best_model_path if best_model_path else "N/A",
                    "confidence_threshold": confidence_threshold if confidence_threshold else 0.0,
                    "img_size": img_size if img_size else "N/A",
                    "slice_size": slice_size if slice_size else "N/A",
                    "source_path": images_dir if images_dir else "Unknown",
                    "quantities_images": image_count if image_count else 0,
                    "processing_time": time_taken if time_taken else "00:00:00",
                    "predict-summary": all_results if all_results else [],
                    "device": device if device else "cpu",
                    "overlap_ratio": overlap_ratio if overlap_ratio else 0.0
                }
            }
        ]
    }


    batch_summary_path = os.path.join(batch_output_dir, 'batch_summary.json')
    with open(batch_summary_path, 'w') as f:
        json.dump(batch_summary, f, indent=4)

    logging.info(f"Batch summary JSON generated at: {batch_summary_path}")

    generate_batch_summary_csv(batch_output_dir, os.path.join(batch_output_dir, 'batch_summary.csv'), all_results)
    logging.info(f"Batch summary CSV generated at: {os.path.join(batch_output_dir, 'batch_summary.csv')}")



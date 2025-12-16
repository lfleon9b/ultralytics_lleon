from collections import Counter
import json
import re
import cv2
import os
import shutil
import logging
from collections import Counter
import base64

from fractions import Fraction
from PIL.TiffImagePlugin import IFDRational
from PIL import ImageFont, ImageDraw, Image, ExifTags
import numpy as np
import matplotlib.pyplot as plt
import torch
import urllib
import yaml
from pyproj import Proj
import pandas as pd

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


def download_yolo_model(yolo_version, model_version):
    """
    Download the YOLO model file if it doesn't exist.

    Args:
    yolo_version (str): The YOLO version ('8', '11', etc.)
    model_version (str): The model size ('n', 's', 'm', 'l', 'x')

    Returns:
    str: Path to the downloaded model file
    """
    model_filename = f'yolo{yolo_version}{model_version}.pt'
    models_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models'))
    os.makedirs(models_dir, exist_ok=True)  # Ensure the models directory exists
    model_path = os.path.join(models_dir, model_filename)

    if not os.path.exists(model_path):
        logging.info(f"Model file {model_filename} not found. Downloading...")
        try:
            # Replace the URL below with the actual download link for the YOLO model
            download_url = f'https://github.com/ultralytics/assets/releases/download/v8.3.0/{model_filename}'
            urllib.request.urlretrieve(download_url, model_path)
            logging.info(f"Model file {model_filename} downloaded and saved to {model_path}")
        except Exception as e:
            logging.error(f"Failed to download the model file: {e}")
            raise FileNotFoundError(f"Could not download the model file: {model_filename}")
    else:
        logging.info(f"Model file {model_filename} already exists at {model_path}")

    return model_path

def get_default_test_images_dir(img_size, test_images_root_dir):
    return os.path.join(
        test_images_root_dir,
        f"Lencu-merge2to11-except5_{img_size}x{img_size}",
        "test",
        "images"
    )

def convert_path_to_windows_format(path):
    """
    Converts a given path to the appropriate Windows format.

    Args:
    path (str): The path to be converted.

    Returns:
    str: The converted path in Windows format.
    """
    if os.name == 'nt':  # Check if the OS is Windows
        if path.startswith('/'):
            # Convert from Unix-like /e/ to Windows E:\
            path = path[1].upper() + ':' + path[2:]  # Convert `/e/` to `E:`
        path = path.replace('/', '\\')  # Convert forward slashes to backslashes
    return path


def apply_minimalistic_label_style_pil(image_path, bboxes, labels, output_path, colors, label_summary):
    """
    Apply minimalistic labels showing only confidence levels on the image with enhanced styling,
    and include a summary of detected objects by class in the legend (in a single row).

    Args:
    image_path (str): Path to the image to modify.
    bboxes (list): List of bounding boxes in format [label_index, x_min, y_min, x_max, y_max, confidence].
    labels (list): List of class names corresponding to label indexes.
    output_path (str): Path to save the modified image.
    colors (list): List of RGB color tuples corresponding to labels.
    label_summary (dict): Summary of detected objects by class.

    Returns: None
    """
    image = Image.open(image_path)
    image_width, image_height = image.size

    # Calculate dynamic font size and border width based on image dimensions
    base_font_size = 12
    base_border_width = 1
    scaling_factor = (image_width + image_height) / 1500  # Adjusted scaling factor for box width

    font_size = int(base_font_size * scaling_factor)
    border_width = max(1, int(base_border_width * scaling_factor))

    # Load a font with calculated size
    try:
        font = ImageFont.truetype("arial.ttf", font_size)
    except IOError:
        font = ImageFont.load_default()

    # Initialize draw object
    draw = ImageDraw.Draw(image, 'RGBA')

    # Draw bounding boxes with only confidence
    for bbox in bboxes:
        label_idx, x_min, y_min, x_max, y_max, confidence = bbox
        color = colors[label_idx % len(colors)]  # Use the colors list provided as an argument

        # Draw rectangle with calculated width
        draw.rectangle([x_min, y_min, x_max, y_max], outline=color + (255,), width=border_width)  # + (255,) adds opacity

        # Display confidence only
        confidence_text = f"{confidence:.2f}"
        
        # Draw transparent background for the label
        text_bbox = draw.textbbox((0, 0), confidence_text, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]
        
        # Adjust box height to fit confidence text
        box_padding = 4  # Padding between text and box edges
        text_background_x1 = max(0, x_min)
        text_background_y1 = max(0, y_min - text_height - 2 * box_padding)
        text_background_x2 = text_background_x1 + text_width + 2 * box_padding
        text_background_y2 = text_background_y1 + text_height + 2 * box_padding

        # Draw transparent rectangle for confidence background
        draw.rectangle([text_background_x1, text_background_y1, text_background_x2, text_background_y2], fill=color + (100,))  # Semi-transparent background

        # Draw confidence text
        text_x = text_background_x1 + box_padding
        text_y = text_background_y1 + box_padding
        draw.text((text_x, text_y), confidence_text, fill=(255, 255, 255, 255), font=font)  # White text

    # Create a new image with extra space for the legend
    legend_height = font_size + 20  # Adjusted for one-row legend
    legend_image = Image.new('RGB', (image_width, image_height + legend_height), (255, 255, 255))
    legend_image.paste(image, (0, 0))

    # Draw the legend at the bottom of the image in a single row
    draw_legend = ImageDraw.Draw(legend_image)
    legend_start_x = 10
    legend_start_y = image_height + 10
    separator = " | "  # Define the separator between items

    # Concatenate the summary into a single row
    for idx, label in enumerate(labels):
        color = colors[idx % len(colors)]
        count = label_summary.get(label, 0)  # Get the count from the summary

        # Draw the label and count in the legend
        draw_legend.rectangle([legend_start_x, legend_start_y, legend_start_x + 20, legend_start_y + 20], fill=color)
        draw_legend.text((legend_start_x + 30, legend_start_y), f"{label}: {count}", fill=color, font=font)
        
        # Move the start position to the right for the next label and add the separator
        legend_start_x += font_size * len(f"{label}: {count}") + 50  # Adjust spacing based on text length
        draw_legend.text((legend_start_x, legend_start_y), separator, fill=(0, 0, 0), font=font)  # Draw the separator
        legend_start_x += font_size * len(separator)  # Adjust for the separator width

    # Save the modified image with legend
    legend_image.save(output_path)
    print(f"Styled image with confidence and legend saved to {output_path}")


def get_project_root():
    """
    Get the root directory of the project.
    """
    current_dir = os.path.dirname(__file__)  # Directory of the current script
    project_root_dir = os.path.abspath(os.path.join(current_dir, '..'))  # Two levels up to get the correct root
    return project_root_dir


def select_model_version():
    """
    Prompt the user to select a valid model version.

    Returns:
    str: The selected model version (n, s, m, l, x).
    """
    valid_versions = ['n', 's', 'm', 'l', 'x']
    while True:
        model_version = input(f"Enter the model version {valid_versions}: ").strip()
        if model_version in valid_versions:
            return model_version
        print(f"Invalid model version. Please enter one of the following: {', '.join(valid_versions)}.")


def draw_styled_boxes_and_summary(frame, results, labels_dict, colors):
    """
    Draw styled bounding boxes with confidence levels on the frame and count detected objects.

    Args:
    frame (numpy.ndarray): Frame from the video.
    results (list): List of detection results.
    labels_dict (dict): Dictionary mapping class indices to class names.
    colors (list): List of RGB color tuples corresponding to labels.

    Returns:
    tuple: Tuple containing the modified frame (PIL.Image) and a dictionary with label counts.
    """
    # Convert the frame from BGR to RGB for PIL processing
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(image, 'RGBA')

    base_font_size = 12
    scaling_factor = (image.width + image.height) / 1500
    font_size = int(base_font_size * scaling_factor)
    border_width = max(1, int(1 * scaling_factor))

    try:
        font = ImageFont.truetype("arial.ttf", font_size)
    except IOError:
        font = ImageFont.load_default()

    label_counts = {key: 0 for key in labels_dict.keys()}

    # Iterate through all results and draw the bounding boxes
    for result in results:
        # Get boxes attribute from result object
        boxes = getattr(result, 'boxes', [result]) if hasattr(result, 'boxes') else [result]

        for box in boxes:
            bbox_info = safe_extract_bbox_info(box)
            if bbox_info is None:
                continue

            x_min, y_min, x_max, y_max, confidence, label_idx = bbox_info

            # Ensure label_idx is in the dictionary
            if label_idx not in labels_dict:
                print(f"Label index {label_idx} not found in labels_dict. Skipping.")
                continue

            # Retrieve label name and corresponding color
            label_name = labels_dict[label_idx]
            color = colors[label_idx % len(colors)]
            label_counts[label_idx] += 1

            # Debug: Print the color being applied for the label
            #print(f"Drawing box for label: {label_name}, color: {color}")

            # Draw rectangle with calculated width
            draw.rectangle([x_min, y_min, x_max, y_max], outline=color + (255,), width=border_width)

            # Display confidence only
            confidence_text = f"{confidence:.2f}"
            text_bbox = draw.textbbox((0, 0), confidence_text, font=font)
            text_width = text_bbox[2] - text_bbox[0]
            text_height = text_bbox[3] - text_bbox[1]
            box_padding = 4
            text_background_x1 = max(0, x_min)
            text_background_y1 = max(0, y_min - text_height - 2 * box_padding)
            text_background_x2 = text_background_x1 + text_width + 2 * box_padding
            text_background_y2 = text_background_y1 + text_height + 2 * box_padding

            # Draw semi-transparent rectangle for confidence background
            draw.rectangle([text_background_x1, text_background_y1, text_background_x2, text_background_y2], fill=color + (100,))

            # Draw confidence text
            text_x = text_background_x1 + box_padding
            text_y = text_background_y1 + box_padding
            draw.text((text_x, text_y), confidence_text, fill=(255, 255, 255, 255), font=font)

    # Convert the image back to BGR format for OpenCV
    styled_frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    # Return the styled frame and the count of detected labels
    return styled_frame, label_counts


def safe_extract_bbox_info(box):
    try:
        # Extract bounding box coordinates
        if hasattr(box, 'xyxy'):
            if hasattr(box.xyxy, 'cpu'):
                xyxy = box.xyxy.cpu().numpy()[0]
            elif isinstance(box.xyxy, (list, tuple)):
                xyxy = box.xyxy[0]
            else:
                xyxy = box.xyxy
        elif hasattr(box, 'xywh'):
            if hasattr(box.xywh, 'cpu'):
                xywh = box.xywh.cpu().numpy()[0]
            elif isinstance(box.xywh, (list, tuple)):
                xywh = box.xywh[0]
            else:
                xywh = box.xywh
            # Convert xywh to xyxy
            xyxy = [xywh[0] - xywh[2] / 2, xywh[1] - xywh[3] / 2, xywh[0] + xywh[2] / 2, xywh[1] + xywh[3] / 2]
        else:
            raise AttributeError("Box object has neither 'xyxy' nor 'xywh' attribute")

        # Ensure all coordinates are integers
        x_min, y_min, x_max, y_max = map(lambda x: int(round(float(x))), xyxy)

        # Extract confidence
        confidence = float(getattr(box, 'conf', 1.0))  # Default confidence is 1.0 if not provided

        # Extract class index
        label_idx = int(getattr(box, 'cls', 0))  # Default class index is 0 if not provided

        return x_min, y_min, x_max, y_max, confidence, label_idx
    except Exception as e:
        print(f"Error extracting bbox info: {str(e)}")
        return None


def create_bottom_summary_image(label_counts, labels, image_width, colors):
    """
    Create a summary image showing the count of recognized elements at the bottom of the main image.

    Args:
    label_counts (dict): Dictionary with label indexes as keys and counts as values.
    labels (list): List of class names corresponding to label indexes.
    image_width (int): Width of the original image.
    colors (list): List of RGB color tuples corresponding to labels.

    Returns:
    PIL.Image: The summary image to be appended at the bottom.
    """
    # Calculate the height of the summary row and create a new image with this height
    row_height = 50
    summary_image = Image.new('RGB', (image_width, row_height), (255, 255, 255))
    draw = ImageDraw.Draw(summary_image)

    # Set up font for the labels
    base_font_size = 16
    try:
        font = ImageFont.truetype("arial.ttf", base_font_size)
    except IOError:
        font = ImageFont.load_default()

    # Define starting position for the text
    x_offset = 10
    padding = 20  # Space between labels

    # Draw the labels and their counts in a single row
    for label_idx, count in label_counts.items():
        label_text = f"{labels[label_idx]}: {count}"

        # Use the corresponding color from the colors list
        color = colors[label_idx % len(colors)]  # Cycle through available colors if needed

        # Convert color to PIL-compatible format
        color_pil = tuple(color)

        # Draw label and count with its corresponding color
        draw.text((x_offset, 10), label_text, fill=color_pil, font=font)

        # Calculate text size and adjust x_offset for the next label
        text_bbox = font.getbbox(label_text)
        text_width = text_bbox[2] - text_bbox[0]
        x_offset += text_width + padding  # Adjust x_offset for the next label

    return summary_image


def combine_frame_with_bottom_summary(default_image, summary_image):
    """
    Combine the default image with the bottom summary.

    Args:
    default_image (PIL.Image): The image with YOLO's default bounding boxes.
    summary_image (PIL.Image): The summary image containing label counts.

    Returns:
    numpy array: Combined image with the summary at the bottom.
    """
    # Get the size of the original image and the summary row
    default_width, default_height = default_image.size
    summary_height = summary_image.height

    # Create a new image that is tall enough to fit the default image and the summary
    combined_image = Image.new('RGB', (default_width, default_height + summary_height), (255, 255, 255))

    # Paste the default image and the summary image into the combined image
    combined_image.paste(default_image, (0, 0))
    combined_image.paste(summary_image, (0, default_height))

    # Convert back to OpenCV format for saving
    combined_frame = cv2.cvtColor(np.array(combined_image), cv2.COLOR_RGB2BGR)
    
    return combined_frame


def dms_to_dd(dms):
    """Convert DMS (Degrees, Minutes, Seconds) to Decimal Degrees (DD)."""
    degrees, minutes, seconds = dms
    return degrees + minutes / 60 + seconds / 3600


def convert_gps_to_utm(gps_info):
    """Convert GPSInfo from latitude/longitude to UTM Zone 19H and include altitude."""
    if isinstance(gps_info, dict) and all(key in gps_info for key in [1, 2, 3, 4, 6]):
        # Latitude conversion from DMS to decimal degrees
        latitude_dms = gps_info[2]  # e.g., (36.0, 31.0, 55.3176) (DMS)
        latitude_dd = dms_to_dd(latitude_dms)
        if gps_info[1] == 'S':  # Handle South latitudes
            latitude_dd = -latitude_dd

        # Longitude conversion from DMS to decimal degrees
        longitude_dms = gps_info[4]  # e.g., (71.0, 54.0, 41.762) (DMS)
        longitude_dd = dms_to_dd(longitude_dms)
        if gps_info[3] == 'W':  # Handle West longitudes
            longitude_dd = -longitude_dd

        # Get the altitude
        altitude = gps_info[6]

        # Set up pyproj for UTM Zone 19H (southern hemisphere)
        utm_proj = Proj(proj="utm", zone=19, south=True, ellps="WGS84")

        # Convert lat/lon (in degrees) to UTM
        easting, northing = utm_proj(longitude_dd, latitude_dd)

        # Return the UTM coordinates with altitude
        utm_result = {
            "UTM_Easting": easting,
            "UTM_Northing": northing,
            "Altitude": altitude
        }
        
        return utm_result
    else:
        print(f"GPSInfo missing necessary fields for UTM conversion. Available fields: {gps_info}")
        return None


def format_time(seconds):
    """
    Convert seconds into SRT timestamp format (hh:mm:ss,ms).

    Args:
    seconds (float): Time in seconds.

    Returns:
    str: Formatted timestamp.
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    sec = int(seconds % 60)
    milliseconds = int((seconds - int(seconds)) * 1000)
    return f"{hours:02}:{minutes:02}:{sec:02},{milliseconds:03}"


def get_image_metadata(image_path):
    """Extract metadata from the image including EXIF data."""
    metadata = {}
    try:
        image = Image.open(image_path)
        exif_data = image._getexif()
        if exif_data:
            for tag, value in exif_data.items():
                tag_name = ExifTags.TAGS.get(tag, tag)
                metadata[tag_name] = value
    except Exception as e:
        print(f"Error reading metadata from image: {str(e)}")
    
    return metadata


def generate_json_output(image_metadata, bboxes, model_info, label_summary, json_output_path):
    """
    Generates a JSON file with the model information, bounding boxes, image metadata, and class summary.
    Also includes UTM coordinates if GPS metadata is available.
    Saves the JSON file to the specified json_output_path.

    Args:
    image_metadata (dict): Metadata of the image (EXIF or GPS data).
    bboxes (list): List of bounding boxes and labels.
    model_info (dict): Information about the model and its configuration.
    label_summary (Counter): A summary of detected objects by class.
    json_output_path (str): Full file path to save the JSON output.

    Returns:
    None
    """
    from fractions import Fraction
    from PIL.TiffImagePlugin import IFDRational

    def clean_metadata(value):
        """Helper function to clean metadata values."""
        if isinstance(value, dict):
            return {k: clean_metadata(v) for k, v in value.items() if k != "MakerNote"}
        elif isinstance(value, list):
            return [clean_metadata(item) for item in value]
        elif isinstance(value, bytes):
            try:
                return value.decode('utf-8')
            except UnicodeDecodeError:
                return str(value)
        elif isinstance(value, (Fraction, IFDRational)):
            return float(value)
        elif isinstance(value, tuple) and all(isinstance(i, (Fraction, IFDRational)) for i in value):
            return tuple(float(i) for i in value)
        else:
            return value

    # Clean image metadata to handle bytes and IFDRational
    cleaned_image_metadata = clean_metadata(image_metadata)

    # Check if GPSInfo is present in metadata and convert to UTM
    utm_coordinates = None
    if "GPSInfo" in cleaned_image_metadata:
        utm_coordinates = convert_gps_to_utm(cleaned_image_metadata["GPSInfo"])

    # Create a structure for output data
    output_data = {
        "model_info": model_info,
        "bounding_boxes": bboxes,
        "label_summary": dict(label_summary),
        "image_metadata": cleaned_image_metadata
    }

    # Include UTM coordinates if available
    if utm_coordinates:
        output_data["UTM_Coordinates"] = utm_coordinates

    # Save the output data to the JSON file
    try:
        with open(json_output_path, 'w') as json_file:
            json.dump(output_data, json_file, indent=4, default=str)
        print(f"JSON output saved to {json_output_path}")
    except Exception as e:
        print(f"Error writing JSON output: {str(e)}")


def generate_summary_by_class(bboxes, labels_dict):
    """
    Generate a summary of predicted objects by class using the bounding boxes and label dictionary.

    Args:
    bboxes (list): A list of bounding boxes where each bounding box contains [label_idx, x_min, y_min, x_max, y_max, confidence].
    labels_dict (dict): A dictionary mapping label indices to class names.

    Returns:
    Counter: A counter object summarizing the number of objects detected per class.
    """
    label_counts = Counter()
    
    # Count the occurrences of each label (class)
    for bbox in bboxes:
        label_idx = bbox[0]  # First element is the label index
        label_name = labels_dict.get(label_idx, 'Unknown')  # Get class name from labels_dict
        label_counts[label_name] += 1
    
    return label_counts


def get_video_metadata(video_path):
    """Extract basic metadata from a video file using OpenCV."""
    metadata = {}
    try:
        video = cv2.VideoCapture(video_path)
        if not video.isOpened():
            raise ValueError("Error opening video file.")
        
        metadata['Frame_Width'] = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        metadata['Frame_Height'] = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        metadata['Frame_Count'] = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        metadata['FPS'] = int(video.get(cv2.CAP_PROP_FPS))
        metadata['Duration_Seconds'] = metadata['Frame_Count'] / metadata['FPS'] if metadata['FPS'] else 'Unknown'
        print(metadata)
    except Exception as e:
        print(f"Error reading metadata from video: {str(e)}")
    finally:
        video.release()
    
    return metadata


def generate_json_output_video(video_metadata, bboxes, model_info, label_summary, json_output_path):
    """
    Generates a JSON file with the model information, bounding boxes, video metadata, and class summary.
    Saves the JSON file to the specified json_output_path for videos.

    Args:
    video_metadata (dict): Metadata of the video.
    bboxes (list): List of bounding boxes and labels.
    model_info (dict): Information about the model and its configuration.
    label_summary (Counter): A summary of detected objects by class.
    json_output_path (str): Full file path to save the JSON output.

    Returns:
    None
    """

    output_data = {
        "model_info": model_info,
        "bounding_boxes": bboxes,
        "label_summary": dict(label_summary),
        "video_metadata": video_metadata
    }

    # Save the output data to the JSON file
    try:
        with open(json_output_path, 'w') as json_file:
            json.dump(output_data, json_file, indent=4, default=str)
        print(f"JSON output saved to {json_output_path}")
    except Exception as e:
        print(f"Error writing JSON output: {str(e)}")


def check_gpu_availability():
    # Check if GPU is available
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        print(f"💻  {num_gpus} GPU(s) available.")
        return num_gpus
    else:
        print("❌ No GPUs found, defaulting to CPU.")
        return 0

# Helper function to convert numpy data types to native Python data types
def numpy_to_native(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj


def prediction_result_to_dict(prediction_result, image_path):
    """
    Convert the prediction result from SAHI to a dictionary format compatible with desired JSON output.

    Args:
    prediction_result (PredictionResult): SAHI prediction result object.
    image_path (str): Path to the image being processed.

    Returns:
    dict: Dictionary containing object prediction details and image metadata.
    """
    logger.debug(f"Processing prediction_result: {type(prediction_result)}")
    
    # Convert object predictions and bounding box details
    return {
        "object_prediction_list": [
            {
                "bbox": [
                    numpy_to_native(pred.bbox.minx),  # x_min
                    numpy_to_native(pred.bbox.miny),  # y_min
                    numpy_to_native(pred.bbox.maxx - pred.bbox.minx),  # width
                    numpy_to_native(pred.bbox.maxy - pred.bbox.miny)  # height
                ],
                "category": pred.category.name,  # Class/category name of the object
                "score": numpy_to_native(pred.score.value)  # Confidence score of the prediction
            } for pred in prediction_result.object_prediction_list
        ],
        # Information about the image, including filename and its shape
        "image": {
            "filename": os.path.basename(image_path),  # Extract image filename
            "shape": prediction_result.image.shape if hasattr(prediction_result.image, 'shape') else None  # Check for image shape
        }
    }


def generate_summary_by_class_sahi_specific(object_predictions, labels_dict):
    """
    Generate a summary of detected objects by their class for SAHI predictions.

    Args:
    object_predictions (list): A list of SAHI ObjectPrediction instances.
    labels_dict (dict): A dictionary mapping class indices to class names.

    Returns:
    dict: A summary count of each object class.
    """
    label_counts = Counter()

    for prediction in object_predictions:
        try:
            class_name = prediction.category.name  # Access the category name directly
            label_counts[class_name] += 1
        except AttributeError:
            logger.error(f"Prediction object does not have expected attributes: {prediction}")
        except Exception as e:
            logger.error(f"Error processing prediction: {str(e)}")

    return dict(label_counts)


def apply_minimalistic_label_style_pil_sahi(image_path, object_predictions, labels, output_path, colors, label_summary):
    """
    Apply minimalistic labels showing only confidence levels on the image with enhanced styling,
    and include a summary of detected objects by class in the legend (in a single row).
    This version is compatible with SAHI's ObjectPrediction format.

    Args:
    image_path (str): Path to the image to modify.
    object_predictions (list): List of SAHI ObjectPrediction instances.
    labels (list): List of class names.
    output_path (str): Path to save the modified image.
    colors (list): List of RGB color tuples corresponding to labels.
    label_summary (dict): Summary of detected objects by class.

    Returns: None
    """
    image = Image.open(image_path)
    image_width, image_height = image.size

    # Calculate dynamic font size and border width based on image dimensions
    base_font_size = 12
    base_border_width = 1
    scaling_factor = (image_width + image_height) / 1500

    font_size = int(base_font_size * scaling_factor)
    border_width = max(1, int(base_border_width * scaling_factor))

    try:
        font = ImageFont.truetype("arial.ttf", font_size)
    except IOError:
        font = ImageFont.load_default()

    draw = ImageDraw.Draw(image, 'RGBA')

    # Handle empty predictions
    if not object_predictions:
        logger.warning("No object predictions found for this image.")
        return

    for pred in object_predictions:
        x_min, y_min = int(pred.bbox.minx), int(pred.bbox.miny)
        x_max, y_max = int(pred.bbox.maxx), int(pred.bbox.maxy)
        confidence = pred.score.value
        label = pred.category.name
        label_idx = labels.index(label)

        color = colors[label_idx % len(colors)]

        # Draw rectangle
        draw.rectangle([x_min, y_min, x_max, y_max], outline=color + (255,), width=border_width)

        # Display confidence
        confidence_text = f"{confidence:.2f}"
        text_bbox = draw.textbbox((0, 0), confidence_text, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]
        
        box_padding = 4
        text_background_x1 = max(0, x_min)
        text_background_y1 = max(0, y_min - text_height - 2 * box_padding)
        text_background_x2 = text_background_x1 + text_width + 2 * box_padding
        text_background_y2 = text_background_y1 + text_height + 2 * box_padding

        # If not enough space above the box, move text below the bounding box
        if text_background_y1 < 0:
            text_background_y1 = y_max + 2 * box_padding
            text_background_y2 = text_background_y1 + text_height + 2 * box_padding

        # Draw transparent rectangle for confidence background
        draw.rectangle([text_background_x1, text_background_y1, text_background_x2, text_background_y2], fill=color + (100,))

        # Draw confidence text
        text_x = text_background_x1 + box_padding
        text_y = text_background_y1 + box_padding
        draw.text((text_x, text_y), confidence_text, fill=(255, 255, 255, 255), font=font)

    # Create legend
    legend_height = font_size + 20
    legend_image = Image.new('RGB', (image_width, image_height + legend_height), (255, 255, 255))
    legend_image.paste(image, (0, 0))

    draw_legend = ImageDraw.Draw(legend_image)
    legend_start_x = 10
    legend_start_y = image_height + 10
    separator = " | "

    for idx, label in enumerate(labels):
        color = colors[idx % len(colors)]
        count = label_summary.get(label, 0)

        draw_legend.rectangle([legend_start_x, legend_start_y, legend_start_x + 20, legend_start_y + 20], fill=color)
        draw_legend.text((legend_start_x + 30, legend_start_y), f"{label}: {count}", fill=color, font=font)
        
        legend_start_x += font_size * len(f"{label}: {count}") + 50
        draw_legend.text((legend_start_x, legend_start_y), separator, fill=(0, 0, 0), font=font)
        legend_start_x += font_size * len(separator)

    legend_image.save(output_path)
    logger.info(f"Styled image with confidence and legend saved to {output_path}")



def get_default_validation_images_dir(img_size, project_root_dir):
    """
    Get the default directories for validation images and labels based on the image size and project root directory.
    Returns paths for both the 'images' and 'labels' subfolders.
    """
    # Corrected base directory to include 'yolov8_project' twice
    base_dir = os.path.join(
        project_root_dir, 
        f'Lencu-{img_size}-merge2to11-except5',

        'test'
    )
    
    images_dir = os.path.join(base_dir, 'images')
    labels_dir = os.path.join(base_dir, 'labels')

    return images_dir, labels_dir


def construct_label_path(image_path, labels_dir):
    """
    Construct the label path from the given image path.
    """
    # Extract the base filename (without the extension)
    base_filename = os.path.basename(image_path)
    
    # Print the base filename for debugging purposes
    print(f"Base filename: {base_filename}")

    # If the image file has ".rf." in the name (Roboflow's convention), keep the important part
    if ".rf." in base_filename:
        base_filename = base_filename.split(".rf.")[0] + ".rf." + base_filename.split(".rf.")[1]
    
    # Replace the image file extension with .txt (for the label)
    label_filename = os.path.splitext(base_filename)[0] + ".txt"
    
    # Print the constructed label filename for debugging
    print(f"Label filename: {label_filename}")
    
    # Construct the full label path
    label_path = os.path.join(labels_dir, label_filename)
    
    # Print the full label path for debugging
    print(f"Constructed label path: {label_path}")
    
    return label_path

def safe_float(value):
    """
    Safely convert a value to float.
    
    Args:
    - value: The value to convert.
    
    Returns:
    - Float value if conversion is successful, None otherwise.
    """
    try:
        if isinstance(value, np.ndarray):
            return float(value.item())
        if isinstance(value, np.float64):
            return float(value)
        return float(value)
    except (TypeError, ValueError):
        return None


def create_flexible_summary(results, class_names, instance_counts):
    """
    Create a flexible summary from the validation results.
    
    Args:
    - results: The results object from YOLO validation.
    - class_names: List of class names.
    - instance_counts: Dictionary of instance counts for each class.
    
    Returns:
    - A string containing a summary of the validation results.
    """
    summary = "Class     Instances     Precision     Recall     mAP50     mAP50-95\n"
    for i, class_name in enumerate(class_names):
        precision = results.box.p[i] if hasattr(results.box, 'p') else None
        recall = results.box.r[i] if hasattr(results.box, 'r') else None
        map50 = results.maps[i] if hasattr(results, 'maps') else None
        map50_95 = results.box.maps[i] if hasattr(results.box, 'maps') else None
        instances = instance_counts[class_name]
        summary += f"{class_name:<10} {instances:<12} {precision:<12} {recall:<12} {map50:<12} {map50_95:<12}\n"
    return summary



def get_instance_counts_from_labels(results):
    """
    Extract the instance counts for each class from the YOLOv8 results using the labels.

    Args:
    - results: The YOLOv8 results object.

    Returns:
    - instance_counts (dict): A dictionary mapping class names to instance counts.
    """
    class_names = list(results.names.values())
    instance_counts = {class_name: 0 for class_name in class_names}

    # Use the ground truth labels to count instances for each class
    for label_file in results.label_files:
        with open(label_file, 'r') as f:
            for line in f:
                class_id = int(line.split()[0])  # The first column in the label file is the class ID
                class_name = class_names[class_id]
                instance_counts[class_name] += 1

    return instance_counts


def create_flexible_summary_batch(class_names, instance_counts, precisions, recalls, mAP50, mAP50_95):
    """
    Create a flexible summary batch from the parsed data.

    Args:
    - class_names (list): List of class names.
    - instance_counts (dict): Dictionary containing instance counts.
    - precisions (list): List of precision values for each class.
    - recalls (list): List of recall values for each class.
    - mAP50 (float): mAP50 value.
    - mAP50_95 (float): mAP50-95 value.

    Returns:
    - summary (str): A string containing the formatted summary.
    """
    summary = "Class     Images  Instances      Box(P          R      mAP50  mAP50-95)\n"

    for idx, class_name in enumerate(class_names):
        instances = instance_counts.get(class_name, 0)
        precision = precisions[idx]
        recall = recalls[idx]
        summary += f"{class_name:<12} {45:<8} {instances:<10} {precision:.3f} {recall:.3f} {mAP50:.3f} {mAP50_95:.3f}\n"

    return summary



def create_excel_from_json(output_dir, excel_filename="validation_results.xlsx"):
    # Initialize an empty list to store each run's data
    data = []

    # Loop over all JSON files in the output directory
    for filename in os.listdir(output_dir):
        if filename.endswith(".json"):
            file_path = os.path.join(output_dir, filename)

            # Read JSON data
            with open(file_path, 'r') as file:
                json_data = json.load(file)

            # Extraer precisión y recall por clase
            precisions = json_data["metrics"]["precision"]
            recalls = json_data["metrics"]["recall"]

            # Extract required fields
            row = {
                # Identificación y Modelo
                "ID ClearML": json_data["clearml_task_id"],
                "Modelo": f"yolov8{json_data['model_name']}.pt",  # Dynamically construct the model name


                 # Configuración
                "Image Size": json_data["img_size"],
                "Training Image Size": json_data["training_image_size"],
                "Batch Size": json_data["batch"],
                "Confidence Threshold": json_data["confidence_threshold"],
                "Max Detections": json_data["max_det"],
                "Seed": json_data["seed"],
                "Device": json_data["device"],
                
                # Métricas globales
                "Precision Valid Mean": json_data["metrics"]["mean_precision"],
                "Recall Valid Mean": json_data["metrics"]["mean_recall"],
                "mAP50": json_data["metrics"]["mAP50"],  # Add mAP50
                "mAP50-95": json_data["metrics"]["mAP50-95"],  # Add mAP50-95

                # Métricas por clase
                "AMBEL Precision": precisions[0] if len(precisions) > 0 else None,
                "AMBEL Recall": recalls[0] if len(recalls) > 0 else None,
                "LENCU Precision": precisions[1] if len(precisions) > 1 else None,
                "LENCU Recall": recalls[1] if len(recalls) > 1 else None,
                "POLAV Precision": precisions[2] if len(precisions) > 2 else None,
                "POLAV Recall": recalls[2] if len(recalls) > 2 else None,
                "POLPE Precision": precisions[3] if len(precisions) > 3 else None,
                "POLPE Recall": recalls[3] if len(recalls) > 3 else None,

                # Métricas de tiempo
                "Mean Time per Image ms": json_data["metrics"]["mean_time_per_image_ms"],
                "Mean Time Across ms": json_data["metrics"]["mean_time_across_runs_ms"],
                "Std Time Across ms": json_data["metrics"]["std_dev_time_ms"],
                "Warmup Runs": json_data["metrics"].get("pre_measurement_warmup_runs", 0),

                # Información adicional
                "Timestamp": json_data["timestamp"],
                "Run Number": json_data["run"],
                "YAML Path": json_data["yaml_path"],
                "Model Path": json_data["best_model_path"]
            }

            # Append row to data list
            data.append(row)

    # Convert data to a DataFrame
    df = pd.DataFrame(data)

    # Save DataFrame to Excel
    excel_path = os.path.join(output_dir, excel_filename)
    df.to_excel(excel_path, index=False)
    print(f"Excel file created at: {excel_path}")


def get_best_model(root_dir, img_size, model_version, output_dir="metrics_output"):
    """
    Function to find the best model based on a weighted score of mAP50-95 and F1-score.

    Args:
        root_dir (str): Root directory of the project.
        img_size (int): Image size used for training.
        model_version (str): YOLO model version used for training ('n', 's', 'm', 'l', 'x').
        output_dir (str): Directory where the metrics JSON file will be saved.

    Returns:
        str: Path to the best model file, or None if no model is found.
    """
    best_score = 0
    best_model_path = None
    best_epoch = None
    best_metrics = {}

    # Create output directory for metrics if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    metrics_json_path = os.path.join(output_dir, f"best_metrics_{model_version}_{img_size}.json")

    # Directory containing training results
    train_results_dir = os.path.join(root_dir, 'training_results', f'img_size_{img_size}', f'yolov8_{model_version}')

    if not os.path.exists(train_results_dir):
        logging.warning(f"No training results found for image size {img_size} and model version {model_version}")
        return None

    # Iterate through all training directories for the specified image size and model version
    for dirpath, dirnames, filenames in os.walk(train_results_dir):
        for filename in filenames:
            if filename == 'results.csv':  # Assuming each run has a results.csv file
                results_path = os.path.join(dirpath, filename)
                try:
                    df = pd.read_csv(results_path)

                    # Strip leading and trailing spaces from column names
                    df.columns = df.columns.str.strip()

                    # Check for required columns (corrected column names)
                    required_columns = [
                        'metrics/mAP50-95(B)', 'metrics/precision(B)', 'metrics/recall(B)'
                    ]
                    if not all(col in df.columns for col in required_columns):
                        logging.warning(f"Missing required columns in {results_path}. Found columns: {df.columns.tolist()}")
                        continue

                    # Iterate through epochs to calculate weighted score
                    for index, row in df.iterrows():
                        mAP50_95 = row['metrics/mAP50-95(B)']
                        precision = row['metrics/precision(B)']
                        recall = row['metrics/recall(B)']

                        # Calculate F1-score
                        f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

                        # Calculate weighted score
                        weighted_score = (0.7 * mAP50_95) + (0.3 * f1_score)

                        # Check if this is the best score
                        if weighted_score > best_score:
                            best_score = weighted_score
                            best_epoch = index + 1  # Epoch index starts at 0
                            best_model_path = os.path.join(dirpath, 'weights', 'best.pt')
                            best_metrics = {
                                'epoch': best_epoch,
                                'mAP50-95': mAP50_95,
                                'precision': precision,
                                'recall': recall,
                                'f1_score': f1_score,
                                'weighted_score': weighted_score,
                                'model_path': best_model_path
                            }

                except Exception as e:
                    logging.warning(f"Error reading {results_path}: {e}")

    # Save the best metrics to a JSON file
    if best_metrics:
        with open(metrics_json_path, 'w') as json_file:
            json.dump(best_metrics, json_file, indent=4)
        print(f"Best metrics saved to {metrics_json_path}")

    # Return the best model path
    if best_model_path and os.path.exists(best_model_path):
        print(f"Best model found: {best_model_path} with weighted score: {best_score:.4f}")
        return best_model_path
    else:
        print("No suitable model found based on the given criteria.")
        return None


'''
E:\vuelos-17sept-24\droneMini\107MEDIA\DJI_0096.MP4
E:\vuelos-17sept-24\droneMini\107MEDIA\DJI_0095.JPG
'''

#C:\Users\Usuario\Downloads\MVI_1720.MOV


import os
import json
import logging
from datetime import datetime
import io
import sys
from predict_yolo import run_inference, run_inference_sahi_directory, run_inference_sahi_single_image, run_inference_single_image, run_inference_video
from validation_yolo import run_validation_on_dataset, run_validation_on_image
from yolo_training import train_yolo
from clearml_utils import fetch_and_store_clearml_data, fetch_and_store_clearml_data_for_project, setup_credentials_clearml_task
from utils import download_yolo_model, get_default_test_images_dir, get_default_validation_images_dir, get_project_root, select_model_version, get_best_model
from clearml import Task
import torch
from utils_prompts import print_separator, select_batch_size, select_clearml_task_or_project, select_full_path_image, select_full_path_video, select_generate_slices, select_mode, select_clearml_account, select_device_order, select_image_size, select_single_image_for_validation, select_training_image_size, select_validation_mode, select_yolo_version
from utils_prompts import select_confidence_threshold, select_overlap_ratio, select_path_with_sahi, select_processing_json_order, select_slice_size, get_directory_input_images

# Define logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():

    #reset_cuda_env()
    # Define a standardized labels dictionary and colors
    labels_dict = {
        0: 'AMBEL', 
        1: 'LENCU', 
        2: 'POLAV', 
        3: 'POLPE',
        4: 'BACKGROUND'
    }
    
    # Define corresponding colors for labels
    colors = [
        (255, 0, 255),  # Color for AMBEL
        (0, 0, 255),    # Color for LENCU (Always blue as per previous request)
        (255, 0, 0),    # Color for POLAV
        (255, 165, 0)   # Color for POLPE
    ]


    # Step 1: Select Mode
    mode = select_mode()
    # Select device order
    devices = select_device_order()
    print(torch.cuda.is_available())  # Should return True if CUDA is working
    print(torch.cuda.get_device_name(0))  # Prints the name of the GPU
    current_dir = os.path.dirname(__file__)  
    project_root_dir = os.path.abspath(os.path.join(current_dir, '..')) 
    test_images_root_dir = os.path.abspath(os.path.join(current_dir, '..', '..'))
    yolo_version = select_yolo_version()

    # Step 1: ClearML Account Selection and Setup
    account_type, account_name = select_clearml_account()  # Select account and get account_name
    try:
        setup_credentials_clearml_task(account_type)  # Connect ClearML once at the start
    except Exception as e:
        logging.error(f"An error occurred during credentials setup: {str(e)}")
        return  # Exit if credentials setup fails

    logging.info(f"Using ClearML account: {account_name}")

    # Step 3: If Train Mode is Selected
    if mode == '1':
        # Select JSON reading order
        json_order = select_processing_json_order()

        
        print(yolo_version)

        # Load the JSON file
        config_file = 'config.json'
        with open(config_file, 'r') as f:
            configs = json.load(f)

        if json_order == "descending":
            configs = list(reversed(configs))  # Reverse the list for descending order

        # Initialize logging and buffers
        c_buffer = io.StringIO()
        sys.stdout = c_buffer

        # Loop over the configurations and train
        for config in configs:
            model_version = config["model_name"]
            yolo_version = yolo_version
            imgsz = config["imgsz"]
            batch_size = config["batch_size"]
            num_runs = 5 
            base_seed = 42
            epochs = config["epoch"]

            # Ensure model file is available
            download_yolo_model(yolo_version, model_version)

            task_name = f"YOLOv{yolo_version}_{model_version}_training_{imgsz}x{imgsz}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

            try:
                # Normalize YAML file path
                config["yaml_file"] = os.path.normpath(config["yaml_file"])

                # Call the train_yolo function with parameters from the config
                results = train_yolo(config, model_version, batch_size, imgsz, devices, num_runs, base_seed, epochs, yolo_version)

                # Log best results
                logging.info(f"Best results for YOLOv8{model_version} with image size {imgsz}:")
                for metric, value in results['results'].items():
                    logging.info(f"  {metric}: {value}")
                logging.info(f"Best seed: {results['seed']}")

            except Exception as e:
                logging.error(f"An error occurred during execution: {str(e)}")


        # Write log buffer to file if it contains any messages
        log_contents = c_buffer.getvalue()
        sys.stdout = sys.__stdout__  # Reset stdout to default

        print(log_contents)  # Or handle it as needed

        if log_contents:
            parent_dir = os.path.abspath(os.path.join(os.getcwd(), '..'))
            log_dir = os.path.join(parent_dir, 'outputs', 'logs')
            os.makedirs(log_dir, exist_ok=True)
            log_filename = f"yolov8_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
            log_filepath = os.path.join(log_dir, log_filename)
            with open(log_filepath, 'w') as log_file:
                log_file.write(log_contents)

    # Step 4: If Predict Multiple Images Mode is Selected
    elif mode == '2':
        img_size = select_image_size()
        model_version = select_model_version()

        print(f"Project Root Directory: {project_root_dir}")
        print(f"Test Images Root Directory: {test_images_root_dir}")
        print_separator()

        # Get the default directory using the helper function
        default_test_images_dir = get_default_test_images_dir(img_size, test_images_root_dir)
        test_images_dir = get_directory_input_images(default_test_images_dir, "test images")

        print(f"Test Images Directory: {test_images_dir}")
        print_separator()

        if not os.path.isdir(test_images_dir):
            print(f"Test images directory not found: {test_images_dir}")
            return

        # Load the trained model
        best_model_path = get_best_model(project_root_dir, img_size, model_version)
        confidence_threshold = select_confidence_threshold()

        if not best_model_path:
            print(f"No trained model found for image size {img_size} and model version {model_version}.")
            return

        print(f"Best trained model found: {best_model_path}")
        
        try:
            run_inference(img_size, model_version, test_images_dir, project_root_dir, confidence_threshold, labels_dict, colors, img_size)
        except Exception as e:
            logging.error(f"An error occurred during prediction: {str(e)}")   

    # Step 5: If Predict Single Image Mode is Selected
    elif mode == '3':
        print(f"Project Root Directory: {project_root_dir}")
        print_separator()


        img_size = select_image_size()
        model_version = select_model_version()

        # Retrieve the best model for inference
        best_model_path = get_best_model(project_root_dir, img_size, model_version)  
        if not best_model_path:
            print(f"No trained model found for image size {img_size} and model version {model_version}.")
            return

        print(f"Best trained model found: {best_model_path}")

        

        # Get the confidence threshold
        confidence_threshold = select_confidence_threshold()
        single_image_path = select_full_path_image()

        # Construct the output directory for the prediction
        output_dir = os.path.join(project_root_dir, 'outputs', 'individual_image', f'img_size_{img_size}', f'yolov8_{model_version}', os.path.basename(single_image_path).split('.')[0])
        os.makedirs(output_dir, exist_ok=True)

        # Run prediction on the single image
        try:
            run_inference_single_image(single_image_path, model_version, best_model_path, output_dir, confidence_threshold, labels_dict, colors, img_size)
            print(f"Prediction completed. Results saved to {output_dir}")
        except Exception as e:
            logging.error(f"An error occurred during prediction: {str(e)}")

    # Step 6: If Predict Video Mode is Selected
    elif mode == '4':
        # Get the root directory of the project
        project_root_dir = get_project_root()
        print(f"Project Root Directory: {project_root_dir}")
        print_separator()

        # Select image size and model version
        img_size = select_image_size()
        model_version = select_model_version()

        # Retrieve the best model for video inference
        best_model_path = get_best_model(project_root_dir, img_size, model_version)  
        if not best_model_path:
            print(f"No trained model found for image size {img_size} and model version {model_version}.")
            return

        print(f"Best trained model found: {best_model_path}")
        print_separator()
    
        # Get confidence threshold from the user
        confidence_threshold = select_confidence_threshold()
        video_path = select_full_path_video()

        # Construct the output directory for the prediction
        output_dir = os.path.join(project_root_dir, 'outputs', 'video_predictions', f'img_size_{img_size}', f'yolov8_{model_version}', os.path.basename(video_path).split('.')[0])
        os.makedirs(output_dir, exist_ok=True)
        # Run prediction on the video
        try:
            run_inference_video(video_path, model_version, best_model_path, output_dir, confidence_threshold, labels_dict, colors)
            print(f"Video prediction completed. Results saved to {output_dir}")
        except Exception as e:
            logging.error(f"An error occurred during video prediction: {str(e)}")

    # If Predict Single Image with SAHI is Selected
    if mode == '5':
        # Get user choice for single image or directory processing
        print_separator()

        # Call the process_with_sahi function to get the image or directory path
        model_version = select_model_version()
        img_size = select_image_size()
        default_test_images_dir = get_default_test_images_dir(img_size, test_images_root_dir)
        choice, path = select_path_with_sahi(default_test_images_dir)
        print(f"Running SAHI inference on: {path}")

        # Retrieve the best model for inference
        best_model_path = get_best_model(project_root_dir, img_size, model_version)

        print(f"Best trained model found: {best_model_path}")
        if not best_model_path:
            print(f"No trained model found for image size {img_size} and model version {model_version}.")
            return

        # Set confidence threshold, slice size, and overlap ratio for SAHI inference
        confidence_threshold = select_confidence_threshold()
        slice_size = select_slice_size(img_size)  # Default slice size is based on image size
        overlap_ratio = select_overlap_ratio()    # Overlap ratio for slicing
        
        # Ask the user whether to generate slice images
        generate_slices = select_generate_slices()

        # Now process the image or directory based on user's choice
        if choice == '1':  # Process single image
            try:
                run_inference_sahi_single_image(
                    image_path=path,              # Single image path
                    device=devices,
                    slice_size=slice_size,
                    overlap_ratio=overlap_ratio,
                    best_model_path=best_model_path,
                    confidence_threshold=confidence_threshold,
                    img_size=img_size,
                    batch_output_dir=os.path.join(project_root_dir, 'outputs', 'SAHI', 'predict_single_image', f'img_size_{img_size}', f'yolov8_{model_version}'),  # Update this to pass batch_output_dir
                    model_version=model_version,
                    labels_dict=labels_dict,
                    colors=colors,
                    generate_slices=generate_slices,
                    images_dir=path,  # In single image mode, this is the directory of the image
                    batch_mode=False  # Make sure to set this to False for single image
                )
            except Exception as e:
                logging.error(f"An error occurred during single image inference: {str(e)}")


        elif choice == '2' or choice == '3':  # Process directory
            try:
                run_inference_sahi_directory(
                    device=devices,                   # The device to run inference on (e.g., 'cuda' or 'cpu')
                    slice_size=slice_size,            # The size of slices for SAHI
                    overlap_ratio=overlap_ratio,      # The overlap ratio for slicing
                    best_model_path=best_model_path,  # Path to the trained YOLO model
                    confidence_threshold=confidence_threshold,  # Confidence threshold for predictions
                    img_size=img_size,                # Image size used for slicing and inference
                    images_dir=path,                  # Directory containing images
                    project_root_dir=project_root_dir,  # Root project directory
                    model_version=model_version,      # Model version (e.g., 'n', 's', 'm', 'l', 'x')
                    labels_dict=labels_dict,          # Dictionary mapping class indices to class names
                    colors=colors,                    # List of RGB color tuples corresponding to labels
                    generate_slices=generate_slices   # Whether to generate individual sliced-region predictions
                )
                print(f"Running SAHI inference on directory: {path}")
            except Exception as e:
                logging.error(f"An error occurred during directory inference: {str(e)}")

    if mode == '6':
        account_type, account_name = select_clearml_account()  # Select account and get account_name
        setup_credentials_clearml_task(account_type)
        logging.info(f"Using ClearML account: {account_name}")

        # Sub-prompt to select either Task or Project
        selection = select_clearml_task_or_project()

        if selection == 'task':
            # Prompt the user to enter the ClearML Task ID and Project Name
            task_id = input("Enter the ClearML Task ID to fetch data: ").strip()
            project_name = input("Enter the ClearML Project Name: ").strip()

            if not task_id or not project_name:
                print("❌ Invalid Task ID or Project Name. Please enter valid values.")
                return

            # Fetch and store the ClearML data for the specific task (not part of a project)
            fetch_and_store_clearml_data(task_id, project_name, project_root_dir, account_name, is_project=False)

        elif selection == 'project':
            # Prompt the user to enter the ClearML Project Name
            project_name = input("Enter the ClearML Project Name to fetch all tasks: ").strip()

            if not project_name:
                print("❌ Invalid Project Name. Please enter a valid ClearML Project Name.")
                return

            # Fetch and store the ClearML data for all tasks in the project
            fetch_and_store_clearml_data_for_project(project_name, project_root_dir, account_name)

    if mode == '7':
        validate_mode = select_validation_mode()  # Request the user to select validation mode (single image or dataset)

        # Ask user for image size to use in the path
        img_size = select_image_size()

        if validate_mode == '3':
            # Ask user for the training image size if using cross-validation mode
            training_img_size = select_training_image_size()
        else:
            training_img_size = img_size  # Use the same image size for validation and training if not in cross-validation mode
        
        model_version = select_model_version()
        
        # Use the  select_batch_size function to get batch size
        batch_size = select_batch_size()

   
        # Retrieve the best model for inference
        best_model_path = get_best_model(project_root_dir, training_img_size, model_version)
        
        confidence_threshold = select_confidence_threshold()

        print(f"Best trained model found: {best_model_path}")
        if not best_model_path:
            print(f"No trained model found for image size {img_size} and model version {model_version}.")
            return

        # Use os.path to get the project root directory (returning 3 levels up)
        current_dir = os.path.dirname(os.path.abspath(__file__))
        #project_images_dir = os.path.abspath(os.path.join(current_dir, '..', '..', '..'))
        project_images_dir = "F:\Datasets\lencu-all-f5-roboflow"
        project_root_dir = os.path.abspath(os.path.join(current_dir, '..'))

        # If validating a single image
        if validate_mode == '1':
            single_image_path = select_single_image_for_validation()


            # Get the validation image and label directories using the provided image size
            images_dir, labels_dir = get_default_validation_images_dir(img_size, project_images_dir)

            # Find the corresponding label for the selected image
            label_path = os.path.join(labels_dir, os.path.basename(single_image_path).rsplit('.', 1)[0] + '.txt')

            if not os.path.exists(label_path):
                logger.error(f"Label file not found for image: {single_image_path}")
                return

            # Run validation on the single image
            run_validation_on_image(
                image_path=single_image_path, 
                label_path=label_path, 
                devices=devices,  # Pass the list of devices or 'cpu'
                best_model_path=best_model_path,  # Path to your trained model
                confidence_threshold=confidence_threshold,  # Adjust as needed
                img_size=img_size,  # Use user-provided image size
                project_root_dir=project_root_dir,  # Add this line
                model_name=model_version,
                batch_size=batch_size,
                labels_dict=labels_dict,                
            )
            logger.info(f"Validation complete for image: {single_image_path}")


        # If validating the whole dataset
        elif validate_mode in ['2', '3']:  # Validation on the entire dataset or cross-validation
            # Get the validation image and label directories using the provided image size
            try:
                num_runs = int(input("Enter the number of validation runs to perform: "))
            except ValueError:
                print("Invalid input. Please enter a valid integer.")
                return
            images_dir, labels_dir = get_default_validation_images_dir(img_size, project_images_dir)
            print(images_dir, labels_dir)
            yaml_path = os.path.join(project_root_dir, f'data_{img_size}.yaml')
            # Run validation on the dataset
            run_validation_on_dataset(
                images_dir,  # Pass the directory for the dataset
                labels_dir,  # Pass the labels directory
                devices,  # or 'cpu' depending on your setup
                best_model_path,  # Path to your trained model
                confidence_threshold,  # Adjust as needed
                img_size,  # Use user-provided image size
                project_root_dir, 
                model_version,
                yaml_path,
                batch_size=batch_size,
                training_image_size=training_img_size,
                num_runs=num_runs,  # Pass user-defined number of runs
                account_type=account_type

            )
            logger.info(f"Validation complete for dataset in {images_dir}")

        else:
            logger.error("Invalid validation mode selected.")




if __name__ == "__main__":
    main()


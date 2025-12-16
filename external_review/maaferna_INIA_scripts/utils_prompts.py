import os
import logging


from utils import check_gpu_availability, convert_path_to_windows_format


# Styled separator to enhance terminal output
def print_separator(char='-', length=50):
    print(char * length)

# Mode selection with styled output
def select_mode():
    """
    Utility function to select the mode of operation.
    """
    while True:
        print_separator('=')
        print("🚀  Select mode:")
        print_separator()
        print("1 - 🛠️  Train")
        print("2 - 📷  Predict Multiple Images")
        print("3 - 🖼️  Predict Single Image")
        print("4 - 🎥  Predict Video")
        print("5 - 🗂️  Process Image(s) Or Directory with SAHI")
        print("6 - 🗄️  Retrieve Data from ClearML")
        print("7 - 🧪  Validation Mode")  # Add validation mode here
        print_separator()
        mode = input("Enter 1, 2, 3, 4, 5, 6, or 7: ").strip()
        if mode in ['1', '2', '3', '4', '5', '6', '7']:  # Include validation mode '7'
            return mode
        print("❌ Invalid input. Please enter 1, 2, 3, 4, 5, 6, or 7.")
        print_separator()


# ClearML account selection with styled output
def select_clearml_account():
    """
    Utility function to select ClearML account.
    """
    while True:
        print_separator('=')
        print("🌐  Select ClearML account:")
        print_separator()
        print("0 - Personal account")
        print("Press Enter - Original account")
        print_separator()
        account_type = input("Select ClearML account (0 for personal, press Enter for original): ").strip()


        if account_type == '0':
            return '0', 'MarcoParra'
        elif account_type == '':
            return '1', 'LorenzoLeon'
        print("❌ Invalid input. Please enter 0 for personal or press Enter for original.")
        print_separator()

def select_device_order():
    """
    Utility function to select the order of devices (GPUs) or fallback to CPU.
    """
    print_separator('=')
    num_gpus = check_gpu_availability()

    if num_gpus > 0:
        # If there are GPUs, allow the user to select the device order
        while True:
            print("💻  Select device order:")
            print_separator()
            print("0 - [0, 1] (GPU 0 first)")
            print("1 - [1, 0] (GPU 1 first)")
            print_separator()
            device_order = input("Enter 0 or 1: ").strip()

            if device_order == '0':
                return [0, 1]  # GPU 0 first
            elif device_order == '1':
                return [1, 0]  # GPU 1 first
            else:
                print("❌ Invalid input. Please enter 0 or 1 for device order.")
                print_separator()
    else:
        # If no GPUs are available, return CPU
        return "cpu"


def select_yolo_version():
    """
    Utility function to allow the user to select the YOLO version and size interactively.
    """
    print_separator('=')
    print("🌟 Select YOLO Version and Size")
    print_separator()
    
    yolo_version = None

    while not yolo_version:
        print("Select YOLO version:")
        print_separator()
        print("1 - YOLOv8")
        print("2 - YOLOv11")
        print_separator()
        version_choice = input("Enter 1 or 2: ").strip()

        if version_choice == '1':
            yolo_version = '8'
        elif version_choice == '2':
            yolo_version = '11'
        else:
            print("❌ Invalid input. Please enter 1 or 2 for YOLO version.")
            print_separator()
    
    return yolo_version

# Image size selection with styled output
def select_image_size():
    """Utility function to prompt the user to select an image size."""
    while True:
        try:
            acceptable_img_sizes = [416, 640, 1024, 2048]

            print_separator('=')
            img_size = int(input(f"📏  Enter image size {acceptable_img_sizes}: ").strip())
            print_separator()
            if img_size in acceptable_img_sizes:
                return img_size
            else:
                print(f"❌ Invalid input. Please enter one of the following sizes: {', '.join(map(str, acceptable_img_sizes))}.")
        except ValueError:
            print(f"❌ Invalid input. Please enter one of the following sizes: {', '.join(map(str, acceptable_img_sizes))}.")
        print_separator()


# Training image size selection with styled output
def select_training_image_size():
    """Utility function to prompt the user to select the training image size."""
    while True:
        try:
            acceptable_img_sizes = [416, 640, 1024, 2048]

            print_separator('=')
            print("🖼️  Select the image size used during training")
            img_size = int(input(f"📏  Enter training image size {acceptable_img_sizes}: ").strip())
            print_separator()
            if img_size in acceptable_img_sizes:
                return img_size
            else:
                print(f"❌ Invalid input. Please enter one of the following sizes: {', '.join(map(str, acceptable_img_sizes))}.")
        except ValueError:
            print(f"❌ Invalid input. Please enter one of the following sizes: {', '.join(map(str, acceptable_img_sizes))}.")
        print_separator()


# Custom path selection with styled output
def select_custom_path(description):
    """
    Utility function to prompt the user to select a custom path or use the default.
    """
    while True:
        print_separator('=')
        print(f"📂  Press Enter to use the default {description} path.")
        custom_path = input(f"Or type a custom path for {description}: ").strip()
        print_separator()
        if custom_path == '' or os.path.exists(custom_path):
            return custom_path
        else:
            print(f"❌ Invalid path: {custom_path}. Please enter a valid path.")
            print_separator()

# Confidence threshold selection with styled output
def select_confidence_threshold():
    """
    Prompt the user to enter a confidence threshold value between 0.0 and 1.0.
    Returns:
        float: The confidence threshold value entered by the user.
    """
    while True:
        try:
            print_separator('=')
            confidence_threshold = float(input("🔒  Enter the confidence threshold (0.0 to 1.0): ").strip())
            print_separator()
            if 0.0 <= confidence_threshold <= 1.0:
                return confidence_threshold
            else:
                print("❌ Invalid input. Please enter a value between 0.0 and 1.0.")
        except ValueError:
            print("❌ Invalid input. Please enter a numerical value between 0.0 and 1.0.")
        print_separator()

# JSON processing order selection with styled output
def select_processing_json_order():
    while True:
        print_separator('=')
        print("📑  Select JSON reading order:")
        print_separator()
        print("0 - Ascending (read JSON in original order)")
        print("1 - Descending (read JSON in reverse order)")
        print_separator()
        read_order = input("Enter 0 or 1: ").strip()

        if read_order == '0':
            return "ascending"
        elif read_order == '1':
            return "descending"
        else:
            print("❌ Invalid input. Please enter 0 or 1 for reading order.")
        print_separator()

# Video path selection with styled output
def select_full_path_video():
    while True:
        print_separator('=')
        video_path = input("📹  Enter the full path of the video to predict: ").strip()
        video_path = convert_path_to_windows_format(video_path)  # Convert the path format for Windows
        
        if os.path.isfile(video_path) and video_path.lower().endswith(('.mp4', '.avi', '.mov')):
            return video_path
        print(f"❌ Invalid video path: {video_path}. Please enter a valid path to a .mp4, .avi, or .mov file.")
        print_separator()

# Single image path selection with styled output
def select_full_path_image():
    while True:
        print_separator('=')
        single_image_path = input("🖼️  Enter the full path of the image to predict: ").strip()
        single_image_path = convert_path_to_windows_format(single_image_path)  # Convert the path format for Windows
        
        if os.path.isfile(single_image_path) and single_image_path.lower().endswith(('.png', '.jpg', '.jpeg')):
            return single_image_path
        print(f"❌ Invalid image path: {single_image_path}. Please enter a valid path to a .png, .jpg, or .jpeg file.")
        print_separator()


# Process Image(s) or Directory using SAHI
def select_path_with_sahi(default_test_images_dir):
    """
    Processes images or directories with SAHI based on user input.

    Args:
        default_test_images_dir (str): Default directory for batch image processing.
    
    Returns:
        tuple: choice (str) and path (str). 'choice' represents whether it is a single image or directory.
               'path' is the selected path to the image or directory.
    """
    print_separator('=')
    print("🗂️  Process Image(s) Or Directory with Custom Path or Default Folder Using SAHI")

    while True:
        print("1 - Process Single Image")
        print("2 - Process Default Directory")
        print("3 - Input Custom Directory Path")
        choice = input("Enter 1, 2, or 3: ").strip()

        if choice == '1':
            # Handle the single image case
            single_image_path = select_full_path_image()  # Method to select a single image path
            print(f"Processing single image: {single_image_path}")
            return choice, single_image_path

        elif choice == '2':
            # Use the provided default test images directory
            print(f"Processing images in default directory: {default_test_images_dir}")
            return choice, default_test_images_dir

        elif choice == '3':
            # Ask the user to input a custom directory
            while True:
                custom_dir = input("Enter the full path to the custom image directory: ").strip()
                if os.path.isdir(custom_dir):
                    print(f"Processing images in custom directory: {custom_dir}")
                    return choice, custom_dir
                else:
                    print(f"Invalid directory: {custom_dir}. Please enter a valid directory.")
        else:
            print("❌ Invalid input. Please enter 1, 2, or 3.")
            print_separator('=')


def select_slice_size(img_size):
    """
    Utility function to allow the user to select the slice size.
    Press Enter to use the default image size or type 1 to input a custom slice size.
    """
    print_separator('=')
    print(f"📏  Select slice size for prediction (Default: {img_size}x{img_size}):")
    print("Press Enter to use the default slice size.")
    print("Or type '1' to input a custom slice size.")
    print_separator()

    # Prompt the user for their choice
    choice = input("Enter '1' for custom size or press Enter to use the default: ").strip()

    # If the user presses Enter, return the default img_size
    if choice == '':
        print(f"✅ Using default slice size: {img_size}x{img_size}")
        return img_size

    # If the user chooses 1, prompt for a custom size
    elif choice == '1':
        while True:
            try:
                custom_slice_size = int(input("Enter custom slice size (must be a positive integer): ").strip())
                if custom_slice_size > 0:
                    print(f"✅ Using custom slice size: {custom_slice_size}x{custom_slice_size}")
                    return custom_slice_size
                else:
                    print("❌ Invalid input. Slice size must be a positive integer.")
            except ValueError:
                print("❌ Invalid input. Please enter a valid integer.")
    else:
        print("❌ Invalid input. Please enter '1' or press Enter to use the default.")
        return select_slice_size(img_size)


def select_overlap_ratio():
    """
    Utility function to allow the user to select the overlap ratio for image slicing.
    Press Enter to use the default overlap ratio (0.2) or type 1 to input a custom overlap ratio.
    """
    print_separator('=')
    print("🔄  Select overlap ratio for image slicing (Default: 0.2):")
    print("Press Enter to use the default overlap ratio.")
    print("Or type '1' to input a custom overlap ratio (value between 0.0 and 1.0).")
    print_separator()

    # Prompt the user for their choice
    choice = input("Enter '1' for custom overlap ratio or press Enter to use the default: ").strip()

    # If the user presses Enter, return the default overlap ratio (0.2)
    if choice == '':
        print("✅ Using default overlap ratio: 0.2")
        return 0.2

    # If the user chooses 1, prompt for a custom overlap ratio
    elif choice == '1':
        while True:
            try:
                custom_overlap_ratio = float(input("Enter custom overlap ratio (value between 0.0 and 1.0): ").strip())
                if 0.0 <= custom_overlap_ratio <= 1.0:
                    print(f"✅ Using custom overlap ratio: {custom_overlap_ratio}")
                    return custom_overlap_ratio
                else:
                    print("❌ Invalid input. Overlap ratio must be between 0.0 and 1.0.")
            except ValueError:
                print("❌ Invalid input. Please enter a valid numerical value.")
    else:
        print("❌ Invalid input. Please enter '1' or press Enter to use the default.")
        return select_overlap_ratio()


# Utility function to ask if slices should be generated
def select_generate_slices():
    """
    Utility function to ask if slice images should be generated during SAHI inference.
    Returns:
        bool: True if slices should be generated, False if not.
    """
    while True:
        print_separator('=')
        print("Do you want to generate slice images for each region?")
        print_separator()
        print("1 - Yes, generate slices.")
        print("2 - No, skip slice generation.")
        print_separator()
        
        choice = input("Enter 1 or 2: ").strip()
        if choice == '1':
            return True
        elif choice == '2':
            return False
        else:
            print("❌ Invalid input. Please enter 1 for Yes or 2 for No.")


def get_directory_input_images(default_dir, description="test images"):
    """
    This function prompts the user to either use a default directory, input a custom directory, or directly input a directory path.

    Args:
        default_dir (str): The default directory to use if the user presses Enter.
        description (str): A description of the directory (e.g., "test images", "output files").

    Returns:
        str: The selected directory path.
    """
    while True:
        print(f"Press Enter to use the default {description} directory.")
        custom_dir_input = input(f"Or enter the full path to a custom {description} directory: ").strip()

        if custom_dir_input == '':  # User pressed Enter, use the default directory
            return default_dir
        elif os.path.isdir(custom_dir_input):  # User provided a valid custom directory
            return custom_dir_input
        else:
            print(f"Invalid directory: {custom_dir_input}. Please enter a valid directory.")



def select_clearml_task_or_project():
    """
    Prompt the user to choose whether they want to fetch data by Task ID or Project ID.
    Returns:
        str: 'task' if Task ID is selected, 'project' if Project ID is selected.
    """
    while True:
        print("Do you want to fetch data by Task ID or Project ID?")
        print("1 - Task ID")
        print("2 - Project ID")
        choice = input("Enter 1 for Task ID or 2 for Project ID: ").strip()
        if choice == '1':
            return 'task'
        elif choice == '2':
            return 'project'
        else:
            print("❌ Invalid input. Please enter 1 or 2.")



def select_validation_mode():
    """
    Utility function to select the validation mode: single image, dataset, or cross-validation with different training image size.
    """
    while True:
        print_separator('=')
        print("Select validation mode:")
        print_separator()
        print("1 - Validate Single Image")
        print("2 - Validate Dataset (from default folder)")
        print("3 - Validate using different training image size")  # New option for cross-validation
        print_separator()
        mode = input("Enter 1, 2, or 3: ").strip()
        if mode in ['1', '2', '3']:
            return mode
        print("❌ Invalid input. Please enter 1, 2, or 3.")
        print_separator()




def select_single_image_for_validation():
    """
    Prompts the user to enter the full path for the single image validation.
    Returns the selected image path.
    """
    return input("Enter full path to the image to validate: ")


def select_batch_size():
    """
    Utility function to prompt the user to enter a batch size for validation.
    """
    while True:
        print_separator('=')
        print("Set the batch size for validation:")
        print_separator()
        batch_size = input("Enter a positive integer for batch size: ").strip()
        
        # Check if input is a valid integer
        if batch_size.isdigit() and int(batch_size) > 0:
            return int(batch_size)
        
        # Error message for invalid input
        print("❌ Invalid input. Please enter a positive integer.")
        print_separator()

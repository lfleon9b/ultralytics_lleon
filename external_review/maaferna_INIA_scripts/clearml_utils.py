
import os
import logging
import json
import re
from clearml import Task
from config import (
    CLEARML_API_HOST, CLEARML_WEB_HOST, CLEARML_FILES_HOST,
    CLEARML_KEY_ORIGINAL, CLEARML_SECRET_ORIGINAL,
    CLEARML_KEY_PERSONAL, CLEARML_SECRET_PERSONAL
)

def setup_credentials_clearml_task(account_type):
    """Setup ClearML credentials and initialize task based on account_type."""
    
    if account_type == '0':
        api_key = CLEARML_KEY_PERSONAL
        api_secret = CLEARML_SECRET_PERSONAL
        logging.info("Using personal ClearML account.")
    else:
        api_key = CLEARML_KEY_ORIGINAL
        api_secret = CLEARML_SECRET_ORIGINAL
        logging.info("Using original ClearML account.")
    
    if not api_key or not api_secret:
        raise ValueError("ClearML credentials are not set in environment variables.")
    
    # Set credentials for ClearML
    task = Task.set_credentials(
        api_host=CLEARML_API_HOST,
        web_host=CLEARML_WEB_HOST,
        files_host=CLEARML_FILES_HOST,
        key=api_key,
        secret=api_secret
    )
 
    return task

def start_clearml_task(name, description, *, project_name="YOLOv8 Research", account_type=None):
    """
    Initializes a new ClearML task with a specified project name.

    Parameters:
        name (str): The task name.
        description (str): The task description.
        project_name (str): The ClearML project to associate with this task.
        account_type (str, optional): Account type for ClearML if needed.

    Returns:
        Task: The initialized ClearML task.
    """
    logging.info("ClearML credentials set successfully")

    # Initialize a new task in ClearML with a dynamic project name
    task = Task.init(
        project_name=project_name,
        task_name=name,
        output_uri=True
    )
    logging.info(f"ClearML task created in project '{project_name}': {task.name}")
    
    return task

def log_clearml_result(task, result):
    """Log the results to ClearML and close the task."""
    task.get_logger().report_text('Results', result)
    task.close()
    task.flush()
    logging.info("ClearML task closed after logging results.")


def fetch_and_store_clearml_data(task_id, project_name, project_root_dir, account_name, is_project=False):
    """
    Fetch ClearML experiment data for a task and store it in the appropriate folder structure.
    Args:
        task_id (str): The ID of the ClearML task to fetch data from.
        project_name (str): The name of the project the task belongs to.
        project_root_dir (str): The root directory of the project.
        account_name (str): The ClearML account name (personal or original).
        is_project (bool): Flag to determine whether this is a project or a single task.
    """
    try:
        # Get the task using its ID
        task = Task.get_task(task_id=task_id)
        
        if not task:
            raise ValueError(f"Task with ID {task_id} not found.")
        
        # Determine the appropriate directory structure
        if is_project:
            task_dir = os.path.join(project_root_dir, 'clear_ml_data', account_name, 'projects', project_name, task_id)
        else:
            task_dir = os.path.join(project_root_dir, 'clear_ml_data', account_name, 'task', project_name, task_id)

        os.makedirs(task_dir, exist_ok=True)

        # Fetch and categorize logs
        logs = task.get_reported_console_output()
        if logs:
            logs_file = os.path.join(task_dir, f"{task_id}_logs.txt")
            metrics_file = os.path.join(task_dir, f"{task_id}_extracted_metrics.txt")
            with open(logs_file, 'w', encoding='utf-8') as log_f, open(metrics_file, 'w', encoding='utf-8') as metrics_f:
                for log_entry in logs:
                    clean_log = ''.join(char for char in log_entry if char.isprintable())
                    log_f.write(clean_log + '\n')
                    
                    # Extract metrics-like entries
                    if re.search(r'mAP@|Precision|Recall|F1-score', clean_log):
                        metrics_f.write(clean_log + '\n')
            
            logging.info(f"Logs saved to {logs_file}")
            logging.info(f"Extracted metrics saved to {metrics_file}")

        # Attempt to fetch scalar metrics
        scalars = task.get_last_scalar_metrics()
        if scalars:
            scalars_file = os.path.join(task_dir, f"{task_id}_scalars.json")
            with open(scalars_file, 'w') as f:
                json.dump(scalars, f, indent=4)
            logging.info(f"Scalar metrics saved to {scalars_file}")
        else:
            logging.warning(f"No scalar metrics found for task {task_id}")

        # Fetch and download artifacts
        artifacts = task.artifacts
        if artifacts:
            artifacts_dir = os.path.join(task_dir, 'artifacts')
            os.makedirs(artifacts_dir, exist_ok=True)
            download_experiment_artifacts(artifacts, artifacts_dir)
            logging.info(f"Artifacts downloaded to {artifacts_dir}")
        else:
            logging.info(f"No artifacts found for task {task_id}")

        print(f"Data for Task ID {task_id} has been saved in {task_dir}")

    except Exception as e:
        logging.error(f"An error occurred while fetching ClearML data for task {task_id}: {str(e)}")




def fetch_and_store_clearml_data_for_project(project_name, project_root_dir, account_name):
    """
    Fetch and store ClearML data for all completed tasks with meaningful data under a project and store it in 'clear_ml_data/projects/project_name/task_id' folder.
    Args:
        project_name (str): The name of the ClearML project.
        project_root_dir (str): The root directory where 'clear_ml_data/projects/project_name/task_id' will be created.
        account_name (str): The ClearML account name (personal or original) for organizing output folders.
    """
    try:
        # Use task_filter to filter for completed tasks in the project
        task_filter = {'status': ['completed']}
        completed_tasks = Task.get_tasks(
            project_name=project_name,
            task_filter=task_filter
        )
        
        if not completed_tasks:
            print(f"No completed tasks found for project: {project_name}")
            return

        print(f"Found {len(completed_tasks)} completed tasks for project: {project_name}")

        # Create the directory to store project tasks
        project_dir = os.path.join(project_root_dir, 'clear_ml_data', account_name, 'projects', project_name)
        os.makedirs(project_dir, exist_ok=True)

        # Fetch and store data for each completed task in the project directory
        for task in completed_tasks:
            task_id = task.id
            print(f"Checking data for Task ID: {task_id}")
            
            # Check for meaningful data
            has_logs = bool(task.get_reported_console_output())
            has_artifacts = bool(task.artifacts)
            
            if has_logs or has_artifacts:
                print(f"Fetching data for Task ID: {task_id}")
                fetch_and_store_clearml_data(task_id, project_name, project_root_dir, account_name, is_project=True)
            else:
                print(f"Task ID: {task_id} has no meaningful data and will be skipped.")

    except Exception as e:
        logging.error(f"An error occurred while fetching tasks for project: {str(e)}")


def download_experiment_artifacts(artifacts, download_dir):
    """
    Download experiment artifacts to the local machine.
    Args:
        artifacts (dict): Dictionary of artifacts retrieved from ClearML.
        download_dir (str): Directory to store downloaded artifacts.
    """
    if not artifacts:
        logging.info("No artifacts available for download.")
        return

    for artifact_name, artifact in artifacts.items():
        local_path = artifact.get_local_copy()  # Correct usage without 'target_path'
        if local_path:
            logging.info(f"Downloaded artifact: {artifact_name} to {local_path}")
        else:
            logging.error(f"Failed to download artifact: {artifact_name}")

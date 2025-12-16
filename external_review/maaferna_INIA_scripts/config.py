import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

CLEARML_API_HOST = os.getenv("CLEARML_API_HOST", "https://api.clear.ml")
CLEARML_WEB_HOST = os.getenv("CLEARML_WEB_HOST", "https://app.clear.ml")
CLEARML_FILES_HOST = os.getenv("CLEARML_FILES_HOST", "https://files.clear.ml")
CLEARML_KEY_ORIGINAL = os.getenv("CLEARML_KEY_ORIGINAL")
CLEARML_SECRET_ORIGINAL = os.getenv("CLEARML_SECRET_ORIGINAL")
CLEARML_KEY_PERSONAL = os.getenv("CLEARML_KEY_PERSONAL")
CLEARML_SECRET_PERSONAL = os.getenv("CLEARML_SECRET_PERSONAL")

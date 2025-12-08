import logging
import datetime
import os
from pathlib import Path

# Get project folder dir
current_file = Path(__file__).resolve()
project_dir = current_file.parent.parent

# Create the ./logs folder if it doesn't exist
log_dir = project_dir / "logs"
os.makedirs(log_dir, exist_ok=True)

# Generate a timestamp for the log filename
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

chatflow_log_filename = os.path.join(log_dir, f"chatflow_{timestamp}.log")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(module)s - %(levelname)s: %(message)s',
    handlers=[
        logging.FileHandler(chatflow_log_filename, encoding='utf-8'),
        logging.StreamHandler()
    ],
    force=True
)
logger_chatflow = logging.getLogger(__name__)
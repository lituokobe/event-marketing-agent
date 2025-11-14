import logging
import datetime
import os
from config.paths import LOG_PATH

# Create the ./logs folder if it doesn't exist
log_dir = LOG_PATH
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
    ]
)
logger_chatflow = logging.getLogger(__name__)
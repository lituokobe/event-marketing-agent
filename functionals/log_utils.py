import logging
import datetime
import os
from data.paths import LOG_PATH

# Create log folder if it doesn't exist
os.makedirs(LOG_PATH, exist_ok=True)

# Generate a timestamp for the log filename
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

chatflow_log_filename = os.path.join(LOG_PATH, f"chatflow_{timestamp}.log")

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
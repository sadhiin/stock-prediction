import os
import sys
import yaml
import time
import logging

# Set up logging
logging_str = "[%(asctime)s: %(levelname)s: %(module)s]: %(message)s"
logging_dir = 'logs'
logging_file = os.path.join(logging_dir, 'running_logs.log')
os.makedirs(logging_dir, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format=logging_str,
    handlers=[
        logging.FileHandler(logging_file),
        logging.StreamHandler(sys.stdout)]
)
# Create logger
logger = logging.getLogger('src')

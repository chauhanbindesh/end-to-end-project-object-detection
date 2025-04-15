import logging
import os
from datetime import datetime
from from_root import from_root
import os

def from_root(*paths):
    root_dir = os.path.dirname(os.path.abspath(__file__))
    while not os.path.exists(os.path.join(root_dir, 'ObjectDetection')) and root_dir != '/':
        root_dir = os.path.dirname(root_dir)
    return os.path.join(root_dir, *paths)
log_file = from_root("logs", "training.log")


LOG_FILE = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"


log_path = os.path.join(from_root(), 'log', LOG_FILE)

os.makedirs(log_path, exist_ok=True)

lOG_FILE_PATH = os.path.join(log_path, LOG_FILE)

logging.basicConfig(
    filename=lOG_FILE_PATH,
    format= "[ %(asctime)s ] %(name)s - %(levelname)s - %(message)s",
    level= logging.INFO
)
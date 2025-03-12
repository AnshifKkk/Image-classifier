# Initialize the app package
import os
import sys
import logging
import logging.handlers
from datetime import datetime

def setup_logging(log_level=logging.INFO, log_dir="logs"):
    """
    Configure application logging
    
    Args:
        log_level: The logging level to use
        log_dir: Directory to store log files
    """
    # Create logs directory if it doesn't exist
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    # Generate log filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d")
    log_file = os.path.join(log_dir, f"image_classifier_{timestamp}.log")
    
    # Configure root logger
    logger = logging.getLogger()
    logger.setLevel(log_level)
    
    # Clear existing handlers to avoid duplicates
    if logger.handlers:
        logger.handlers.clear()
    
    # Create formatters
    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - [%(name)s:%(filename)s:%(lineno)d] - %(message)s'
    )
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler with rotation
    file_handler = logging.handlers.RotatingFileHandler(
        log_file,
        maxBytes=10 * 1024 * 1024,  # 10 MB
        backupCount=5
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    # Set urllib3 and PIL log levels higher to reduce noise
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('PIL').setLevel(logging.WARNING)
    
    # Set TensorFlow log level
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
    
    logging.info(f"Logging initialized. Log file: {log_file}")
    logging.info(f"Python version: {sys.version}")
    logging.info(f"Running in environment: {os.getenv('ENVIRONMENT', 'development')}")
    
    return logger

# Set up application logging
log_level = os.getenv('LOG_LEVEL', 'INFO').upper()
numeric_level = getattr(logging, log_level, logging.INFO)
setup_logging(log_level=numeric_level)

# Log startup
logging.info("Initializing image-classifier-api application")

import numpy as np
from PIL import Image
import io
import logging
import time
from functools import wraps

logger = logging.getLogger(__name__)

def timing_decorator(func):
    """Decorator to log the execution time of functions"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            execution_time = time.time() - start_time
            logger.debug(f"{func.__name__} executed in {execution_time:.4f} seconds")
            return result
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"{func.__name__} failed after {execution_time:.4f} seconds with error: {str(e)}")
            raise
    return wrapper

class ImageProcessingError(Exception):
    """Custom exception for image processing errors"""
    pass

@timing_decorator
def preprocess_image(image, target_size=(150, 150)):
    """
    Preprocess an image for prediction
    
    Args:
        image: PIL Image object
        target_size: Tuple of (height, width) to resize the image to
        
    Returns:
        Preprocessed image as numpy array ready for model input
        
    Raises:
        ImageProcessingError: If image processing fails
    """
    try:
        # Validate image
        if not isinstance(image, Image.Image):
            raise ImageProcessingError("Invalid image object")
        
        # Log original image details
        logger.debug(f"Original image: size={image.size}, mode={image.mode}")
        
        # Resize the image
        if image.size != target_size:
            image = image.resize(target_size)
            logger.debug(f"Resized image to {target_size}")
        
        # Convert to RGB if not already
        if image.mode != "RGB":
            image = image.convert("RGB")
            logger.debug(f"Converted image to RGB mode")
        
        # Convert to numpy array and normalize
        img_array = np.array(image).astype(np.float32)
        
        # Check for corrupted image data
        if img_array.size == 0 or np.isnan(img_array).any():
            raise ImageProcessingError("Corrupted image data detected")
        
        # Scale pixel values to [0, 1]
        img_array = img_array / 255.0
        
        logger.debug(f"Preprocessed image shape: {img_array.shape}")
        return img_array
    
    except ImageProcessingError as e:
        # Re-raise custom exceptions
        logger.error(f"Image processing error: {str(e)}")
        raise
    except Exception as e:
        # Convert other exceptions to ImageProcessingError
        error_msg = f"Error preprocessing image: {str(e)}"
        logger.error(error_msg)
        raise ImageProcessingError(error_msg) from e

@timing_decorator
def load_image_from_bytes(image_bytes, target_size=(150, 150)):
    """
    Load an image from bytes and preprocess it
    
    Args:
        image_bytes: Raw image bytes
        target_size: Target size to resize to
        
    Returns:
        Preprocessed image array
        
    Raises:
        ImageProcessingError: If loading or preprocessing fails
    """
    try:
        # Validate input
        if not image_bytes:
            raise ImageProcessingError("Empty image bytes provided")
        
        # Log incoming image size
        logger.debug(f"Loading image from {len(image_bytes)} bytes")
        
        # Open image from bytes
        image = Image.open(io.BytesIO(image_bytes))
        
        # Preprocess the image
        return preprocess_image(image, target_size)
    
    except ImageProcessingError:
        # Re-raise custom exceptions
        raise
    except Exception as e:
        # Convert other exceptions to ImageProcessingError
        error_msg = f"Error loading image from bytes: {str(e)}"
        logger.error(error_msg)
        raise ImageProcessingError(error_msg) from e

def validate_image_format(filename):
    """
    Validate if the image format is supported
    
    Args:
        filename: Name of the uploaded file
        
    Returns:
        bool: True if supported, False otherwise
    """
    if not filename:
        return False
        
    allowed_extensions = {'.jpg', '.jpeg', '.png'}
    file_extension = os.path.splitext(filename.lower())[1]
    
    return file_extension in allowed_extensions

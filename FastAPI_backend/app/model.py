import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import logging
import time
from functools import wraps

logger = logging.getLogger(__name__)

class ModelError(Exception):
    """Custom exception for model-related errors"""
    pass

def model_timing_decorator(func):
    """Decorator to log execution time for model functions"""
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

class ImageClassifier:
    def __init__(self, model_path="model/saved_models/mobilenet_model.h5"):
        """
        Initialize the image classifier
        
        Args:
            model_path: Path to the trained model file
            
        Raises:
            ModelError: If model initialization fails
        """
        self.model_path = model_path
        self.model = None
        self.class_names = ['buildings', 'forest', 'glacier', 'mountain', 'sea', 'street']
        self.is_loaded = False
        self.img_size = 150
        self.model_memory_usage = None
        
        # Disable eager execution warnings from TensorFlow
        tf.get_logger().setLevel('ERROR')
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
        
        # Load the model
        try:
            self.load_model()
        except Exception as e:
            logger.error(f"Failed to initialize model: {str(e)}")
            # Allow initialization even if model fails to load
            # Service can be started and model loaded later
    
    @model_timing_decorator
    def load_model(self):
        """
        Load the trained model from disk
        
        Raises:
            ModelError: If model loading fails
        """
        try:
            # Check if model file exists
            if not os.path.exists(self.model_path):
                error_msg = f"Model file not found at {self.model_path}"
                logger.error(error_msg)
                raise ModelError(error_msg)
            
            logger.info(f"Loading model from {self.model_path}")
            
            # Load model with error handling
            try:
                # Configure memory growth to avoid OOM errors
                physical_devices = tf.config.list_physical_devices('GPU')
                if physical_devices:
                    for device in physical_devices:
                        tf.config.experimental.set_memory_growth(device, True)
                        logger.info(f"Configured memory growth for GPU: {device}")
                
                # Load the model
                self.model = load_model(self.model_path)
                
                # Warm up the model with a dummy prediction
                dummy_input = np.zeros((1, self.img_size, self.img_size, 3), dtype=np.float32)
                self.model.predict(dummy_input)
                
                # Estimate model memory usage
                self.model_memory_usage = self._estimate_model_size()
                
                self.is_loaded = True
                logger.info(f"Model loaded successfully. Memory usage: ~{self.model_memory_usage:.2f} MB")
                
            except tf.errors.ResourceExhaustedError as e:
                raise ModelError(f"Out of memory error loading model: {str(e)}")
            except Exception as e:
                raise ModelError(f"Error loading TensorFlow model: {str(e)}")
            
        except ModelError:
            # Re-raise custom exceptions
            self.is_loaded = False
            raise
        except Exception as e:
            # Convert other exceptions to ModelError
            error_msg = f"Unexpected error loading model: {str(e)}"
            logger.error(error_msg)
            self.is_loaded = False
            raise ModelError(error_msg) from e
    
    def _estimate_model_size(self):
        """Estimate the model size in memory (MB)"""
        if not self.model:
            return 0
            
        # Calculate total parameters
        total_params = self.model.count_params()
        
        # Rough estimate: 4 bytes per parameter (float32)
        size_bytes = total_params * 4
        size_mb = size_bytes / (1024 * 1024)
        
        return size_mb
    
    @model_timing_decorator
    def predict(self, preprocessed_image):
        """
        Make a prediction on a preprocessed image
        
        Args:
            preprocessed_image: A preprocessed image ready for model input
            
        Returns:
            Dictionary with prediction results
            
        Raises:
            ModelError: If prediction fails
        """
        try:
            # Check if model is loaded
            if not self.is_loaded or self.model is None:
                try:
                    logger.warning("Model not loaded. Attempting to load it now.")
                    self.load_model()
                except Exception as e:
                    raise ModelError(f"Model is not loaded and couldn't be loaded: {str(e)}")
            
            # Validate input shape
            if preprocessed_image is None:
                raise ModelError("Preprocessed image is None")
                
            # Add logging for image shape
            logger.debug(f"Input image shape: {preprocessed_image.shape}")
            
            # Ensure image has batch dimension
            if len(preprocessed_image.shape) == 3:
                preprocessed_image = np.expand_dims(preprocessed_image, axis=0)
                logger.debug(f"Added batch dimension. New shape: {preprocessed_image.shape}")
            
            # Check for invalid values
            if np.isnan(preprocessed_image).any() or np.isinf(preprocessed_image).any():
                raise ModelError("Input image contains NaN or Inf values")
            
            # Check value range
            if preprocessed_image.min() < 0 or preprocessed_image.max() > 1:
                logger.warning(f"Image values outside expected range [0,1]. Min: {preprocessed_image.min()}, Max: {preprocessed_image.max()}")
                # Clip values to ensure they're in range
                preprocessed_image = np.clip(preprocessed_image, 0, 1)
            
            # Make prediction with error handling
            try:
                logger.debug("Starting prediction")
                predictions = self.model.predict(preprocessed_image)
                logger.debug("Prediction completed successfully")
            except tf.errors.ResourceExhaustedError as e:
                raise ModelError(f"Out of memory error during prediction: {str(e)}")
            except Exception as e:
                raise ModelError(f"Error during model prediction: {str(e)}")
            
            # Process results
            if predictions is None or len(predictions) == 0:
                raise ModelError("Model returned empty predictions")
                
            # Get the predicted class index
            predicted_class_index = np.argmax(predictions[0])
            
            # Get the class name
            if predicted_class_index >= len(self.class_names):
                raise ModelError(f"Predicted class index {predicted_class_index} out of range")
                
            predicted_class = self.class_names[predicted_class_index]
            
            # Get the confidence score (probability)
            confidence = float(predictions[0][predicted_class_index])
            
            # Get top 3 predictions for more detailed response
            top_indices = np.argsort(predictions[0])[-3:][::-1]
            top_predictions = [
                {
                    "class": self.class_names[idx],
                    "confidence": float(predictions[0][idx])
                }
                for idx in top_indices
            ]
            
            result = {
                "predicted_class": predicted_class,
                "confidence": confidence,
                "top_predictions": top_predictions
            }
            
            logger.debug(f"Prediction result: {result}")
            return result
            
        except ModelError:
            # Re-raise custom exceptions
            raise
        except Exception as e:
            # Convert other exceptions to ModelError
            error_msg = f"Unexpected error during prediction: {str(e)}"
            logger.error(error_msg)
            raise ModelError(error_msg) from e

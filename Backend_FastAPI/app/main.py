from fastapi import FastAPI, File, UploadFile, Depends, HTTPException, status, Request
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import logging
import time
import os
import traceback
import uuid
from contextlib import asynccontextmanager

from app.auth import authenticate_user
from app.model import ImageClassifier
from app.utils import load_image_from_bytes

# Configure logger
logger = logging.getLogger(__name__)

# Model instance
classifier = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize model on startup
    global classifier
    try:
        logger.info("Loading model...")
        classifier = ImageClassifier()
        logger.info("Model loaded successfully")
    except Exception as e:
        logger.critical(f"Failed to load model: {str(e)}")
        logger.critical(traceback.format_exc())
    yield
    # Cleanup on shutdown
    logger.info("Shutting down application")

# Initialize FastAPI app
app = FastAPI(
    title="Image Classifier API",
    description="API for classifying images",
    version="1.0.0",
    lifespan=lifespan
)

# CORS settings
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Setup Basic Auth
security = HTTPBasic()

# Middleware for request logging and timing
@app.middleware("http")
async def log_requests(request: Request, call_next):
    request_id = str(uuid.uuid4())
    request.state.request_id = request_id
    
    logger.info(f"Request {request_id} started: {request.method} {request.url.path}")
    
    start_time = time.time()
    
    try:
        response = await call_next(request)
        process_time = time.time() - start_time
        response.headers["X-Process-Time"] = str(process_time)
        logger.info(f"Request {request_id} completed: {response.status_code} in {process_time:.4f}s")
        return response
    except Exception as e:
        process_time = time.time() - start_time
        logger.error(f"Request {request_id} failed after {process_time:.4f}s: {str(e)}")
        logger.error(traceback.format_exc())
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"detail": "Internal server error", "request_id": request_id}
        )

# Error handler for HTTP exceptions
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    logger.warning(f"HTTPException: {exc.status_code} - {exc.detail}")
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail, "request_id": getattr(request.state, "request_id", "unknown")}
    )

# Error handler for unexpected exceptions
@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {str(exc)}")
    logger.error(traceback.format_exc())
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "detail": "An unexpected error occurred",
            "request_id": getattr(request.state, "request_id", "unknown")
        }
    )

@app.get("/")
async def root():
    """Health check endpoint"""
    return {"status": "online", "service": "image-classifier-api"}

@app.get("/health")
async def health_check():
    """Detailed health check endpoint"""
    model_status = "ready" if classifier and classifier.is_loaded else "not_loaded"
    return {
        "status": "healthy",
        "model_status": model_status,
        "environment": os.getenv("ENVIRONMENT", "development")
    }

@app.post("/predict")
async def predict_image(
    file: UploadFile = File(...),
    credentials: HTTPBasicCredentials = Depends(security)
):
    """
    Predict the class of an uploaded image
    
    The image should be uploaded as a multipart/form-data file
    """
    request_id = getattr(Request.state, "request_id", str(uuid.uuid4()))
    
    # Authenticate user
    authenticate_user(credentials)
    
    # Check if model is loaded
    if not classifier or not classifier.is_loaded:
        logger.error(f"Request {request_id} failed: Model not loaded")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded"
        )
    
    try:
        # Read image bytes
        image_bytes = await file.read()
        
        if not image_bytes:
            logger.warning(f"Request {request_id} failed: Empty file uploaded")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Empty file uploaded"
            )
        
        # Process image
        logger.info(f"Processing image: {file.filename}, size: {len(image_bytes)} bytes")
        preprocessed_image = load_image_from_bytes(image_bytes)
        
        # Make prediction
        logger.info(f"Making prediction for image: {file.filename}")
        prediction = classifier.predict(preprocessed_image)
        
        logger.info(f"Prediction successful for {file.filename}: {prediction['predicted_class']} ({prediction['confidence']:.4f})")
        return prediction
        
    except HTTPException:
        # Re-raise HTTP exceptions to be handled by the exception handler
        raise
    except ValueError as e:
        logger.error(f"Value error during prediction: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid input: {str(e)}"
        )
    except Exception as e:
        logger.error(f"Error processing image {file.filename if file else 'unknown'}: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error processing image"
        )

import os
from fastapi import HTTPException, status
from fastapi.security import HTTPBasicCredentials
from dotenv import load_dotenv
import secrets
import logging

# Load environment variables from .env file
load_dotenv()

# Get API credentials from environment
API_USERNAME = os.getenv("API_USERNAME", "admin")
API_PASSWORD = os.getenv("API_PASSWORD", "imageclassifierproject")

def authenticate_user(credentials: HTTPBasicCredentials):
    """
    Authenticate a user using HTTP Basic Authentication
    
    Args:
        credentials: HTTP Basic Auth credentials
        
    Raises:
        HTTPException: If authentication fails
    """
    correct_username = secrets.compare_digest(credentials.username, API_USERNAME)
    correct_password = secrets.compare_digest(credentials.password, API_PASSWORD)
    
    if not (correct_username and correct_password):
        logging.warning(f"Failed authentication attempt for user: {credentials.username}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid credentials",
            headers={"WWW-Authenticate": "Basic"},
        )
    
    logging.info(f"Successful authentication for user: {credentials.username}")
    return True

# Image Classifier Project

This repository contains an image classification application with a Streamlit frontend and FastAPI backend. The frontend is deployed on Streamlit Cloud, and the backend is hosted on Render.

### Backend Setup

1. Navigate to the FastAPI backend directory:
   ```
   cd FastAPI_backend
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Run the backend server:
   ```
   uvicorn app.main:app --reload
   ```
   The API will be available at http://localhost:8000

### Frontend Setup

1. Navigate to the Streamlit frontend directory:
   ```
   cd Streamlit_frontend
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Run the Streamlit application:
   ```
   python -m streamlit run app.py
   ```
   The application will be available at http://localhost:8501

## Docker Setup

Alternatively, you can run both the frontend and backend using Docker:

### Backend Docker Setup

1. Navigate to the FastAPI backend directory:
   ```
   cd FastAPI_backend
   ```

2. Build the Docker image:
   ```
   docker build -t image-classifier-backend .
   ```

3. Run the Docker container:
   ```
   docker run -p 8000:8000 image-classifier-backend
   ```

### Frontend Docker Setup

1. Navigate to the Streamlit frontend directory:
   ```
   cd streamlit_frontend
   ```

2. Build the Docker image:
   ```
   docker build -t image-classifier-frontend .
   ```

3. Run the Docker container:
   ```
   docker run -p 8501:8501 image-classifier-frontend
   ```

## Deployment

### Backend Deployment (Render)

The backend API is deployed on Render. The API endpoint is available at:
[https://image-classifier-rngj.onrender.com]

### Frontend Deployment (Streamlit Cloud)

The frontend application is deployed on Streamlit Cloud. The application is available at:
[https://image-classifier-tnb86bthtu2vwrme6zdlq5.streamlit.app/]

## API Documentation

After running the backend, API documentation is available at:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## Features

- Check out project report

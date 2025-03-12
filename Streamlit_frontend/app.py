import streamlit as st
import requests
import base64
from PIL import Image
import io
import time
import os
from dotenv import load_dotenv
import logging
import pandas as pd
import plotly.express as px
import json

# Load environment variables
load_dotenv()

# Configure basic logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)

# API Configuration
API_URL = os.getenv("API_URL", "http://localhost:8000")
API_USERNAME = os.getenv("API_USERNAME", "admin")
API_PASSWORD = os.getenv("API_PASSWORD", "password")

# Set page configuration
st.set_page_config(
    page_title="Landscape Classifier",
    page_icon="🏞️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap');

* {
    font-family: 'Poppins', sans-serif;
}

.main-header {
    font-size: 2.8rem;
    font-weight: 700;
    background: linear-gradient(90deg, #4B8BBE 0%, #306998 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    text-align: center;
    margin-bottom: 1.5rem;
    padding-top: 1rem;
}

.sub-header {
    font-size: 1.8rem;
    font-weight: 600;
    color: #306998;
    margin-bottom: 1.2rem;
}

.result-text {
    font-size: 1.4rem;
    font-weight: 700;
    color: #1F4E79;
    margin: 1rem 0;
}

.result-card {
    background-color: #f8f9fa;
    border-radius: 12px;
    padding: 1.5rem;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08);
    margin-bottom: 1.5rem;
}

.stButton > button {
    background-color: #4B8BBE;
    color: white;
    font-weight: 600;
    border-radius: 8px;
    padding: 0.5rem 2rem;
    border: none;
    transition: all 0.3s ease;
}

.stButton > button:hover {
    background-color: #306998;
    box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
    transform: translateY(-2px);
}

.stAlert > div {
    padding: 0.75rem 1rem;
    border-radius: 8px;
    font-weight: 500;
}

.category-item {
    background-color: #e6f2ff;
    border-radius: 8px;
    padding: 0.5rem 1rem;
    margin: 0.5rem 0;
    font-weight: 500;
    color: #1F4E79;
    display: inline-block;
    margin-right: 0.5rem;
}

.image-container {
    border-radius: 12px;
    overflow: hidden;
    box-shadow: 0 6px 16px rgba(0, 0, 0, 0.1);
    margin-bottom: 1.5rem;
}

.stExpander {
    border-radius: 8px;
    overflow: hidden;
}

.upload-section {
    background-color: #f8f9fa;
    border-radius: 12px;
    padding: 2rem;
    border: 2px dashed #4B8BBE;
    margin-bottom: 1.5rem;
    text-align: center;
}

.sidebar-section {
    background-color: #f0f5fa;
    border-radius: 8px;
    padding: 1rem;
    margin-bottom: 1rem;
}

/* Status indicator styles */
.status-indicator {
    display: inline-block;
    width: 10px;
    height: 10px;
    border-radius: 50%;
    margin-right: 8px;
}

.status-online {
    background-color: #28a745;
}

.status-offline {
    background-color: #dc3545;
}

/* Footer styles */
.footer {
    text-align: center;
    padding: 1.5rem 0;
    color: #6c757d;
    font-size: 0.9rem;
    border-top: 1px solid #e9ecef;
    margin-top: 2rem;
}

</style>
""", unsafe_allow_html=True)

@st.cache_data(ttl=300)
def check_api_health():
    """Check if the API is online"""
    try:
        response = requests.get(f"{API_URL}/health", timeout=5)
        return response.status_code == 200, response.json() if response.status_code == 200 else None
    except requests.exceptions.RequestException as e:
        logger.error(f"API health check failed: {str(e)}")
        return False, None

def classify_image(image_bytes):
    """Send image to API for classification"""
    try:
        files = {"file": image_bytes}
        response = requests.post(
            f"{API_URL}/predict",
            files=files,
            auth=(API_USERNAME, API_PASSWORD),
            timeout=30
        )
        
        if response.status_code == 200:
            return True, response.json()
        else:
            error_msg = f"API request failed with status {response.status_code}"
            try:
                error_detail = response.json().get("detail", "Unknown error")
                error_msg = f"{error_msg}: {error_detail}"
            except:
                pass
            return False, error_msg
    except requests.exceptions.RequestException as e:
        logger.error(f"API request failed: {str(e)}")
        return False, f"API request failed: {str(e)}"

def display_prediction_results(prediction):
    """Display the prediction results with enhanced visualizations"""
    st.markdown("<div class='result-card'>", unsafe_allow_html=True)
    st.markdown("<h2 class='sub-header'>Prediction Results</h2>", unsafe_allow_html=True)
    
    # Get the top prediction
    top_prediction = prediction['predicted_class']
    confidence = prediction['confidence']
    
    # Show main prediction with custom styling
    st.markdown(
        f"<p class='result-text'>🔍 This image looks like a <span style='color:#4B8BBE;'>{top_prediction.title()}</span> "
        f"with <span style='color:#4B8BBE;'>{confidence:.1%}</span> confidence</p>", 
        unsafe_allow_html=True
    )
    
    # Create DataFrame for visualization
    results_df = pd.DataFrame(prediction["top_predictions"])
    results_df["confidence_pct"] = results_df["confidence"] * 100
    results_df["class"] = results_df["class"].str.title()
    
    # Create enhanced horizontal bar chart using Plotly
    fig = px.bar(
        results_df,
        x="confidence_pct",
        y="class",
        orientation="h",
        labels={"confidence_pct": "Confidence (%)", "class": "Landscape Type"},
        color="confidence_pct",
        color_continuous_scale="Blues",
        range_color=[0, 100],
    )
    
    # Update layout for a more polished look
    fig.update_layout(
        xaxis_title="Confidence (%)",
        yaxis_title=None,
        yaxis=dict(
            autorange="reversed",
            showgrid=False
        ),
        height=350,
        margin=dict(l=0, r=0, t=10, b=10),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(family="Poppins, sans-serif", size=14),
        xaxis=dict(showgrid=True, gridcolor='#EEE'),
        coloraxis_showscale=False,
    )
    
    # Add bar annotations
    fig.update_traces(
        texttemplate='%{x:.1f}%',
        textposition='outside',
        hovertemplate='<b>%{y}</b><br>Confidence: %{x:.1f}%<extra></extra>',
        marker_line_color='#306998',
        marker_line_width=1,
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Display landscape description based on prediction
    landscape_descriptions = {
        "buildings": "Urban architecture with man-made structures.",
        "forest": "Dense collection of trees and woodland vegetation.",
        "glacier": "Slow-moving mass of ice formed by snow accumulation over time.",
        "mountain": "Large elevated landform with steep slopes rising prominently above surroundings.",
        "sea": "Large body of saltwater composing much of Earth's hydrosphere.",
        "street": "Public thoroughfare in a built environment, typically paved."
    }
    
    predicted_lower = top_prediction.lower()
    if predicted_lower in landscape_descriptions:
        st.info(f"**{top_prediction}**: {landscape_descriptions[predicted_lower]}")
    
    # Show raw JSON in a cleaner expander
    with st.expander("View Technical Details"):
        st.json(prediction)
    
    st.markdown("</div>", unsafe_allow_html=True)

def save_uploadedfile(uploaded_file):
    """Save uploaded file temporarily and return bytes"""
    bytes_data = uploaded_file.getvalue()
    return bytes_data

def display_categories():
    """Display available classification categories with styled badges"""
    categories = ['Buildings', 'Forest', 'Glacier', 'Mountain', 'Sea', 'Street']
    category_html = ""
    
    for category in categories:
        category_html += f"<div class='category-item'>{category}</div>"
    
    st.markdown(category_html, unsafe_allow_html=True)

def main():
    """Main Streamlit application with enhanced UI"""
    st.markdown("<h1 class='main-header'>Landscape Classifier</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align:center; margin-bottom:2rem;'>Identify different types of landscapes with AI-powered image recognition</p>", unsafe_allow_html=True)
    
    # Sidebar with enhanced styling
    with st.sidebar:
        st.markdown("<div class='sidebar-section'>", unsafe_allow_html=True)
        st.markdown("### 📖 About")
        st.markdown(
            "Landscape Classifier uses a MobileNet-based neural network to identify and classify "
            "landscapes and environments from your photos."
        )
        st.markdown("</div>", unsafe_allow_html=True)
        
        # API Status with visual indicator
        st.markdown("<div class='sidebar-section'>", unsafe_allow_html=True)
        st.markdown("### 🔌 System Status")
        
        # Check API health
        api_online, health_data = check_api_health()
        
        if api_online:
            st.markdown(
                "<p><span class='status-indicator status-online'></span> <b>API Status:</b> Online</p>",
                unsafe_allow_html=True
            )
            if health_data:
                st.markdown(f"<p><b>Model:</b> {health_data.get('model_status', 'Active')}</p>", unsafe_allow_html=True)
                st.markdown(f"<p><b>Environment:</b> {health_data.get('environment', 'Production')}</p>", unsafe_allow_html=True)
        else:
            st.markdown(
                "<p><span class='status-indicator status-offline'></span> <b>API Status:</b> Offline</p>",
                unsafe_allow_html=True
            )
            st.warning("Please check that the API server is running and correctly configured.")
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Categories with visual styling
        st.markdown("<div class='sidebar-section'>", unsafe_allow_html=True)
        st.markdown("### 🏷️ What Can It Identify?")
        st.markdown("This model can classify images into these categories:")
        display_categories()
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Usage tips
        st.markdown("<div class='sidebar-section'>", unsafe_allow_html=True)
        st.markdown("### 💡 Tips")
        st.markdown(
            "- Upload clear, well-lit images\n"
            "- Landscape orientation works best\n"
            "- Avoid heavily filtered photos\n"
            "- For best results, ensure the landscape is the main subject"
        )
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Main content
    col1, col2 = st.columns([1, 1], gap="large")
    
    with col1:
        st.markdown("<h2 class='sub-header'>Upload Your Image</h2>", unsafe_allow_html=True)
        
        # Enhanced upload section
        st.markdown("<div class='upload-section'>", unsafe_allow_html=True)
        uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])
        
        if not uploaded_file:
            st.markdown(
                "<p>📤 Drag and drop or click to upload a landscape photo</p>",
                unsafe_allow_html=True
            )
            st.markdown("<p style='font-size:0.9rem; color:#6c757d;'>Supported formats: JPG, JPEG, PNG</p>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
        
        if uploaded_file is not None:
            # Display uploaded image with enhanced styling
            st.markdown("<h3 class='sub-header'>Your Image</h3>", unsafe_allow_html=True)
            st.markdown("<div class='image-container'>", unsafe_allow_html=True)
            image = Image.open(uploaded_file)
            st.image(image, use_column_width=True)
            st.markdown("</div>", unsafe_allow_html=True)
            
            # Image details
            file_details = {
                "Filename": uploaded_file.name,
                "File size": f"{uploaded_file.size / 1024:.1f} KB",
                "Image dimensions": f"{image.width} × {image.height} px"
            }
            
            with st.expander("Image Details"):
                for key, value in file_details.items():
                    st.markdown(f"**{key}:** {value}")
            
            # Enhanced classification button
            if st.button("📊 Analyze Image", type="primary"):
                # Show spinner during classification
                with st.spinner("Analyzing your landscape..."):
                    # Convert to bytes for API request
                    image_bytes = save_uploadedfile(uploaded_file)
                    
                    # Classify image
                    success, result = classify_image(image_bytes)
                    
                    # Show results
                    with col2:
                        if success:
                            display_prediction_results(result)
                        else:
                            st.error(f"Classification failed: {result}")
    
    # Show empty placeholder or example in the second column if no image is uploaded
    with col2:
        if not uploaded_file:
            st.markdown("<h2 class='sub-header'>Results Preview</h2>", unsafe_allow_html=True)
            st.markdown("<div class='result-card' style='min-height:400px; display:flex; flex-direction:column; justify-content:center; align-items:center;'>", unsafe_allow_html=True)
            st.markdown("<p style='color:#6c757d; text-align:center;'>👈 Upload an image to see AI-powered landscape classification results here</p>", unsafe_allow_html=True)
            st.markdown("<p style='font-size:0.9rem; color:#6c757d; text-align:center;'>Try uploading images of natural landscapes, urban environments, or scenic views</p>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)
    
    # Enhanced footer
    st.markdown("<div class='footer'>", unsafe_allow_html=True)
    st.markdown("Powered by TensorFlow & FastAPI | Built with Streamlit | © 2025 Landscape Classifier", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
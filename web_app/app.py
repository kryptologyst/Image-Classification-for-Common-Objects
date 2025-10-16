"""
Streamlit web interface for Image Classification Project.
"""

import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import sys
import os

# Add src directory to path
sys.path.append(str(Path(__file__).parent / "src"))

from config import Config
from logger import setup_logging
from models import ModelFactory
from visualizer import Visualizer


# Page configuration
st.set_page_config(
    page_title="Image Classification Demo",
    page_icon="🖼️",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .prediction-result {
        font-size: 1.5rem;
        font-weight: bold;
        text-align: center;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .correct-prediction {
        background-color: #d4edda;
        color: #155724;
        border: 1px solid #c3e6cb;
    }
    .incorrect-prediction {
        background-color: #f8d7da;
        color: #721c24;
        border: 1px solid #f5c6cb;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_config():
    """Load configuration."""
    return Config("config/config.yaml")


@st.cache_resource
def load_model(model_type: str):
    """Load trained model."""
    config = load_config()
    
    # Try to load existing model first
    models_dir = Path(config.get('paths.models_dir', 'models'))
    model_files = list(models_dir.glob(f"{model_type}_model_final.h5"))
    
    if model_files:
        model = tf.keras.models.load_model(str(model_files[0]))
        st.success(f"Loaded existing {model_type} model from {model_files[0]}")
    else:
        # Create new model if no saved model exists
        st.warning(f"No saved {model_type} model found. Creating new model...")
        model = ModelFactory.create_model(model_type, config)
    
    return model


def preprocess_image(image: Image.Image) -> np.ndarray:
    """Preprocess uploaded image."""
    # Resize to model input size
    image = image.resize((32, 32))
    
    # Convert to RGB if needed
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Convert to numpy array and normalize
    image_array = np.array(image) / 255.0
    
    # Add batch dimension
    image_array = np.expand_dims(image_array, axis=0)
    
    return image_array


def main():
    """Main Streamlit app."""
    st.markdown('<h1 class="main-header">🖼️ Image Classification Demo</h1>', unsafe_allow_html=True)
    
    # Load configuration
    config = load_config()
    class_names = config.get_class_names()
    
    # Sidebar
    st.sidebar.title("Model Selection")
    model_type = st.sidebar.selectbox(
        "Choose Model Type",
        ["cnn", "resnet", "efficientnet"],
        help="Select the type of model to use for classification"
    )
    
    # Load model
    try:
        model = load_model(model_type)
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        st.info("Please train a model first using the training script.")
        return
    
    # Main content
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("📤 Upload Image")
        
        # File uploader
        uploaded_file = st.file_uploader(
            "Choose an image file",
            type=['png', 'jpg', 'jpeg'],
            help="Upload an image to classify. Supported formats: PNG, JPG, JPEG"
        )
        
        # Sample images
        st.subheader("🎯 Sample Images")
        sample_images = {
            "Airplane": "https://images.unsplash.com/photo-1436491865332-7a61a109cc05?w=150",
            "Car": "https://images.unsplash.com/photo-1552519507-da3b142c6e3d?w=150",
            "Bird": "https://images.unsplash.com/photo-1444464666168-49d633b86797?w=150",
            "Cat": "https://images.unsplash.com/photo-1514888286974-6c03e2ca1dba?w=150",
            "Dog": "https://images.unsplash.com/photo-1552053831-71594a27632d?w=150"
        }
        
        selected_sample = st.selectbox("Or try a sample image:", list(sample_images.keys()))
        if st.button("Load Sample Image"):
            st.image(sample_images[selected_sample], caption=f"Sample: {selected_sample}", width=200)
    
    with col2:
        st.header("🔍 Classification Results")
        
        if uploaded_file is not None:
            # Display uploaded image
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", width=300)
            
            # Preprocess and predict
            processed_image = preprocess_image(image)
            
            # Make prediction
            prediction = model.predict(processed_image, verbose=0)
            predicted_class_idx = np.argmax(prediction[0])
            predicted_class = class_names[predicted_class_idx]
            confidence = prediction[0][predicted_class_idx]
            
            # Display results
            st.markdown(f"""
            <div class="prediction-result">
                <strong>Predicted Class:</strong> {predicted_class}<br>
                <strong>Confidence:</strong> {confidence:.3f}
            </div>
            """, unsafe_allow_html=True)
            
            # Confidence bar
            st.subheader("Confidence Scores")
            confidence_data = {
                'Class': class_names,
                'Confidence': prediction[0]
            }
            
            fig = px.bar(
                confidence_data, 
                x='Confidence', 
                y='Class',
                orientation='h',
                title="Prediction Confidence by Class",
                color='Confidence',
                color_continuous_scale='Blues'
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
            
            # Top 3 predictions
            st.subheader("Top 3 Predictions")
            top_3_indices = np.argsort(prediction[0])[-3:][::-1]
            
            for i, idx in enumerate(top_3_indices):
                confidence_score = prediction[0][idx]
                st.markdown(f"""
                <div class="metric-card">
                    <strong>{i+1}. {class_names[idx]}</strong><br>
                    Confidence: {confidence_score:.3f}
                </div>
                """, unsafe_allow_html=True)
    
    # Model information
    st.header("📊 Model Information")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Model Type", model_type.upper())
    
    with col2:
        st.metric("Number of Classes", len(class_names))
    
    with col3:
        st.metric("Input Shape", f"{config.get('model.input_shape', [32, 32, 3])}")
    
    # Class names
    st.subheader("Available Classes")
    cols = st.columns(5)
    for i, class_name in enumerate(class_names):
        with cols[i % 5]:
            st.write(f"• {class_name}")
    
    # Instructions
    st.header("📋 Instructions")
    st.markdown("""
    1. **Upload an Image**: Use the file uploader to upload an image of a common object
    2. **Choose Model**: Select from CNN, ResNet, or EfficientNet models
    3. **View Results**: See the predicted class and confidence scores
    4. **Try Sample Images**: Use the sample images to test the model
    
    **Supported Classes**: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck
    
    **Note**: The model works best with images of common objects similar to those in the CIFAR-10 dataset.
    """)


if __name__ == "__main__":
    main()

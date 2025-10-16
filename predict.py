"""
Command-line interface for image classification inference.
"""

import argparse
import numpy as np
import tensorflow as tf
from PIL import Image
from pathlib import Path
import sys
import os

# Add src directory to path
sys.path.append(str(Path(__file__).parent / "src"))

from config import Config
from logger import setup_logging


def preprocess_image(image_path: str) -> np.ndarray:
    """
    Preprocess image for model inference.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        Preprocessed image array
    """
    # Load and resize image
    image = Image.open(image_path)
    image = image.resize((32, 32))
    
    # Convert to RGB if needed
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Convert to numpy array and normalize
    image_array = np.array(image) / 255.0
    
    # Add batch dimension
    image_array = np.expand_dims(image_array, axis=0)
    
    return image_array


def predict_image(model_path: str, image_path: str, config_path: str = "config/config.yaml"):
    """
    Predict the class of an image.
    
    Args:
        model_path: Path to the saved model
        image_path: Path to the image file
        config_path: Path to configuration file
    """
    # Load configuration
    config = Config(config_path)
    class_names = config.get_class_names()
    
    # Setup logging
    logger = setup_logging(level="INFO")
    
    try:
        # Load model
        logger.info(f"Loading model from: {model_path}")
        model = tf.keras.models.load_model(model_path)
        
        # Preprocess image
        logger.info(f"Processing image: {image_path}")
        processed_image = preprocess_image(image_path)
        
        # Make prediction
        logger.info("Making prediction...")
        prediction = model.predict(processed_image, verbose=0)
        predicted_class_idx = np.argmax(prediction[0])
        predicted_class = class_names[predicted_class_idx]
        confidence = prediction[0][predicted_class_idx]
        
        # Display results
        print("\n" + "="*50)
        print("IMAGE CLASSIFICATION RESULTS")
        print("="*50)
        print(f"Image: {image_path}")
        print(f"Predicted Class: {predicted_class}")
        print(f"Confidence: {confidence:.3f}")
        print("\nAll Class Probabilities:")
        
        for i, (class_name, prob) in enumerate(zip(class_names, prediction[0])):
            print(f"  {class_name}: {prob:.3f}")
        
        print("="*50)
        
    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        raise


def main():
    """Main CLI function."""
    parser = argparse.ArgumentParser(description="Classify an image using a trained model")
    parser.add_argument("image_path", help="Path to the image file")
    parser.add_argument("--model", required=True, help="Path to the saved model file")
    parser.add_argument("--config", default="config/config.yaml", help="Path to configuration file")
    
    args = parser.parse_args()
    
    # Validate inputs
    if not Path(args.image_path).exists():
        print(f"Error: Image file not found: {args.image_path}")
        sys.exit(1)
    
    if not Path(args.model).exists():
        print(f"Error: Model file not found: {args.model}")
        sys.exit(1)
    
    # Make prediction
    predict_image(args.model, args.image_path, args.config)


if __name__ == "__main__":
    main()

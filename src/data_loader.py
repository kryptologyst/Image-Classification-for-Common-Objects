"""
Data handling module for Image Classification Project.
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from typing import Tuple, Optional, Dict, Any
import logging

from .config import Config
from .logger import get_logger

logger = get_logger(__name__)


class DataLoader:
    """Data loader and preprocessor for image classification."""
    
    def __init__(self, config: Config):
        """
        Initialize data loader.
        
        Args:
            config: Configuration object
        """
        self.config = config
        self.data_config = config.get_data_config()
        self.class_names = config.get_class_names()
        
    def load_cifar10(self) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
        """
        Load CIFAR-10 dataset.
        
        Returns:
            Tuple of (x_train, y_train), (x_test, y_test)
        """
        logger.info("Loading CIFAR-10 dataset...")
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
        
        logger.info(f"Training set shape: {x_train.shape}")
        logger.info(f"Test set shape: {x_test.shape}")
        
        return (x_train, y_train), (x_test, y_test)
    
    def preprocess_data(
        self, 
        x_train: np.ndarray, 
        x_test: np.ndarray, 
        y_train: np.ndarray, 
        y_test: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Preprocess the data (normalize, encode labels).
        
        Args:
            x_train: Training images
            x_test: Test images
            y_train: Training labels
            y_test: Test labels
            
        Returns:
            Preprocessed data tuple
        """
        logger.info("Preprocessing data...")
        
        # Normalize pixel values to 0-1 range
        if self.data_config.get('normalize', True):
            x_train = x_train.astype('float32') / 255.0
            x_test = x_test.astype('float32') / 255.0
            logger.info("Data normalized to 0-1 range")
        
        # One-hot encode labels
        num_classes = self.config.get('model.num_classes', 10)
        y_train_cat = to_categorical(y_train, num_classes)
        y_test_cat = to_categorical(y_test, num_classes)
        logger.info(f"Labels one-hot encoded for {num_classes} classes")
        
        return x_train, x_test, y_train_cat, y_test_cat
    
    def create_data_generators(
        self, 
        x_train: np.ndarray, 
        y_train: np.ndarray,
        x_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None
    ) -> Tuple[tf.keras.utils.Sequence, Optional[tf.keras.utils.Sequence]]:
        """
        Create data generators for training and validation.
        
        Args:
            x_train: Training images
            y_train: Training labels
            x_val: Validation images (optional)
            y_val: Validation labels (optional)
            
        Returns:
            Tuple of training and validation data generators
        """
        batch_size = self.config.get('model.batch_size', 64)
        
        # Training data generator
        train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
            rotation_range=20,
            width_shift_range=0.1,
            height_shift_range=0.1,
            horizontal_flip=True,
            zoom_range=0.1
        )
        
        train_generator = train_datagen.flow(
            x_train, y_train,
            batch_size=batch_size
        )
        
        # Validation data generator (if validation data provided)
        val_generator = None
        if x_val is not None and y_val is not None:
            val_datagen = tf.keras.preprocessing.image.ImageDataGenerator()
            val_generator = val_datagen.flow(
                x_val, y_val,
                batch_size=batch_size
            )
        
        logger.info("Data generators created")
        return train_generator, val_generator
    
    def get_data_info(self) -> Dict[str, Any]:
        """
        Get information about the dataset.
        
        Returns:
            Dictionary with dataset information
        """
        return {
            'dataset_name': self.data_config.get('dataset_name', 'cifar10'),
            'num_classes': len(self.class_names),
            'class_names': self.class_names,
            'input_shape': self.config.get('model.input_shape', [32, 32, 3])
        }

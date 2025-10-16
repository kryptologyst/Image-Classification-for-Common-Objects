"""
Model definitions for Image Classification Project.
Includes both traditional CNN and modern transformer-based models.
"""

import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, Flatten, Dense, Dropout, 
    BatchNormalization, GlobalAveragePooling2D, Input
)
from tensorflow.keras.applications import ResNet50, EfficientNetB0
from tensorflow.keras.optimizers import Adam
from typing import Dict, Any, Optional, Tuple
import logging

from .config import Config
from .logger import get_logger

logger = get_logger(__name__)


class CNNModelBuilder:
    """Builder class for CNN models."""
    
    def __init__(self, config: Config):
        """
        Initialize model builder.
        
        Args:
            config: Configuration object
        """
        self.config = config
        self.model_config = config.get_model_config()
        
    def build_cnn_model(self) -> Sequential:
        """
        Build a traditional CNN model.
        
        Returns:
            Compiled CNN model
        """
        logger.info("Building CNN model...")
        
        input_shape = tuple(self.model_config.get('input_shape', [32, 32, 3]))
        num_classes = self.model_config.get('num_classes', 10)
        dropout_rate = self.model_config.get('dropout_rate', 0.5)
        
        model = Sequential([
            # First convolutional block
            Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
            BatchNormalization(),
            MaxPooling2D(2, 2),
            
            # Second convolutional block
            Conv2D(64, (3, 3), activation='relu'),
            BatchNormalization(),
            MaxPooling2D(2, 2),
            
            # Third convolutional block
            Conv2D(128, (3, 3), activation='relu'),
            BatchNormalization(),
            MaxPooling2D(2, 2),
            
            # Fourth convolutional block
            Conv2D(256, (3, 3), activation='relu'),
            BatchNormalization(),
            GlobalAveragePooling2D(),
            
            # Dense layers
            Dense(512, activation='relu'),
            Dropout(dropout_rate),
            Dense(256, activation='relu'),
            Dropout(dropout_rate),
            Dense(num_classes, activation='softmax')
        ])
        
        self._compile_model(model)
        logger.info("CNN model built successfully")
        return model
    
    def build_resnet_model(self, pretrained: bool = True) -> Model:
        """
        Build a ResNet-based model.
        
        Args:
            pretrained: Whether to use pretrained weights
            
        Returns:
            Compiled ResNet model
        """
        logger.info("Building ResNet model...")
        
        input_shape = tuple(self.model_config.get('input_shape', [32, 32, 3]))
        num_classes = self.model_config.get('num_classes', 10)
        
        # Load pretrained ResNet50
        base_model = ResNet50(
            weights='imagenet' if pretrained else None,
            include_top=False,
            input_shape=input_shape
        )
        
        # Freeze base model layers if using pretrained weights
        if pretrained:
            base_model.trainable = False
        
        # Add custom classification head
        inputs = Input(shape=input_shape)
        x = base_model(inputs, training=False)
        x = GlobalAveragePooling2D()(x)
        x = Dense(512, activation='relu')(x)
        x = Dropout(0.5)(x)
        outputs = Dense(num_classes, activation='softmax')(x)
        
        model = Model(inputs, outputs)
        self._compile_model(model)
        
        logger.info("ResNet model built successfully")
        return model
    
    def build_efficientnet_model(self, pretrained: bool = True) -> Model:
        """
        Build an EfficientNet-based model.
        
        Args:
            pretrained: Whether to use pretrained weights
            
        Returns:
            Compiled EfficientNet model
        """
        logger.info("Building EfficientNet model...")
        
        input_shape = tuple(self.model_config.get('input_shape', [32, 32, 3]))
        num_classes = self.model_config.get('num_classes', 10)
        
        # Load pretrained EfficientNetB0
        base_model = EfficientNetB0(
            weights='imagenet' if pretrained else None,
            include_top=False,
            input_shape=input_shape
        )
        
        # Freeze base model layers if using pretrained weights
        if pretrained:
            base_model.trainable = False
        
        # Add custom classification head
        inputs = Input(shape=input_shape)
        x = base_model(inputs, training=False)
        x = GlobalAveragePooling2D()(x)
        x = Dense(512, activation='relu')(x)
        x = Dropout(0.5)(x)
        outputs = Dense(num_classes, activation='softmax')(x)
        
        model = Model(inputs, outputs)
        self._compile_model(model)
        
        logger.info("EfficientNet model built successfully")
        return model
    
    def _compile_model(self, model: Model) -> None:
        """
        Compile the model with optimizer, loss, and metrics.
        
        Args:
            model: Model to compile
        """
        training_config = self.config.get_training_config()
        
        optimizer = Adam(learning_rate=self.model_config.get('learning_rate', 0.001))
        loss = training_config.get('loss', 'categorical_crossentropy')
        metrics = training_config.get('metrics', ['accuracy'])
        
        model.compile(
            optimizer=optimizer,
            loss=loss,
            metrics=metrics
        )
        
        logger.info(f"Model compiled with optimizer: {optimizer}, loss: {loss}, metrics: {metrics}")


class ModelFactory:
    """Factory class for creating different types of models."""
    
    @staticmethod
    def create_model(model_type: str, config: Config) -> Model:
        """
        Create a model based on the specified type.
        
        Args:
            model_type: Type of model to create ('cnn', 'resnet', 'efficientnet')
            config: Configuration object
            
        Returns:
            Compiled model
        """
        builder = CNNModelBuilder(config)
        
        if model_type.lower() == 'cnn':
            return builder.build_cnn_model()
        elif model_type.lower() == 'resnet':
            return builder.build_resnet_model()
        elif model_type.lower() == 'efficientnet':
            return builder.build_efficientnet_model()
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    @staticmethod
    def get_available_models() -> list:
        """
        Get list of available model types.
        
        Returns:
            List of available model types
        """
        return ['cnn', 'resnet', 'efficientnet']

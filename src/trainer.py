"""
Training module for Image Classification Project.
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import (
    EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, 
    LearningRateScheduler, CSVLogger
)
from typing import Dict, Any, Optional, Tuple, List
import os
from pathlib import Path
import logging

from .config import Config
from .logger import get_logger

logger = get_logger(__name__)


class Trainer:
    """Trainer class for image classification models."""
    
    def __init__(self, config: Config):
        """
        Initialize trainer.
        
        Args:
            config: Configuration object
        """
        self.config = config
        self.training_config = config.get_training_config()
        self.model_config = config.get_model_config()
        
    def create_callbacks(self, model_name: str) -> List[tf.keras.callbacks.Callback]:
        """
        Create training callbacks.
        
        Args:
            model_name: Name of the model for saving
            
        Returns:
            List of callbacks
        """
        callbacks = []
        
        # Early stopping
        if self.training_config.get('early_stopping', {}).get('enabled', True):
            early_stopping = EarlyStopping(
                monitor=self.training_config['early_stopping'].get('monitor', 'val_loss'),
                patience=self.training_config['early_stopping'].get('patience', 5),
                restore_best_weights=True,
                verbose=1
            )
            callbacks.append(early_stopping)
            logger.info("Early stopping callback added")
        
        # Model checkpoint
        if self.training_config.get('model_checkpoint', {}).get('enabled', True):
            models_dir = Path(self.config.get('paths.models_dir', 'models'))
            models_dir.mkdir(parents=True, exist_ok=True)
            
            checkpoint_path = models_dir / f"{model_name}_best.h5"
            model_checkpoint = ModelCheckpoint(
                filepath=str(checkpoint_path),
                monitor=self.training_config['model_checkpoint'].get('monitor', 'val_accuracy'),
                save_best_only=self.training_config['model_checkpoint'].get('save_best_only', True),
                verbose=1
            )
            callbacks.append(model_checkpoint)
            logger.info(f"Model checkpoint callback added: {checkpoint_path}")
        
        # Learning rate reduction
        lr_reduction = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=1e-7,
            verbose=1
        )
        callbacks.append(lr_reduction)
        logger.info("Learning rate reduction callback added")
        
        # CSV logger
        logs_dir = Path(self.config.get('paths.logs_dir', 'logs'))
        logs_dir.mkdir(parents=True, exist_ok=True)
        
        csv_logger = CSVLogger(
            filename=str(logs_dir / f"{model_name}_training.csv"),
            append=True
        )
        callbacks.append(csv_logger)
        logger.info(f"CSV logger callback added: {logs_dir / f'{model_name}_training.csv'}")
        
        return callbacks
    
    def train_model(
        self,
        model: tf.keras.Model,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        model_name: str = "model"
    ) -> tf.keras.callbacks.History:
        """
        Train the model.
        
        Args:
            model: Model to train
            x_train: Training images
            y_train: Training labels
            x_val: Validation images (optional)
            y_val: Validation labels (optional)
            model_name: Name of the model
            
        Returns:
            Training history
        """
        logger.info(f"Starting training for {model_name}...")
        
        # Create callbacks
        callbacks = self.create_callbacks(model_name)
        
        # Training parameters
        epochs = self.model_config.get('epochs', 10)
        batch_size = self.model_config.get('batch_size', 64)
        
        # Validation data
        validation_data = None
        if x_val is not None and y_val is not None:
            validation_data = (x_val, y_val)
            logger.info(f"Using validation data: {x_val.shape[0]} samples")
        
        # Train the model
        history = model.fit(
            x_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=validation_data,
            callbacks=callbacks,
            verbose=1
        )
        
        logger.info(f"Training completed for {model_name}")
        return history
    
    def evaluate_model(
        self,
        model: tf.keras.Model,
        x_test: np.ndarray,
        y_test: np.ndarray,
        class_names: List[str]
    ) -> Dict[str, Any]:
        """
        Evaluate the model on test data.
        
        Args:
            model: Trained model
            x_test: Test images
            y_test: Test labels
            class_names: List of class names
            
        Returns:
            Evaluation results
        """
        logger.info("Evaluating model on test data...")
        
        # Evaluate model
        test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
        
        # Get predictions
        predictions = model.predict(x_test, verbose=0)
        predicted_classes = np.argmax(predictions, axis=1)
        true_classes = np.argmax(y_test, axis=1)
        
        # Calculate per-class accuracy
        class_accuracy = {}
        for i, class_name in enumerate(class_names):
            class_mask = true_classes == i
            if np.sum(class_mask) > 0:
                class_acc = np.mean(predicted_classes[class_mask] == true_classes[class_mask])
                class_accuracy[class_name] = class_acc
        
        results = {
            'test_loss': float(test_loss),
            'test_accuracy': float(test_accuracy),
            'class_accuracy': class_accuracy,
            'predictions': predictions,
            'predicted_classes': predicted_classes,
            'true_classes': true_classes
        }
        
        logger.info(f"Test accuracy: {test_accuracy:.4f}")
        logger.info(f"Test loss: {test_loss:.4f}")
        
        return results
    
    def save_model(self, model: tf.keras.Model, model_name: str) -> str:
        """
        Save the trained model.
        
        Args:
            model: Model to save
            model_name: Name for the saved model
            
        Returns:
            Path to saved model
        """
        models_dir = Path(self.config.get('paths.models_dir', 'models'))
        models_dir.mkdir(parents=True, exist_ok=True)
        
        model_path = models_dir / f"{model_name}_final.h5"
        model.save(str(model_path))
        
        logger.info(f"Model saved to: {model_path}")
        return str(model_path)
    
    def load_model(self, model_path: str) -> tf.keras.Model:
        """
        Load a saved model.
        
        Args:
            model_path: Path to the saved model
            
        Returns:
            Loaded model
        """
        logger.info(f"Loading model from: {model_path}")
        model = tf.keras.models.load_model(model_path)
        logger.info("Model loaded successfully")
        return model

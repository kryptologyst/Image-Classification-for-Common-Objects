"""
Visualization and explainability module for Image Classification Project.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from typing import Dict, Any, List, Optional, Tuple
import tensorflow as tf
from pathlib import Path
import logging

from .config import Config
from .logger import get_logger

logger = get_logger(__name__)


class Visualizer:
    """Visualization and explainability class for image classification."""
    
    def __init__(self, config: Config):
        """
        Initialize visualizer.
        
        Args:
            config: Configuration object
        """
        self.config = config
        self.class_names = config.get_class_names()
        
    def plot_training_history(self, history: tf.keras.callbacks.History, save_path: Optional[str] = None) -> None:
        """
        Plot training history (loss and accuracy curves).
        
        Args:
            history: Training history object
            save_path: Path to save the plot (optional)
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Plot training & validation accuracy
        ax1.plot(history.history['accuracy'], label='Training Accuracy')
        if 'val_accuracy' in history.history:
            ax1.plot(history.history['val_accuracy'], label='Validation Accuracy')
        ax1.set_title('Model Accuracy')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.legend()
        ax1.grid(True)
        
        # Plot training & validation loss
        ax2.plot(history.history['loss'], label='Training Loss')
        if 'val_loss' in history.history:
            ax2.plot(history.history['val_loss'], label='Validation Loss')
        ax2.set_title('Model Loss')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Training history plot saved to: {save_path}")
        
        plt.show()
    
    def plot_confusion_matrix(
        self, 
        y_true: np.ndarray, 
        y_pred: np.ndarray, 
        save_path: Optional[str] = None
    ) -> None:
        """
        Plot confusion matrix.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            save_path: Path to save the plot (optional)
        """
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm, 
            annot=True, 
            fmt='d', 
            cmap='Blues',
            xticklabels=self.class_names,
            yticklabels=self.class_names
        )
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Confusion matrix plot saved to: {save_path}")
        
        plt.show()
    
    def plot_class_accuracy(self, class_accuracy: Dict[str, float], save_path: Optional[str] = None) -> None:
        """
        Plot per-class accuracy.
        
        Args:
            class_accuracy: Dictionary with class names and their accuracies
            save_path: Path to save the plot (optional)
        """
        classes = list(class_accuracy.keys())
        accuracies = list(class_accuracy.values())
        
        plt.figure(figsize=(12, 6))
        bars = plt.bar(classes, accuracies, color='skyblue', edgecolor='navy', alpha=0.7)
        
        # Add value labels on bars
        for bar, acc in zip(bars, accuracies):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{acc:.3f}', ha='center', va='bottom')
        
        plt.title('Per-Class Accuracy')
        plt.xlabel('Class')
        plt.ylabel('Accuracy')
        plt.xticks(rotation=45)
        plt.ylim(0, 1)
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Class accuracy plot saved to: {save_path}")
        
        plt.show()
    
    def visualize_predictions(
        self, 
        images: np.ndarray, 
        true_labels: np.ndarray, 
        predicted_labels: np.ndarray,
        probabilities: np.ndarray,
        num_samples: int = 16,
        save_path: Optional[str] = None
    ) -> None:
        """
        Visualize model predictions on sample images.
        
        Args:
            images: Sample images
            true_labels: True labels
            predicted_labels: Predicted labels
            probabilities: Prediction probabilities
            num_samples: Number of samples to display
            save_path: Path to save the plot (optional)
        """
        fig, axes = plt.subplots(4, 4, figsize=(15, 15))
        axes = axes.ravel()
        
        for i in range(min(num_samples, len(images))):
            axes[i].imshow(images[i])
            
            true_class = self.class_names[true_labels[i]]
            pred_class = self.class_names[predicted_labels[i]]
            confidence = probabilities[i][predicted_labels[i]]
            
            # Color based on correctness
            color = 'green' if true_labels[i] == predicted_labels[i] else 'red'
            
            axes[i].set_title(
                f'True: {true_class}\nPred: {pred_class}\nConf: {confidence:.3f}',
                color=color
            )
            axes[i].axis('off')
        
        # Hide unused subplots
        for i in range(num_samples, len(axes)):
            axes[i].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Predictions visualization saved to: {save_path}")
        
        plt.show()
    
    def plot_feature_maps(
        self, 
        model: tf.keras.Model, 
        image: np.ndarray, 
        layer_name: str,
        save_path: Optional[str] = None
    ) -> None:
        """
        Visualize feature maps from a specific layer.
        
        Args:
            model: Trained model
            image: Input image
            layer_name: Name of the layer to visualize
            save_path: Path to save the plot (optional)
        """
        # Create a model that outputs the feature maps
        feature_extractor = tf.keras.Model(
            inputs=model.input,
            outputs=model.get_layer(layer_name).output
        )
        
        # Get feature maps
        feature_maps = feature_extractor.predict(np.expand_dims(image, axis=0))
        
        # Plot feature maps
        num_filters = min(16, feature_maps.shape[-1])
        fig, axes = plt.subplots(4, 4, figsize=(12, 12))
        axes = axes.ravel()
        
        for i in range(num_filters):
            axes[i].imshow(feature_maps[0, :, :, i], cmap='viridis')
            axes[i].set_title(f'Filter {i}')
            axes[i].axis('off')
        
        # Hide unused subplots
        for i in range(num_filters, len(axes)):
            axes[i].axis('off')
        
        plt.suptitle(f'Feature Maps from {layer_name}')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Feature maps plot saved to: {save_path}")
        
        plt.show()
    
    def generate_classification_report(
        self, 
        y_true: np.ndarray, 
        y_pred: np.ndarray,
        save_path: Optional[str] = None
    ) -> str:
        """
        Generate and display classification report.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            save_path: Path to save the report (optional)
            
        Returns:
            Classification report as string
        """
        report = classification_report(
            y_true, y_pred, 
            target_names=self.class_names,
            digits=4
        )
        
        print(report)
        
        if save_path:
            with open(save_path, 'w') as f:
                f.write(report)
            logger.info(f"Classification report saved to: {save_path}")
        
        return report
    
    def save_all_visualizations(
        self,
        history: tf.keras.callbacks.History,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        class_accuracy: Dict[str, float],
        images: np.ndarray,
        probabilities: np.ndarray,
        results_dir: str = "results"
    ) -> None:
        """
        Save all visualizations to the results directory.
        
        Args:
            history: Training history
            y_true: True labels
            y_pred: Predicted labels
            class_accuracy: Per-class accuracy
            images: Sample images
            probabilities: Prediction probabilities
            results_dir: Results directory path
        """
        results_path = Path(results_dir)
        results_path.mkdir(parents=True, exist_ok=True)
        
        # Save training history
        self.plot_training_history(history, str(results_path / "training_history.png"))
        
        # Save confusion matrix
        self.plot_confusion_matrix(y_true, y_pred, str(results_path / "confusion_matrix.png"))
        
        # Save class accuracy
        self.plot_class_accuracy(class_accuracy, str(results_path / "class_accuracy.png"))
        
        # Save predictions visualization
        self.visualize_predictions(
            images[:16], y_true[:16], y_pred[:16], probabilities[:16],
            save_path=str(results_path / "predictions_sample.png")
        )
        
        # Save classification report
        self.generate_classification_report(y_true, y_pred, str(results_path / "classification_report.txt"))
        
        logger.info(f"All visualizations saved to: {results_path}")

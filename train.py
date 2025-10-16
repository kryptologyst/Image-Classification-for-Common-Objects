"""
Main training script for Image Classification Project.
"""

import argparse
import numpy as np
from pathlib import Path
import sys
import os

# Add src directory to path
sys.path.append(str(Path(__file__).parent / "src"))

from config import Config
from logger import setup_logging, get_logger
from data_loader import DataLoader
from models import ModelFactory
from trainer import Trainer
from visualizer import Visualizer


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train image classification model")
    parser.add_argument("--model-type", type=str, default="cnn", 
                       choices=["cnn", "resnet", "efficientnet"],
                       help="Type of model to train")
    parser.add_argument("--config", type=str, default="config/config.yaml",
                       help="Path to configuration file")
    parser.add_argument("--epochs", type=int, default=None,
                       help="Number of training epochs (overrides config)")
    parser.add_argument("--batch-size", type=int, default=None,
                       help="Batch size (overrides config)")
    parser.add_argument("--save-results", action="store_true",
                       help="Save all results and visualizations")
    
    args = parser.parse_args()
    
    # Load configuration
    config = Config(args.config)
    
    # Setup logging
    logger = setup_logging(
        level=config.get('logging.level', 'INFO'),
        log_file=config.get('logging.file', 'logs/training.log')
    )
    
    # Create necessary directories
    config.create_directories()
    
    logger.info("=" * 50)
    logger.info("Starting Image Classification Training")
    logger.info("=" * 50)
    logger.info(f"Model type: {args.model_type}")
    logger.info(f"Configuration: {args.config}")
    
    try:
        # Initialize components
        data_loader = DataLoader(config)
        trainer = Trainer(config)
        visualizer = Visualizer(config)
        
        # Load and preprocess data
        logger.info("Loading and preprocessing data...")
        (x_train, y_train), (x_test, y_test) = data_loader.load_cifar10()
        x_train, x_test, y_train_cat, y_test_cat = data_loader.preprocess_data(
            x_train, x_test, y_train, y_test
        )
        
        # Split training data for validation
        validation_split = config.get('data.validation_split', 0.2)
        if validation_split > 0:
            split_idx = int(len(x_train) * (1 - validation_split))
            x_val = x_train[split_idx:]
            y_val = y_train_cat[split_idx:]
            x_train = x_train[:split_idx]
            y_train_cat = y_train_cat[:split_idx]
            logger.info(f"Split data: {len(x_train)} train, {len(x_val)} validation")
        else:
            x_val, y_val = None, None
        
        # Create model
        logger.info(f"Creating {args.model_type} model...")
        model = ModelFactory.create_model(args.model_type, config)
        
        # Override config with command line arguments
        if args.epochs:
            config._config['model']['epochs'] = args.epochs
        if args.batch_size:
            config._config['model']['batch_size'] = args.batch_size
        
        # Train model
        logger.info("Starting model training...")
        history = trainer.train_model(
            model, x_train, y_train_cat, x_val, y_val, 
            model_name=f"{args.model_type}_model"
        )
        
        # Evaluate model
        logger.info("Evaluating model...")
        results = trainer.evaluate_model(model, x_test, y_test_cat, config.get_class_names())
        
        # Print results
        logger.info("=" * 50)
        logger.info("TRAINING RESULTS")
        logger.info("=" * 50)
        logger.info(f"Test Accuracy: {results['test_accuracy']:.4f}")
        logger.info(f"Test Loss: {results['test_loss']:.4f}")
        
        # Per-class accuracy
        logger.info("\nPer-Class Accuracy:")
        for class_name, accuracy in results['class_accuracy'].items():
            logger.info(f"  {class_name}: {accuracy:.4f}")
        
        # Save model
        model_path = trainer.save_model(model, f"{args.model_type}_model")
        
        # Generate visualizations
        if args.save_results:
            logger.info("Generating visualizations...")
            visualizer.save_all_visualizations(
                history=history,
                y_true=results['true_classes'],
                y_pred=results['predicted_classes'],
                class_accuracy=results['class_accuracy'],
                images=x_test[:100],  # Use first 100 test images
                probabilities=results['predictions'][:100],
                results_dir=config.get('paths.results_dir', 'results')
            )
        else:
            # Show plots interactively
            visualizer.plot_training_history(history)
            visualizer.plot_confusion_matrix(results['true_classes'], results['predicted_classes'])
            visualizer.plot_class_accuracy(results['class_accuracy'])
            visualizer.visualize_predictions(
                x_test[:16], results['true_classes'][:16], 
                results['predicted_classes'][:16], results['predictions'][:16]
            )
        
        logger.info("=" * 50)
        logger.info("Training completed successfully!")
        logger.info(f"Model saved to: {model_path}")
        logger.info("=" * 50)
        
    except Exception as e:
        logger.error(f"Training failed with error: {str(e)}")
        raise


if __name__ == "__main__":
    main()

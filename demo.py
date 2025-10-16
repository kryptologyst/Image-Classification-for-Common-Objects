"""
Demo script to test the image classification system.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import os

# Add src directory to path
sys.path.append(str(Path(__file__).parent / "src"))

from config import Config
from logger import setup_logging
from data_loader import DataLoader
from models import ModelFactory
from trainer import Trainer
from visualizer import Visualizer


def demo_training():
    """Demonstrate the training process with a small subset of data."""
    print("🚀 Starting Image Classification Demo")
    print("=" * 50)
    
    # Setup
    config = Config("config/config.yaml")
    logger = setup_logging(level="INFO")
    
    # Initialize components
    data_loader = DataLoader(config)
    trainer = Trainer(config)
    visualizer = Visualizer(config)
    
    print("📊 Loading CIFAR-10 dataset...")
    (x_train, y_train), (x_test, y_test) = data_loader.load_cifar10()
    
    # Use only a small subset for demo
    demo_size = 1000
    x_train_demo = x_train[:demo_size]
    y_train_demo = y_train[:demo_size]
    x_test_demo = x_test[:200]
    y_test_demo = y_test[:200]
    
    print(f"Using demo subset: {len(x_train_demo)} train, {len(x_test_demo)} test")
    
    # Preprocess data
    print("🔄 Preprocessing data...")
    x_train_proc, x_test_proc, y_train_proc, y_test_proc = data_loader.preprocess_data(
        x_train_demo, x_test_demo, y_train_demo, y_test_demo
    )
    
    # Create and train model
    print("🧠 Creating CNN model...")
    model = ModelFactory.create_model('cnn', config)
    
    # Override epochs for demo
    config._config['model']['epochs'] = 3
    
    print("🏋️ Training model (3 epochs for demo)...")
    history = trainer.train_model(
        model, x_train_proc, y_train_proc, 
        model_name="demo_cnn_model"
    )
    
    # Evaluate model
    print("📈 Evaluating model...")
    results = trainer.evaluate_model(
        model, x_test_proc, y_test_proc, config.get_class_names()
    )
    
    # Display results
    print("\n" + "=" * 50)
    print("DEMO RESULTS")
    print("=" * 50)
    print(f"Test Accuracy: {results['test_accuracy']:.4f}")
    print(f"Test Loss: {results['test_loss']:.4f}")
    
    print("\nPer-Class Accuracy:")
    for class_name, accuracy in results['class_accuracy'].items():
        print(f"  {class_name}: {accuracy:.4f}")
    
    # Generate visualizations
    print("\n📊 Generating visualizations...")
    visualizer.plot_training_history(history)
    visualizer.plot_confusion_matrix(results['true_classes'], results['predicted_classes'])
    visualizer.plot_class_accuracy(results['class_accuracy'])
    
    # Show some predictions
    print("\n🔍 Sample Predictions:")
    sample_indices = np.random.choice(len(x_test_demo), 5, replace=False)
    for idx in sample_indices:
        true_class = config.get_class_names()[results['true_classes'][idx]]
        pred_class = config.get_class_names()[results['predicted_classes'][idx]]
        confidence = results['predictions'][idx][results['predicted_classes'][idx]]
        correct = "✅" if results['true_classes'][idx] == results['predicted_classes'][idx] else "❌"
        
        print(f"  {correct} True: {true_class}, Predicted: {pred_class} (conf: {confidence:.3f})")
    
    print("\n🎉 Demo completed successfully!")
    print("=" * 50)


if __name__ == "__main__":
    demo_training()

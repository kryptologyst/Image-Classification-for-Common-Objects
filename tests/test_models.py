"""
Unit tests for Image Classification Project.
"""

import unittest
import numpy as np
import tensorflow as tf
from pathlib import Path
import sys
import os

# Add src directory to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from config import Config
from data_loader import DataLoader
from models import ModelFactory, CNNModelBuilder
from trainer import Trainer
from visualizer import Visualizer


class TestConfig(unittest.TestCase):
    """Test configuration management."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = Config("config/config.yaml")
    
    def test_config_loading(self):
        """Test configuration loading."""
        self.assertIsNotNone(self.config._config)
        self.assertIn('model', self.config._config)
        self.assertIn('data', self.config._config)
    
    def test_get_method(self):
        """Test configuration get method."""
        epochs = self.config.get('model.epochs')
        self.assertIsNotNone(epochs)
        self.assertIsInstance(epochs, int)
    
    def test_get_with_default(self):
        """Test configuration get with default value."""
        non_existent = self.config.get('non.existent.key', 'default')
        self.assertEqual(non_existent, 'default')
    
    def test_get_model_config(self):
        """Test getting model configuration."""
        model_config = self.config.get_model_config()
        self.assertIsInstance(model_config, dict)
        self.assertIn('epochs', model_config)
    
    def test_get_class_names(self):
        """Test getting class names."""
        class_names = self.config.get_class_names()
        self.assertIsInstance(class_names, list)
        self.assertEqual(len(class_names), 10)


class TestDataLoader(unittest.TestCase):
    """Test data loading and preprocessing."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = Config("config/config.yaml")
        self.data_loader = DataLoader(self.config)
    
    def test_load_cifar10(self):
        """Test CIFAR-10 data loading."""
        (x_train, y_train), (x_test, y_test) = self.data_loader.load_cifar10()
        
        # Check shapes
        self.assertEqual(x_train.shape, (50000, 32, 32, 3))
        self.assertEqual(x_test.shape, (10000, 32, 32, 3))
        self.assertEqual(y_train.shape, (50000, 1))
        self.assertEqual(y_test.shape, (10000, 1))
        
        # Check data types
        self.assertEqual(x_train.dtype, np.uint8)
        self.assertEqual(y_train.dtype, np.uint8)
    
    def test_preprocess_data(self):
        """Test data preprocessing."""
        # Create dummy data
        x_train = np.random.randint(0, 256, (100, 32, 32, 3), dtype=np.uint8)
        x_test = np.random.randint(0, 256, (50, 32, 32, 3), dtype=np.uint8)
        y_train = np.random.randint(0, 10, (100, 1), dtype=np.uint8)
        y_test = np.random.randint(0, 10, (50, 1), dtype=np.uint8)
        
        x_train_proc, x_test_proc, y_train_proc, y_test_proc = self.data_loader.preprocess_data(
            x_train, x_test, y_train, y_test
        )
        
        # Check normalization
        self.assertTrue(np.all(x_train_proc >= 0))
        self.assertTrue(np.all(x_train_proc <= 1))
        
        # Check one-hot encoding
        self.assertEqual(y_train_proc.shape, (100, 10))
        self.assertEqual(y_test_proc.shape, (50, 10))
    
    def test_get_data_info(self):
        """Test getting data information."""
        info = self.data_loader.get_data_info()
        self.assertIn('dataset_name', info)
        self.assertIn('num_classes', info)
        self.assertIn('class_names', info)
        self.assertIn('input_shape', info)


class TestModels(unittest.TestCase):
    """Test model creation and compilation."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = Config("config/config.yaml")
        self.builder = CNNModelBuilder(self.config)
    
    def test_cnn_model_creation(self):
        """Test CNN model creation."""
        model = self.builder.build_cnn_model()
        
        # Check model structure
        self.assertIsInstance(model, tf.keras.Model)
        self.assertEqual(len(model.layers), 12)  # Expected number of layers
        
        # Check input shape
        self.assertEqual(model.input_shape, (None, 32, 32, 3))
        
        # Check output shape
        self.assertEqual(model.output_shape, (None, 10))
    
    def test_model_factory(self):
        """Test model factory."""
        # Test CNN model
        cnn_model = ModelFactory.create_model('cnn', self.config)
        self.assertIsInstance(cnn_model, tf.keras.Model)
        
        # Test available models
        available_models = ModelFactory.get_available_models()
        self.assertIn('cnn', available_models)
        self.assertIn('resnet', available_models)
        self.assertIn('efficientnet', available_models)
    
    def test_model_compilation(self):
        """Test model compilation."""
        model = self.builder.build_cnn_model()
        
        # Check optimizer
        self.assertIsInstance(model.optimizer, tf.keras.optimizers.Optimizer)
        
        # Check loss function
        self.assertIsNotNone(model.loss)
        
        # Check metrics
        self.assertIsInstance(model.metrics, list)


class TestTrainer(unittest.TestCase):
    """Test training functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = Config("config/config.yaml")
        self.trainer = Trainer(self.config)
        
        # Create dummy data
        self.x_train = np.random.random((100, 32, 32, 3))
        self.y_train = tf.keras.utils.to_categorical(np.random.randint(0, 10, 100), 10)
        self.x_test = np.random.random((50, 32, 32, 3))
        self.y_test = tf.keras.utils.to_categorical(np.random.randint(0, 10, 50), 10)
    
    def test_create_callbacks(self):
        """Test callback creation."""
        callbacks = self.trainer.create_callbacks("test_model")
        
        self.assertIsInstance(callbacks, list)
        self.assertGreater(len(callbacks), 0)
        
        # Check callback types
        callback_types = [type(callback).__name__ for callback in callbacks]
        self.assertIn('EarlyStopping', callback_types)
        self.assertIn('ModelCheckpoint', callback_types)
    
    def test_evaluate_model(self):
        """Test model evaluation."""
        # Create a simple model
        model = tf.keras.Sequential([
            tf.keras.layers.Flatten(input_shape=(32, 32, 3)),
            tf.keras.layers.Dense(10, activation='softmax')
        ])
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        
        # Train briefly
        model.fit(self.x_train, self.y_train, epochs=1, verbose=0)
        
        # Evaluate
        results = self.trainer.evaluate_model(model, self.x_test, self.y_test, 
                                            self.config.get_class_names())
        
        self.assertIn('test_loss', results)
        self.assertIn('test_accuracy', results)
        self.assertIn('class_accuracy', results)
        self.assertIn('predictions', results)


class TestVisualizer(unittest.TestCase):
    """Test visualization functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = Config("config/config.yaml")
        self.visualizer = Visualizer(self.config)
        
        # Create dummy data
        self.y_true = np.random.randint(0, 10, 100)
        self.y_pred = np.random.randint(0, 10, 100)
        self.class_accuracy = {name: np.random.random() for name in self.config.get_class_names()}
    
    def test_class_accuracy_plot(self):
        """Test class accuracy plotting."""
        # This test just checks that the method runs without error
        try:
            self.visualizer.plot_class_accuracy(self.class_accuracy)
            # If we get here, the method ran successfully
            self.assertTrue(True)
        except Exception as e:
            self.fail(f"plot_class_accuracy raised an exception: {e}")
    
    def test_classification_report(self):
        """Test classification report generation."""
        report = self.visualizer.generate_classification_report(self.y_true, self.y_pred)
        
        self.assertIsInstance(report, str)
        self.assertIn('precision', report.lower())
        self.assertIn('recall', report.lower())
        self.assertIn('f1-score', report.lower())


if __name__ == '__main__':
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test cases
    test_suite.addTest(unittest.makeSuite(TestConfig))
    test_suite.addTest(unittest.makeSuite(TestDataLoader))
    test_suite.addTest(unittest.makeSuite(TestModels))
    test_suite.addTest(unittest.makeSuite(TestTrainer))
    test_suite.addTest(unittest.makeSuite(TestVisualizer))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Print summary
    print(f"\n{'='*50}")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"{'='*50}")

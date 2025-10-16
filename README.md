# Image Classification for Common Objects

A production-ready image classification system that can classify common everyday objects using state-of-the-art deep learning models. This project supports multiple model architectures including traditional CNNs, ResNet, and EfficientNet, with a user-friendly Streamlit web interface.

## Features

- **Multiple Model Architectures**: CNN, ResNet50, and EfficientNetB0
- **Modern Best Practices**: Type hints, comprehensive logging, configuration management
- **Interactive Web Interface**: Streamlit-based demo application
- **Comprehensive Evaluation**: Confusion matrices, per-class accuracy, training history
- **Production Ready**: Proper project structure, unit tests, documentation
- **Extensible Design**: Easy to add new models and datasets

## 📁 Project Structure

```
├── src/                    # Source code modules
│   ├── config.py          # Configuration management
│   ├── logger.py          # Logging utilities
│   ├── data_loader.py     # Data loading and preprocessing
│   ├── models.py          # Model definitions
│   ├── trainer.py         # Training utilities
│   └── visualizer.py      # Visualization and explainability
├── config/                # Configuration files
│   └── config.yaml        # Main configuration
├── web_app/               # Streamlit web interface
│   └── app.py             # Main Streamlit app
├── tests/                 # Unit tests
│   └── test_models.py     # Test cases
├── data/                  # Data directory (auto-created)
├── models/                # Saved models (auto-created)
├── logs/                  # Training logs (auto-created)
├── results/               # Results and visualizations (auto-created)
├── train.py               # Main training script
├── requirements.txt       # Python dependencies
├── .gitignore            # Git ignore rules
└── README.md             # This file
```

## 🛠️ Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/kryptologyst/Image-Classification-for-Common-Objects.git
   cd Image-Classification-for-Common-Objects
   ```

2. **Create a virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Quick Start

### Training a Model

Train a CNN model with default settings:
```bash
python train.py --model-type cnn
```

Train a ResNet model with custom epochs:
```bash
python train.py --model-type resnet --epochs 20 --batch-size 32
```

Train an EfficientNet model and save all results:
```bash
python train.py --model-type efficientnet --save-results
```

### Using the Web Interface

Launch the Streamlit web app:
```bash
streamlit run web_app/app.py
```

Then open your browser to `http://localhost:8501` and:
1. Select a model type (CNN, ResNet, or EfficientNet)
2. Upload an image or use a sample image
3. View the classification results and confidence scores

## Model Performance

The models are trained on the CIFAR-10 dataset with the following classes:
- airplane, automobile, bird, cat, deer
- dog, frog, horse, ship, truck

Typical performance metrics:
- **CNN**: ~75-80% accuracy
- **ResNet50**: ~85-90% accuracy  
- **EfficientNetB0**: ~88-92% accuracy

## 🔧 Configuration

The project uses YAML configuration files for easy customization. Key settings in `config/config.yaml`:

```yaml
model:
  epochs: 10
  batch_size: 64
  learning_rate: 0.001
  dropout_rate: 0.5

training:
  early_stopping:
    enabled: true
    patience: 5
  model_checkpoint:
    enabled: true
    save_best_only: true
```

## Testing

Run the test suite:
```bash
python -m pytest tests/ -v
```

Run tests with coverage:
```bash
python -m pytest tests/ --cov=src --cov-report=html
```

## Advanced Features

### Model Explainability

The project includes visualization tools for model interpretability:

```python
from src.visualizer import Visualizer
from src.config import Config

config = Config()
visualizer = Visualizer(config)

# Plot training history
visualizer.plot_training_history(history)

# Generate confusion matrix
visualizer.plot_confusion_matrix(y_true, y_pred)

# Visualize predictions
visualizer.visualize_predictions(images, true_labels, pred_labels, probabilities)
```

### Custom Model Training

```python
from src.models import ModelFactory
from src.trainer import Trainer
from src.config import Config

config = Config()
model = ModelFactory.create_model('resnet', config)
trainer = Trainer(config)

# Train with custom parameters
history = trainer.train_model(model, x_train, y_train, x_val, y_val)
```

## Deployment

### Local Deployment

1. Train your model:
   ```bash
   python train.py --model-type efficientnet --save-results
   ```

2. Launch the web interface:
   ```bash
   streamlit run web_app/app.py
   ```

### Production Deployment

For production deployment, consider:
- Using Docker containers
- Implementing model versioning
- Adding API endpoints with FastAPI/Flask
- Setting up monitoring and logging

## API Reference

### DataLoader Class

```python
class DataLoader:
    def load_cifar10() -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]
    def preprocess_data(x_train, x_test, y_train, y_test) -> Tuple[np.ndarray, ...]
    def get_data_info() -> Dict[str, Any]
```

### ModelFactory Class

```python
class ModelFactory:
    @staticmethod
    def create_model(model_type: str, config: Config) -> Model
    @staticmethod
    def get_available_models() -> list
```

### Trainer Class

```python
class Trainer:
    def train_model(model, x_train, y_train, x_val=None, y_val=None, model_name="model") -> History
    def evaluate_model(model, x_test, y_test, class_names) -> Dict[str, Any]
    def save_model(model, model_name) -> str
```

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make your changes and add tests
4. Run the test suite: `python -m pytest tests/`
5. Commit your changes: `git commit -am 'Add feature'`
6. Push to the branch: `git push origin feature-name`
7. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- CIFAR-10 dataset creators
- TensorFlow/Keras team for the deep learning framework
- Streamlit team for the web interface framework
- The open-source community for various libraries and tools

## Future Enhancements

- [ ] Support for custom datasets
- [ ] Model ensemble methods
- [ ] Real-time video classification
- [ ] Mobile app integration
- [ ] Cloud deployment templates
- [ ] Advanced data augmentation
- [ ] Hyperparameter optimization
- [ ] Model compression and quantization
# Image-Classification-for-Common-Objects

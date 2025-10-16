"""
Configuration management module for Image Classification Project.
"""

import yaml
import os
from typing import Dict, Any, List
from pathlib import Path


class Config:
    """Configuration manager for the image classification project."""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """
        Initialize configuration manager.
        
        Args:
            config_path: Path to the configuration YAML file
        """
        self.config_path = config_path
        self._config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        try:
            with open(self.config_path, 'r') as file:
                return yaml.safe_load(file)
        except FileNotFoundError:
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
        except yaml.YAMLError as e:
            raise ValueError(f"Error parsing YAML configuration: {e}")
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value by key.
        
        Args:
            key: Configuration key (supports dot notation like 'model.epochs')
            default: Default value if key not found
            
        Returns:
            Configuration value
        """
        keys = key.split('.')
        value = self._config
        
        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default
    
    def get_model_config(self) -> Dict[str, Any]:
        """Get model configuration."""
        return self.get('model', {})
    
    def get_data_config(self) -> Dict[str, Any]:
        """Get data configuration."""
        return self.get('data', {})
    
    def get_training_config(self) -> Dict[str, Any]:
        """Get training configuration."""
        return self.get('training', {})
    
    def get_class_names(self) -> List[str]:
        """Get class names."""
        return self.get('class_names', [])
    
    def create_directories(self) -> None:
        """Create necessary directories."""
        paths = self.get('paths', {})
        for path_name, path_value in paths.items():
            Path(path_value).mkdir(parents=True, exist_ok=True)
    
    def reload(self) -> None:
        """Reload configuration from file."""
        self._config = self._load_config()

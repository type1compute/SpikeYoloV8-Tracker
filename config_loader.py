"""
Configuration loader for Traffic_Monitoring SpikeYOLO project.
Handles loading and managing configuration from YAML files.
"""

import yaml
import os
import logging
from typing import Dict, Any, Optional, List
from pathlib import Path

logger = logging.getLogger(__name__)


class ConfigLoader:
    """
    Configuration loader for Traffic_Monitoring SpikeYOLO project.
    """
    
    def __init__(self, config_path: str = "config.yaml"):
        """
        Initialize configuration loader.
        
        Args:
            config_path: Path to configuration YAML file
        """
        self.config_path = config_path
        self.config = self._load_config()
        self._resolve_paths()
        
        logger.info(f"Configuration loaded from {config_path}")
        logger.info(f"Current environment: {self.get('current_environment')}")
    
    def _load_config(self) -> Dict[str, Any]:
        """
        Load configuration from YAML file.
        
        Returns:
            Configuration dictionary
        """
        try:
            with open(self.config_path, 'r') as file:
                config = yaml.safe_load(file)
            return config
        except FileNotFoundError:
            logger.error(f"Configuration file not found: {self.config_path}")
            raise
        except yaml.YAMLError as e:
            logger.error(f"Error parsing YAML configuration: {e}")
            raise
    
    def _resolve_paths(self):
        """
        Resolve environment-specific paths.
        """
        current_env = self.config.get('current_environment', 'vm')
        env_config = self.config.get('environments', {}).get(current_env, {})
        
        logger.info(f"Resolving paths for environment: {current_env}")
        logger.debug(f"Environment config keys: {list(env_config.keys())}")
        
        # Override paths with environment-specific ones
        for section in ['data', 'model']:
            if section in self.config:
                for key, value in env_config.items():
                    # Check if this key belongs to the section (e.g., checkpoint_dir belongs to model section)
                    # Map environment keys to config sections
                    section_mapping = {
                        'data_root': 'data',
                        'annotation_dir': 'data',
                        'checkpoint_dir': 'model',
                        'checkpoint_file': 'model',
                        'results_dir': 'model',
                        'logs_dir': 'model',
                        'checkpoint_path': 'model'
                    }
                    
                    target_section = section_mapping.get(key, section)
                    if target_section == section:

                        if key in self.config[section]:
                            old_value = self.config[section][key]
                            self.config[section][key] = value
                            logger.debug(f"Overrode {section}.{key}: {old_value} -> {value}")
                        else:
                            # Add missing key into the existing section
                            self.config.setdefault(section, {})
                            self.config[section][key] = value
                            logger.debug(f"Added {section}.{key}: {value}")

        # Create directories if they don't exist
        self._create_directories()
        
        # Log final checkpoint directory
        final_checkpoint_dir = self.get_checkpoint_dir()
        logger.info(f"Final checkpoint directory: {final_checkpoint_dir}")
    
    def _create_directories(self):
        """
        Create necessary directories if they don't exist.
        """
        directories_to_create = [
            self.get('model.checkpoint_dir'),
            self.get('model.results_dir'),
            self.get('model.logs_dir'),
        ]
        
        for directory in directories_to_create:
            if directory:
                Path(directory).mkdir(parents=True, exist_ok=True)
                logger.debug(f"Created directory: {directory}")
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value using dot notation.
        
        Args:
            key: Configuration key in dot notation (e.g., 'data.data_root')
            default: Default value if key not found
            
        Returns:
            Configuration value
        """
        keys = key.split('.')
        value = self.config
        
        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default
    
    def get_window_us(self) -> int:
        return self.get('architecture.time_window_us', 100000)
    def get_data_root(self) -> str:
        """Get data root directory path."""
        return self.get('data.data_root', './data')
    
    def get_annotation_dir(self) -> str:
        """Get annotation directory path."""
        return self.get('data.annotation_dir', './annotations')
    
    def get_checkpoint_dir(self) -> str:
        """Get checkpoint directory path."""
        return self.get('model.checkpoint_dir', './checkpoints')

    def get_checkpoint_path(self) -> str:
        """Get checkpoint file path."""
        checkpoint_dir = self.get('model.checkpoint_dir', './checkpoints')
        # Prefer 'checkpoint_path' (used in defaults), fall back to 'checkpoint_file' (used by some envs)
        filename = self.get('model.checkpoint_path', None) or self.get('model.checkpoint_file', 'model.pth')
        return os.path.join(checkpoint_dir, filename)
    
    def get_best_model_path(self) -> str:
        """Get best model file path."""
        checkpoint_dir = self.get('model.checkpoint_dir', './checkpoints')
        best_model_file = self.get('model.best_model_path', 'best_model.pth')
        return os.path.join(checkpoint_dir, best_model_file)
    
    def get_results_dir(self) -> str:
        """Get results directory path."""
        return self.get('model.results_dir', './results')
    
    def get_logs_dir(self) -> str:
        """Get logs directory path."""
        return self.get('model.logs_dir', './logs')
    
    def get_train_folders(self) -> List[str]:
        """Get list of training folder names."""
        return self.get('data.train_folders', [])
    
    def get_val_folders(self) -> List[str]:
        """Get list of validation folder names."""
        return self.get('data.val_folders', [])
    
    def get_test_folders(self) -> List[str]:
        """Get list of test folder names."""
        return self.get('data.test_folders', [])
    
    def get_class_names(self) -> List[str]:
        """Get list of class names."""
        # If using 3-class annotations, return 3-class names
        if self.get('data_processing.use_3_class_annotations', False):
            class_names = self.get('classes_3', [])
            if not class_names:
                # Fallback to default 3-class names if not in config
                # Mapping: class_id 0 = Pedestrian, class_id 1 = Vehicle, class_id 2 = Micro-mobility
                return ["Pedestrian", "Vehicle", "Micro-mobility"]
            return class_names
        # Otherwise return 8-class names
        class_names = self.get('classes_8', [])
        if not class_names:
            # Fallback to 'classes' for backward compatibility
            class_names = self.get('classes', [])
        return class_names
    
    def get_num_classes(self) -> int:
        """Get number of classes."""
        # If using 3-class annotations, override num_classes to 3
        if self.get('data_processing.use_3_class_annotations', False):
            return 3
        return self.get('architecture.num_classes', 8)
    
    def get_input_size(self) -> tuple[int, int]:
        """Get input size as tuple."""
        size = [self.get('data_processing.image_height', 720), self.get('data_processing.image_width', 1280)]
        return tuple(size)
    
    def get_time_steps(self) -> int:
        """Get number of time steps."""
        return self.get('architecture.time_steps', 8)
    
    def get_track_feature_dim(self) -> int:
        """Get tracking feature dimension."""
        return self.get('architecture.track_feature_dim', 128)
    
    def get_training_params(self) -> Dict[str, Any]:
        """Get training parameters."""
        return self.get('training', {})
    
    def get_data_processing_params(self) -> Dict[str, Any]:
        """Get data processing parameters."""
        return self.get('data_processing', {})
    
    def get_evaluation_params(self) -> Dict[str, Any]:
        """Get evaluation parameters."""
        return self.get('evaluation', {})
    
    def get_tracking_params(self) -> Dict[str, Any]:
        """Get tracking parameters."""
        return self.get('tracking', {})
    
    def get_logging_params(self) -> Dict[str, Any]:
        """Get logging parameters."""
        return self.get('logging', {})
    
    def get_device_params(self) -> Dict[str, Any]:
        """Get device parameters."""
        return self.get('device', {})
    
    def get_experiment_params(self) -> Dict[str, Any]:
        """Get experiment parameters."""
        return self.get('experiment', {})
    
    def get_data_path(self, path_type: str) -> str:
        """Get data path by type."""
        if path_type == 'hdf5':
            return self.get_data_root()
        elif path_type == 'annotations':
            return self.get_annotation_dir()
        elif path_type == 'root':
            return self.get_data_root()
        else:
            raise ValueError(f"Unknown path type: {path_type}")
    
    def set_environment(self, environment: str):
        """
        Set current environment and reload paths.
        
        Args:
            environment: Environment name ('local', 'vm', 'docker')
        """
        self.config['current_environment'] = environment
        self._resolve_paths()
        logger.info(f"Switched to environment: {environment}")
    
    def update_config(self, updates: Dict[str, Any]):
        """
        Update configuration with new values.
        
        Args:
            updates: Dictionary of configuration updates
        """
        def update_nested_dict(d, u):
            for k, v in u.items():
                if isinstance(v, dict):
                    d[k] = update_nested_dict(d.get(k, {}), v)
                else:
                    d[k] = v
            return d
        
        self.config = update_nested_dict(self.config, updates)
        self._resolve_paths()
        logger.info("Configuration updated")
    
    def save_config(self, output_path: Optional[str] = None):
        """
        Save current configuration to file.
        
        Args:
            output_path: Output file path (defaults to original config path)
        """
        if output_path is None:
            output_path = self.config_path
        
        with open(output_path, 'w') as file:
            yaml.dump(self.config, file, default_flow_style=False, indent=2)
        
        logger.info(f"Configuration saved to {output_path}")
    
    def print_config(self):
        """Print current configuration."""
        print("=" * 60)
        print("CURRENT CONFIGURATION")
        print("=" * 60)
        
        print(f"Environment: {self.get('current_environment')}")
        print(f"Data Root: {self.get_data_root()}")
        print(f"Annotation Dir: {self.get_annotation_dir()}")
        print(f"Checkpoint Path: {self.get_checkpoint_path()}")
        print(f"Results Dir: {self.get_results_dir()}")
        print(f"Logs Dir: {self.get_logs_dir()}")
        
        print(f"\nArchitecture:")
        print(f"  Classes: {self.get_num_classes()}")
        print(f"  Input Size: {self.get_input_size()}")
        print(f"  Time Steps: {self.get_time_steps()}")
        print(f"  Track Feature Dim: {self.get_track_feature_dim()}")
        
        print(f"\nTraining:")
        training_params = self.get_training_params()
        for key, value in training_params.items():
            print(f"  {key}: {value}")
        
        print("=" * 60)


def load_config(config_path: str = "config.yaml") -> ConfigLoader:
    """
    Load configuration from file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        ConfigLoader instance
    """
    return ConfigLoader(config_path)


def create_default_config(output_path: str = "config_default.yaml"):
    """
    Create a default configuration file.
    
    Args:
        output_path: Output file path
    """
    default_config = {
        'data': {
            'data_root': './data',
            'annotation_dir': './annotations',
            'train_folders': ['train'],
            'val_folders': ['val'],
            'test_folders': ['test']
        },
        'model': {
            'checkpoint_dir': './checkpoints',
            'checkpoint_path': 'model.pth',
            'best_model_path': 'best_model.pth',
            'results_dir': './results',
            'logs_dir': './logs'
        },
        'architecture': {
            'num_classes': 8,
            'input_size': [720, 1280],
            'time_steps': 8,
            'track_feature_dim': 128,
            'model_name': 'eTraM_SpikeYOLO'
        },
        'training': {
            'epochs': 5,
            'batch_size': 1,
            'learning_rate': 0.001,
            'weight_decay': 0.0005
        },
        'current_environment': 'local',
        'classes': ['car', 'truck', 'bus', 'motorcycle', 'bicycle', 'person', 'traffic_light', 'stop_sign']
    }
    
    with open(output_path, 'w') as file:
        yaml.dump(default_config, file, default_flow_style=False, indent=2)
    
    print(f"Default configuration created: {output_path}")


if __name__ == "__main__":
    # Test the configuration loader
    try:
        config = load_config()
        config.print_config()
        
        # Test environment switching
        print("\nTesting environment switching...")
        config.set_environment('local')
        print(f"Local data root: {config.get_data_root()}")
        
        config.set_environment('vm')
        print(f"VM data root: {config.get_data_root()}")
        
    except Exception as e:
        print(f"Error testing configuration: {e}")
        print("Creating default configuration...")
        create_default_config()


# Create a global instance for easy import
config = ConfigLoader()

# Alias for backward compatibility
Config = ConfigLoader
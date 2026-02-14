"""
Configuration management utilities.
"""

import yaml
from pathlib import Path
from typing import Dict, Any


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    Handles inheritance from base config.
    
    Args:
        config_path: Path to config file
        
    Returns:
        Configuration dictionary
    """
    config_path = Path(config_path)
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Handle base config inheritance
    if '_base_' in config:
        base_path = config_path.parent / config['_base_']
        with open(base_path, 'r') as f:
            base_config = yaml.safe_load(f)
        
        # Merge configs (child overrides base)
        merged_config = deep_merge(base_config, config)
        # Remove the _base_ key
        merged_config.pop('_base_', None)
        return merged_config
    
    return config


def deep_merge(base: dict, override: dict) -> dict:
    """
    Deep merge two dictionaries.
    
    Args:
        base: Base dictionary
        override: Override dictionary
        
    Returns:
        Merged dictionary
    """
    result = base.copy()
    
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    
    return result


def save_config(config: Dict[str, Any], save_path: str):
    """
    Save configuration to YAML file.
    
    Args:
        config: Configuration dictionary
        save_path: Path to save config
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(save_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, indent=2)


def print_config(config: Dict[str, Any], indent: int = 0):
    """
    Pretty print configuration.
    
    Args:
        config: Configuration dictionary
        indent: Indentation level
    """
    for key, value in config.items():
        if isinstance(value, dict):
            print(" " * indent + f"{key}:")
            print_config(value, indent + 2)
        else:
            print(" " * indent + f"{key}: {value}")

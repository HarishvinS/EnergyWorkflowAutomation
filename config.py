"""
Configuration settings for the Emissions Compliance Forecasting Module.

This module contains all configuration parameters for data processing,
model training, and output generation.
"""

import os
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
MODELS_DIR = PROJECT_ROOT / "models"

# Ensure output directories exist
OUTPUT_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)

# Data file configuration
DATA_FILE = "quarterly-emissions-e59aa51f-1169-445d-918e-fb306c4f282a.csv"
DATA_PATH = PROJECT_ROOT / DATA_FILE

# Model configuration
MODEL_CONFIG = {
    "random_forest": {
        "n_estimators": 200,
        "max_depth": 15,
        "min_samples_split": 10,
        "min_samples_leaf": 5,
        "max_features": "sqrt",
        "random_state": 42,
        "n_jobs": -1
    }
}

# Feature configuration
NUMERICAL_FEATURES = [
    'Gross Load (MWh)',
    'Heat Input (mmBtu)',
    'Sum of the Operating Time',
    'Load_to_Heat_Ratio',
    'Heat_Efficiency',
    'Operating_Intensity'
]

CATEGORICAL_FEATURES = [
    'Primary Fuel Type',
    'Unit Type',
    'Quarter'
]

TARGET_VARIABLES = [
    'SO2 Rate (lbs/mmBtu)',
    'NOx Rate (lbs/mmBtu)',
    'CO2 Rate (short tons/mmBtu)'
]

# Data preprocessing configuration
PREPROCESSING_CONFIG = {
    "outlier_removal": {
        "method": "iqr",
        "iqr_multiplier": 3.0  # More conservative than 1.5
    },
    "missing_values": {
        "numerical_strategy": "median",
        "categorical_strategy": "mode"
    },
    "scaling": {
        "method": "robust",  # or "standard"
        "with_centering": True,
        "with_scaling": True
    }
}

# Training configuration
TRAINING_CONFIG = {
    "test_size": 0.2,
    "random_state": 42,
    "stratify_column": "Quarter",
    "cross_validation": {
        "cv_folds": 5,
        "scoring": "r2"
    }
}

# Output configuration
OUTPUT_CONFIG = {
    "predictions_file": "predictions.csv",
    "summary_file": "weekly_summary.txt",
    "nox_plot_file": "nox_predictions.png",
    "comparison_plot_file": "model_predictions_comparison.png",
    "feature_importance_file": "feature_importance.png"
}

# Logging configuration
LOGGING_CONFIG = {
    "level": "INFO",
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "log_file": "emissions_forecasting.log"
}

# Performance thresholds
PERFORMANCE_THRESHOLDS = {
    "r2_excellent": 0.8,
    "r2_good": 0.6,
    "r2_moderate": 0.3,
    "r2_poor": 0.0,
    "overfitting_threshold": 0.1  # Train R² - Test R² difference
}

# EPA compliance thresholds (example values - should be updated with actual regulations)
EPA_THRESHOLDS = {
    "SO2 Rate (lbs/mmBtu)": {
        "warning": 0.1,
        "violation": 0.2
    },
    "NOx Rate (lbs/mmBtu)": {
        "warning": 0.15,
        "violation": 0.25
    },
    "CO2 Rate (short tons/mmBtu)": {
        "warning": 0.08,
        "violation": 0.1
    }
}

# Visualization configuration
PLOT_CONFIG = {
    "figure_size": (12, 8),
    "dpi": 300,
    "style": "seaborn-v0_8",
    "color_palette": "viridis",
    "font_size": 12
}

def get_config():
    """
    Get the complete configuration dictionary.
    
    Returns:
        dict: Complete configuration settings
    """
    return {
        "paths": {
            "project_root": PROJECT_ROOT,
            "data_dir": DATA_DIR,
            "output_dir": OUTPUT_DIR,
            "models_dir": MODELS_DIR,
            "data_path": DATA_PATH
        },
        "features": {
            "numerical": NUMERICAL_FEATURES,
            "categorical": CATEGORICAL_FEATURES,
            "targets": TARGET_VARIABLES
        },
        "model": MODEL_CONFIG,
        "preprocessing": PREPROCESSING_CONFIG,
        "training": TRAINING_CONFIG,
        "output": OUTPUT_CONFIG,
        "logging": LOGGING_CONFIG,
        "performance": PERFORMANCE_THRESHOLDS,
        "epa": EPA_THRESHOLDS,
        "plotting": PLOT_CONFIG
    }

def validate_config():
    """
    Validate configuration settings and check for required files.
    
    Returns:
        bool: True if configuration is valid
        
    Raises:
        FileNotFoundError: If required data file is missing
        ValueError: If configuration values are invalid
    """
    # Check if data file exists
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Data file not found: {DATA_PATH}")
    
    # Validate model configuration
    rf_config = MODEL_CONFIG["random_forest"]
    if rf_config["n_estimators"] <= 0:
        raise ValueError("n_estimators must be positive")
    
    if rf_config["max_depth"] <= 0:
        raise ValueError("max_depth must be positive")
    
    # Validate training configuration
    if not 0 < TRAINING_CONFIG["test_size"] < 1:
        raise ValueError("test_size must be between 0 and 1")
    
    # Validate performance thresholds
    thresholds = PERFORMANCE_THRESHOLDS
    if not all(0 <= v <= 1 for v in [thresholds["r2_excellent"], thresholds["r2_good"], 
                                     thresholds["r2_moderate"]]):
        raise ValueError("R² thresholds must be between 0 and 1")
    
    return True

if __name__ == "__main__":
    # Validate configuration when run as script
    try:
        validate_config()
        print("Configuration validation passed!")
        print(f"Data file found: {DATA_PATH}")
        print(f"Output directory: {OUTPUT_DIR}")
    except Exception as e:
        print(f"Configuration validation failed: {e}")

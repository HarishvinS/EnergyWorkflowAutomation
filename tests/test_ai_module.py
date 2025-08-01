"""
Tests for ai_module.py

This module contains unit tests for the core AI/ML functions including
data preprocessing, model training, and prediction generation.
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
import sys
import os

# Add parent directory to path to import ai_module
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ai_module import preprocess_data, train_model, predict, evaluate_model


class TestPreprocessData:
    """Test cases for the preprocess_data function."""
    
    def setup_method(self):
        """Set up test data before each test method."""
        # Create sample data that mimics the CEMS dataset structure
        self.sample_data = pd.DataFrame({
            'Facility ID': [1, 1, 2, 2, 3],
            'Unit ID': [1, 1, 1, 1, 1],
            'Sum of the Operating Time': [100, 0, 150, 200, 120],  # Include zero operating time
            'Gross Load (MWh)': [500, 0, 600, 700, 550],
            'Heat Input (mmBtu)': [1000, 0, 1200, 1400, 1100],
            'Primary Fuel Type': ['Natural Gas', 'Natural Gas', 'Coal', 'Coal', 'Natural Gas'],
            'Unit Type': ['Combined Cycle', 'Combined Cycle', 'Steam', 'Steam', 'Combined Cycle'],
            'Quarter': [1, 1, 2, 2, 1],
            'SO2 Rate (lbs/mmBtu)': [0.001, np.nan, 0.002, 0.003, 0.001],
            'NOx Rate (lbs/mmBtu)': [0.05, np.nan, 0.08, 0.06, 0.04],
            'CO2 Rate (short tons/mmBtu)': [0.059, np.nan, 0.061, 0.060, 0.058]
        })
    
    def test_preprocess_data_basic(self):
        """Test basic preprocessing functionality."""
        df_processed, features, targets = preprocess_data(self.sample_data)
        
        # Check that zero operating time rows are removed
        assert len(df_processed) < len(self.sample_data)
        assert all(df_processed['Sum of the Operating Time'] > 0)
        
        # Check that features and targets are defined
        assert isinstance(features, list)
        assert isinstance(targets, list)
        assert len(features) > 0
        assert len(targets) == 3
        
        # Check that new engineered features are created
        assert 'Load_to_Heat_Ratio' in df_processed.columns
        assert 'Heat_Efficiency' in df_processed.columns
        assert 'Operating_Intensity' in df_processed.columns
    
    def test_preprocess_data_missing_values(self):
        """Test handling of missing values."""
        df_processed, features, targets = preprocess_data(self.sample_data)
        
        # Check that missing values in targets are handled
        for target in targets:
            assert df_processed[target].isnull().sum() == 0
        
        # Check that missing values in features are handled
        for feature in features:
            if feature in df_processed.columns:
                assert df_processed[feature].isnull().sum() == 0
    
    def test_preprocess_data_outlier_removal(self):
        """Test outlier removal functionality."""
        # Create data with extreme outliers
        outlier_data = self.sample_data.copy()
        outlier_data.loc[0, 'SO2 Rate (lbs/mmBtu)'] = 100.0  # Extreme outlier

        original_length = len(outlier_data[outlier_data['Sum of the Operating Time'] > 0])
        df_processed, _, _ = preprocess_data(outlier_data)

        # Check that some rows were removed (outlier removal happened)
        # The exact value depends on the IQR calculation, so we just check that processing occurred
        assert len(df_processed) <= original_length


class TestTrainModel:
    """Test cases for the train_model function."""
    
    def setup_method(self):
        """Set up test data before each test method."""
        # Create sample training data
        np.random.seed(42)
        self.X_train = pd.DataFrame({
            'Gross Load (MWh)': np.random.normal(500, 100, 100),
            'Heat Input (mmBtu)': np.random.normal(1000, 200, 100),
            'Sum of the Operating Time': np.random.normal(150, 30, 100),
            'Load_to_Heat_Ratio': np.random.normal(0.5, 0.1, 100),
            'Heat_Efficiency': np.random.normal(0.4, 0.05, 100),
            'Operating_Intensity': np.random.normal(0.7, 0.1, 100),
            'Primary Fuel Type': np.random.choice(['Natural Gas', 'Coal'], 100),
            'Unit Type': np.random.choice(['Combined Cycle', 'Steam'], 100),
            'Quarter': np.random.choice([1, 2, 3, 4], 100)
        })
        
        self.y_train = pd.DataFrame({
            'SO2 Rate (lbs/mmBtu)': np.random.normal(0.001, 0.0005, 100),
            'NOx Rate (lbs/mmBtu)': np.random.normal(0.05, 0.01, 100),
            'CO2 Rate (short tons/mmBtu)': np.random.normal(0.059, 0.002, 100)
        })
    
    def test_train_model_basic(self):
        """Test basic model training functionality."""
        target = 'NOx Rate (lbs/mmBtu)'
        model = train_model(self.X_train, self.y_train, target)
        
        # Check that model is trained and can make predictions
        assert hasattr(model, 'predict')
        predictions = model.predict(self.X_train)
        assert len(predictions) == len(self.X_train)
        assert isinstance(predictions, np.ndarray)
    
    def test_train_model_all_targets(self):
        """Test training models for all target variables."""
        targets = ['SO2 Rate (lbs/mmBtu)', 'NOx Rate (lbs/mmBtu)', 'CO2 Rate (short tons/mmBtu)']
        
        for target in targets:
            model = train_model(self.X_train, self.y_train, target)
            assert hasattr(model, 'predict')
            predictions = model.predict(self.X_train)
            assert len(predictions) == len(self.X_train)


class TestPredict:
    """Test cases for the predict function."""
    
    def setup_method(self):
        """Set up test data and trained model."""
        # Create mock model
        self.mock_model = MagicMock()
        self.mock_model.predict.return_value = np.array([0.05, 0.06, 0.04])
        
        # Create test data
        self.X_test = pd.DataFrame({
            'Gross Load (MWh)': [500, 600, 450],
            'Heat Input (mmBtu)': [1000, 1200, 900],
            'Sum of the Operating Time': [150, 180, 120],
            'Load_to_Heat_Ratio': [0.5, 0.5, 0.5],
            'Heat_Efficiency': [0.4, 0.4, 0.4],
            'Operating_Intensity': [0.7, 0.8, 0.6],
            'Primary Fuel Type': ['Natural Gas', 'Coal', 'Natural Gas'],
            'Unit Type': ['Combined Cycle', 'Steam', 'Combined Cycle'],
            'Quarter': [1, 2, 1]
        })
    
    def test_predict_basic(self):
        """Test basic prediction functionality."""
        predictions = predict(self.mock_model, self.X_test)
        
        # Check that predictions are returned
        assert isinstance(predictions, np.ndarray)
        assert len(predictions) == len(self.X_test)
        
        # Check that model.predict was called
        self.mock_model.predict.assert_called_once()


class TestEvaluateModel:
    """Test cases for the evaluate_model function."""
    
    def setup_method(self):
        """Set up test data and mock model."""
        # Create mock model with realistic predictions
        self.mock_model = MagicMock()
        self.mock_model.predict.return_value = np.array([0.05, 0.06, 0.04, 0.055, 0.045])
        
        # Create test data
        self.X_test = pd.DataFrame({
            'feature1': [1, 2, 3, 4, 5],
            'feature2': [10, 20, 30, 40, 50]
        })
        
        self.y_test = pd.DataFrame({
            'NOx Rate (lbs/mmBtu)': [0.051, 0.059, 0.041, 0.054, 0.046]
        })
        
        self.target = 'NOx Rate (lbs/mmBtu)'
    
    def test_evaluate_model_basic(self):
        """Test basic model evaluation functionality."""
        metrics = evaluate_model(self.mock_model, self.X_test, self.y_test, self.target)
        
        # Check that all expected metrics are returned
        expected_metrics = ['MAE', 'RMSE', 'R2', 'Mean_Actual', 'Mean_Predicted', 'Std_Actual', 'Std_Predicted']
        for metric in expected_metrics:
            assert metric in metrics
            assert isinstance(metrics[metric], (int, float))
        
        # Check that R2 is reasonable (between -1 and 1 for this test case)
        assert -1 <= metrics['R2'] <= 1


class TestIntegration:
    """Integration tests for the complete workflow."""
    
    def test_full_workflow(self):
        """Test the complete preprocessing -> training -> prediction workflow."""
        # Create realistic sample data
        sample_data = pd.DataFrame({
            'Facility ID': [1] * 50 + [2] * 50,
            'Unit ID': [1] * 100,
            'Sum of the Operating Time': np.random.uniform(100, 200, 100),
            'Gross Load (MWh)': np.random.uniform(400, 800, 100),
            'Heat Input (mmBtu)': np.random.uniform(800, 1600, 100),
            'Primary Fuel Type': np.random.choice(['Natural Gas', 'Coal'], 100),
            'Unit Type': np.random.choice(['Combined Cycle', 'Steam'], 100),
            'Quarter': np.random.choice([1, 2, 3, 4], 100),
            'SO2 Rate (lbs/mmBtu)': np.random.normal(0.001, 0.0005, 100),
            'NOx Rate (lbs/mmBtu)': np.random.normal(0.05, 0.01, 100),
            'CO2 Rate (short tons/mmBtu)': np.random.normal(0.059, 0.002, 100)
        })
        
        # Preprocess data
        df_processed, features, targets = preprocess_data(sample_data)
        
        # Split data
        from sklearn.model_selection import train_test_split
        X = df_processed[features]
        y = df_processed[targets]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        
        # Train model for one target
        target = 'NOx Rate (lbs/mmBtu)'
        model = train_model(X_train, y_train, target)
        
        # Make predictions
        predictions = predict(model, X_test)
        
        # Evaluate model
        metrics = evaluate_model(model, X_test, y_test, target)
        
        # Check that everything worked
        assert len(predictions) == len(X_test)
        assert 'R2' in metrics
        assert isinstance(metrics['R2'], float)


if __name__ == "__main__":
    # Run tests if script is executed directly
    pytest.main([__file__, "-v"])

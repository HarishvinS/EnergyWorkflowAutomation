"""
Enhanced model evaluation and validation module.

This module provides comprehensive model evaluation capabilities including
cross-validation, advanced metrics, and detailed visualizations.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import cross_val_score, cross_validate, TimeSeriesSplit
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    mean_absolute_percentage_error, explained_variance_score
)
from sklearn.inspection import permutation_importance
import logging
from typing import Dict, List, Tuple, Any
import warnings

# Set up logging
logger = logging.getLogger(__name__)

class ModelEvaluator:
    """
    Comprehensive model evaluation class with advanced metrics and visualizations.
    """
    
    def __init__(self, models: Dict[str, Any], X_test: pd.DataFrame, y_test: pd.DataFrame):
        """
        Initialize the evaluator with trained models and test data.
        
        Args:
            models: Dictionary of trained models {target: model}
            X_test: Test features
            y_test: Test targets
        """
        self.models = models
        self.X_test = X_test
        self.y_test = y_test
        self.results = {}
        self.predictions = {}
        
    def evaluate_all_models(self) -> Dict[str, Dict[str, float]]:
        """
        Evaluate all models with comprehensive metrics.
        
        Returns:
            Dictionary of evaluation results for each target
        """
        logger.info("Starting comprehensive model evaluation")
        
        for target, model in self.models.items():
            logger.info(f"Evaluating model for {target}")
            
            # Generate predictions
            y_pred = model.predict(self.X_test)
            self.predictions[target] = y_pred
            
            # Calculate comprehensive metrics
            metrics = self._calculate_comprehensive_metrics(
                self.y_test[target], y_pred, target
            )
            
            self.results[target] = metrics
            
        return self.results
    
    def _calculate_comprehensive_metrics(self, y_true: pd.Series, y_pred: np.ndarray, 
                                       target: str) -> Dict[str, float]:
        """
        Calculate comprehensive evaluation metrics.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            target: Target variable name
            
        Returns:
            Dictionary of metrics
        """
        metrics = {}
        
        # Basic regression metrics
        metrics['MAE'] = mean_absolute_error(y_true, y_pred)
        metrics['MSE'] = mean_squared_error(y_true, y_pred)
        metrics['RMSE'] = np.sqrt(metrics['MSE'])
        metrics['R2'] = r2_score(y_true, y_pred)
        metrics['Explained_Variance'] = explained_variance_score(y_true, y_pred)
        
        # Percentage-based metrics (handle division by zero)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                metrics['MAPE'] = mean_absolute_percentage_error(y_true, y_pred)
            except:
                metrics['MAPE'] = np.inf
        
        # Custom metrics
        metrics['Mean_Actual'] = float(y_true.mean())
        metrics['Mean_Predicted'] = float(y_pred.mean())
        metrics['Std_Actual'] = float(y_true.std())
        metrics['Std_Predicted'] = float(y_pred.std())
        
        # Residual analysis
        residuals = y_true - y_pred
        metrics['Mean_Residual'] = float(residuals.mean())
        metrics['Std_Residual'] = float(residuals.std())
        metrics['Max_Residual'] = float(residuals.abs().max())
        
        # Prediction intervals
        metrics['Prediction_Range'] = float(y_pred.max() - y_pred.min())
        metrics['Actual_Range'] = float(y_true.max() - y_true.min())
        
        return metrics
    
    def cross_validate_models(self, X_train: pd.DataFrame, y_train: pd.DataFrame, 
                            cv_folds: int = 5) -> Dict[str, Dict[str, Any]]:
        """
        Perform cross-validation for all models.
        
        Args:
            X_train: Training features
            y_train: Training targets
            cv_folds: Number of cross-validation folds
            
        Returns:
            Cross-validation results for each target
        """
        logger.info(f"Performing {cv_folds}-fold cross-validation")
        
        cv_results = {}
        scoring = ['r2', 'neg_mean_absolute_error', 'neg_mean_squared_error']
        
        for target, model in self.models.items():
            logger.info(f"Cross-validating model for {target}")
            
            # Perform cross-validation
            cv_scores = cross_validate(
                model, X_train, y_train[target], 
                cv=cv_folds, scoring=scoring, return_train_score=True
            )
            
            # Process results
            cv_results[target] = {
                'test_r2_mean': cv_scores['test_r2'].mean(),
                'test_r2_std': cv_scores['test_r2'].std(),
                'test_mae_mean': -cv_scores['test_neg_mean_absolute_error'].mean(),
                'test_mae_std': cv_scores['test_neg_mean_absolute_error'].std(),
                'test_rmse_mean': np.sqrt(-cv_scores['test_neg_mean_squared_error'].mean()),
                'train_r2_mean': cv_scores['train_r2'].mean(),
                'train_r2_std': cv_scores['train_r2'].std(),
                'overfitting_score': cv_scores['train_r2'].mean() - cv_scores['test_r2'].mean()
            }
            
        return cv_results
    
    def calculate_feature_importance(self, X_train: pd.DataFrame, y_train: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """
        Calculate feature importance for all models using permutation importance.
        
        Args:
            X_train: Training features
            y_train: Training targets
            
        Returns:
            Feature importance DataFrames for each target
        """
        logger.info("Calculating feature importance using permutation importance")
        
        importance_results = {}
        
        for target, model in self.models.items():
            logger.info(f"Calculating feature importance for {target}")
            
            # Calculate permutation importance
            perm_importance = permutation_importance(
                model, X_train, y_train[target], 
                n_repeats=10, random_state=42, n_jobs=-1
            )
            
            # Create DataFrame
            importance_df = pd.DataFrame({
                'feature': X_train.columns,
                'importance_mean': perm_importance.importances_mean,
                'importance_std': perm_importance.importances_std
            }).sort_values('importance_mean', ascending=False)
            
            importance_results[target] = importance_df
            
        return importance_results
    
    def create_comprehensive_visualizations(self, save_plots: bool = True) -> None:
        """
        Create comprehensive visualizations for model evaluation.
        
        Args:
            save_plots: Whether to save plots to files
        """
        logger.info("Creating comprehensive visualizations")
        
        # Set style
        plt.style.use('seaborn-v0_8')
        
        # 1. Predictions vs Actual (all targets)
        self._plot_predictions_vs_actual(save_plots)
        
        # 2. Residual plots
        self._plot_residuals(save_plots)
        
        # 3. Model performance comparison
        self._plot_performance_comparison(save_plots)
        
        # 4. Prediction distributions
        self._plot_prediction_distributions(save_plots)
    
    def _plot_predictions_vs_actual(self, save_plots: bool) -> None:
        """Create predictions vs actual plots for all targets."""
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        fig.suptitle('Predicted vs Actual Emission Rates', fontsize=16)
        
        targets = list(self.models.keys())
        
        for i, target in enumerate(targets):
            if target in self.predictions:
                y_true = self.y_test[target]
                y_pred = self.predictions[target]
                
                # Scatter plot
                axes[i].scatter(y_true, y_pred, alpha=0.6, s=30)
                
                # Perfect prediction line
                min_val = min(y_true.min(), y_pred.min())
                max_val = max(y_true.max(), y_pred.max())
                axes[i].plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
                
                # Formatting
                axes[i].set_xlabel(f'Actual {target}')
                axes[i].set_ylabel(f'Predicted {target}')
                axes[i].set_title(f'{target}\nR² = {self.results[target]["R2"]:.3f}')
                axes[i].grid(True, alpha=0.3)
                axes[i].legend()
        
        plt.tight_layout()
        if save_plots:
            plt.savefig('comprehensive_predictions_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def _plot_residuals(self, save_plots: bool) -> None:
        """Create residual plots for all targets."""
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle('Residual Analysis', fontsize=16)
        
        targets = list(self.models.keys())
        
        for i, target in enumerate(targets):
            if target in self.predictions:
                y_true = self.y_test[target]
                y_pred = self.predictions[target]
                residuals = y_true - y_pred
                
                # Residuals vs Predicted
                axes[0, i].scatter(y_pred, residuals, alpha=0.6, s=30)
                axes[0, i].axhline(y=0, color='r', linestyle='--')
                axes[0, i].set_xlabel(f'Predicted {target}')
                axes[0, i].set_ylabel('Residuals')
                axes[0, i].set_title(f'Residuals vs Predicted - {target}')
                axes[0, i].grid(True, alpha=0.3)
                
                # Residual distribution
                axes[1, i].hist(residuals, bins=30, alpha=0.7, edgecolor='black')
                axes[1, i].axvline(x=0, color='r', linestyle='--')
                axes[1, i].set_xlabel('Residuals')
                axes[1, i].set_ylabel('Frequency')
                axes[1, i].set_title(f'Residual Distribution - {target}')
                axes[1, i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        if save_plots:
            plt.savefig('residual_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def _plot_performance_comparison(self, save_plots: bool) -> None:
        """Create performance comparison chart."""
        metrics_to_plot = ['R2', 'MAE', 'RMSE']
        targets = list(self.results.keys())
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle('Model Performance Comparison', fontsize=16)
        
        for i, metric in enumerate(metrics_to_plot):
            values = [self.results[target][metric] for target in targets]
            
            bars = axes[i].bar(targets, values, alpha=0.7)
            axes[i].set_title(f'{metric} by Target')
            axes[i].set_ylabel(metric)
            axes[i].tick_params(axis='x', rotation=45)
            axes[i].grid(True, alpha=0.3)
            
            # Add value labels on bars
            for bar, value in zip(bars, values):
                axes[i].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                           f'{value:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        if save_plots:
            plt.savefig('performance_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def _plot_prediction_distributions(self, save_plots: bool) -> None:
        """Create prediction distribution plots."""
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        fig.suptitle('Prediction vs Actual Distributions', fontsize=16)
        
        targets = list(self.models.keys())
        
        for i, target in enumerate(targets):
            if target in self.predictions:
                y_true = self.y_test[target]
                y_pred = self.predictions[target]
                
                # Plot distributions
                axes[i].hist(y_true, bins=30, alpha=0.7, label='Actual', density=True)
                axes[i].hist(y_pred, bins=30, alpha=0.7, label='Predicted', density=True)
                
                axes[i].set_xlabel(target)
                axes[i].set_ylabel('Density')
                axes[i].set_title(f'Distribution Comparison - {target}')
                axes[i].legend()
                axes[i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        if save_plots:
            plt.savefig('distribution_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def generate_evaluation_report(self) -> str:
        """
        Generate a comprehensive evaluation report.
        
        Returns:
            Formatted evaluation report as string
        """
        report = "# Comprehensive Model Evaluation Report\n\n"
        report += f"Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        
        # Overall performance summary
        report += "## Overall Performance Summary\n\n"
        for target, metrics in self.results.items():
            report += f"### {target}\n"
            report += f"- R² Score: {metrics['R2']:.4f}\n"
            report += f"- Mean Absolute Error: {metrics['MAE']:.4f}\n"
            report += f"- Root Mean Square Error: {metrics['RMSE']:.4f}\n"
            report += f"- Explained Variance: {metrics['Explained_Variance']:.4f}\n"
            
            # Performance assessment
            r2 = metrics['R2']
            if r2 >= 0.8:
                assessment = "[EXCELLENT]"
            elif r2 >= 0.6:
                assessment = "[GOOD]"
            elif r2 >= 0.3:
                assessment = "[MODERATE]"
            else:
                assessment = "[POOR]"
            
            report += f"- Performance Assessment: {assessment}\n\n"
        
        # Detailed metrics
        report += "## Detailed Metrics\n\n"
        for target, metrics in self.results.items():
            report += f"### {target} - Detailed Analysis\n"
            report += f"- Mean Actual: {metrics['Mean_Actual']:.6f}\n"
            report += f"- Mean Predicted: {metrics['Mean_Predicted']:.6f}\n"
            report += f"- Std Actual: {metrics['Std_Actual']:.6f}\n"
            report += f"- Std Predicted: {metrics['Std_Predicted']:.6f}\n"
            report += f"- Mean Residual: {metrics['Mean_Residual']:.6f}\n"
            report += f"- Max Absolute Residual: {metrics['Max_Residual']:.6f}\n\n"
        
        return report

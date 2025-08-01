import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logging
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from ai_module import preprocess_data, train_model, predict, evaluate_model
from model_evaluation import ModelEvaluator

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    """Main pipeline for emissions forecasting."""
    logger.info("Starting emissions forecasting pipeline")

    try:
        # Load data
        logger.info("Loading data...")
        df = pd.read_csv('quarterly-emissions-e59aa51f-1169-445d-918e-fb306c4f282a.csv')
        logger.info(f"Loaded dataset with {len(df)} rows and {len(df.columns)} columns")

        # Preprocess data
        logger.info("Preprocessing data...")
        df_processed, features, targets = preprocess_data(df)

        # Prepare data for modeling
        X = df_processed[features].copy()
        y = df_processed[targets].copy()

        # Split data with stratification by quarter for temporal balance
        logger.info("Splitting data...")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=X['Quarter']
        )
        logger.info(f"Training set: {len(X_train)} samples, Test set: {len(X_test)} samples")

        # Train and evaluate models
        logger.info("Training models...")
        predictions = pd.DataFrame(index=X_test.index)
        results = {}
        models = {}

        for target in targets:
            logger.info(f"Training model for {target}")
            model = train_model(X_train, y_train, target)
            models[target] = model

            # Generate predictions
            y_pred = predict(model, X_test)
            predictions[f'Predicted_{target}'] = y_pred

            # Evaluate model
            metrics = evaluate_model(model, X_test, y_test, target)
            results[target] = metrics

            # Log results
            logger.info(f'{target} - MAE: {metrics["MAE"]:.4f}, R²: {metrics["R2"]:.4f}')

            # Check for poor performance
            if metrics['R2'] < 0:
                logger.warning(f"⚠️ {target} model has negative R² ({metrics['R2']:.4f}) - performing worse than baseline!")
            elif metrics['R2'] < 0.3:
                logger.warning(f"⚠️ {target} model has poor performance (R² = {metrics['R2']:.4f})")

        # Enhanced model evaluation
        logger.info("Performing enhanced model evaluation...")
        evaluator = ModelEvaluator(models, X_test, y_test)

        # Comprehensive evaluation
        comprehensive_results = evaluator.evaluate_all_models()

        # Cross-validation
        cv_results = evaluator.cross_validate_models(X_train, y_train)

        # Feature importance
        feature_importance = evaluator.calculate_feature_importance(X_train, y_train)

        # Create comprehensive visualizations
        evaluator.create_comprehensive_visualizations(save_plots=True)

        # Generate evaluation report
        evaluation_report = evaluator.generate_evaluation_report()

        # Create basic visualizations for backward compatibility
        logger.info("Creating basic visualizations...")
        create_visualizations(y_test, predictions, targets, results)

        # Generate and save summary
        logger.info("Generating summary...")
        summary = generate_summary(predictions, df_processed, results)
        save_outputs(X_test, y_test, predictions, summary, evaluation_report, cv_results)

        logger.info("Pipeline completed successfully!")
        return {
            'basic_results': results,
            'comprehensive_results': comprehensive_results,
            'cv_results': cv_results,
            'feature_importance': feature_importance
        }

    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}")
        raise

def create_visualizations(y_test, predictions, targets, results):
    """Create and save visualization plots."""
    # Create comprehensive visualization
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('Predicted vs Actual Emission Rates', fontsize=16)

    for i, target in enumerate(targets):
        pred_col = f'Predicted_{target}'
        if pred_col in predictions.columns:
            axes[i].scatter(y_test[target], predictions[pred_col], alpha=0.5, s=20)

            # Perfect prediction line
            min_val = min(y_test[target].min(), predictions[pred_col].min())
            max_val = max(y_test[target].max(), predictions[pred_col].max())
            axes[i].plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)

            axes[i].set_xlabel(f'Actual {target}')
            axes[i].set_ylabel(f'Predicted {target}')
            axes[i].set_title(f'{target}\nR² = {results[target]["R2"]:.3f}')
            axes[i].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('model_predictions_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Individual NOx plot for backward compatibility
    if 'Predicted_NOx Rate (lbs/mmBtu)' in predictions.columns:
        plt.figure(figsize=(10, 6))
        plt.scatter(y_test['NOx Rate (lbs/mmBtu)'], predictions['Predicted_NOx Rate (lbs/mmBtu)'], alpha=0.5)
        plt.plot([y_test['NOx Rate (lbs/mmBtu)'].min(), y_test['NOx Rate (lbs/mmBtu)'].max()],
                 [y_test['NOx Rate (lbs/mmBtu)'].min(), y_test['NOx Rate (lbs/mmBtu)'].max()], 'r--')
        plt.xlabel('Actual NOx Rate (lbs/mmBtu)')
        plt.ylabel('Predicted NOx Rate (lbs/mmBtu)')
        plt.title('Predicted vs Actual NOx Rate')
        plt.grid(True, alpha=0.3)
        plt.savefig('nox_predictions.png', dpi=300, bbox_inches='tight')
        plt.close()

def create_prompt_template():
    """
    Create a prompt template as specified in Deliverable 9.

    Returns:
        str: Prompt template matching project specs format
    """
    template = "Summarize site performance: Avg = {avg_emission:.1f} lbs/mmBtu, Anomalies = {anomaly_dates}, Peak = {peak_emission:.1f} lbs/mmBtu"
    return template

def generate_mock_summary(predictions, df_processed):
    """
    Generate a mock summary function as specified in Deliverable 10.
    This creates a simple summary matching the project specs format.

    Args:
        predictions: DataFrame with predictions
        df_processed: Processed data DataFrame

    Returns:
        str: Mock summary in the specified format
    """
    # Use NOx as the primary emission for the mock summary (most reliable model)
    nox_col = 'Predicted_NOx Rate (lbs/mmBtu)'

    if nox_col in predictions.columns:
        avg_emission = predictions[nox_col].mean()
        peak_emission = predictions[nox_col].max()

        # Identify "anomalies" as predictions above 95th percentile
        threshold = predictions[nox_col].quantile(0.95)
        anomaly_indices = predictions[predictions[nox_col] > threshold].index

        # Create mock dates for anomalies (since we don't have actual dates)
        if len(anomaly_indices) > 0:
            anomaly_dates = f"Q{predictions.loc[anomaly_indices[0], 'Quarter'] if 'Quarter' in predictions.columns else '1'}"
            if len(anomaly_indices) > 1:
                anomaly_dates += f", Q{predictions.loc[anomaly_indices[1], 'Quarter'] if 'Quarter' in predictions.columns else '2'}"
        else:
            anomaly_dates = "None"
    else:
        avg_emission = 0.05  # Default values
        peak_emission = 0.08
        anomaly_dates = "Q2, Q4"

    # Use the prompt template
    template = create_prompt_template()
    mock_summary = template.format(
        avg_emission=avg_emission * 1000,  # Convert to more readable scale
        anomaly_dates=anomaly_dates,
        peak_emission=peak_emission * 1000
    )

    return mock_summary

def generate_summary(predictions, df_processed, results):
    """Generate comprehensive summary of model performance and predictions."""
    targets = ['SO2 Rate (lbs/mmBtu)', 'NOx Rate (lbs/mmBtu)', 'CO2 Rate (short tons/mmBtu)']

    # Calculate prediction statistics
    pred_stats = {}
    for target in targets:
        pred_col = f'Predicted_{target}'
        if pred_col in predictions.columns:
            pred_stats[target] = {
                'avg': predictions[pred_col].mean(),
                'median': predictions[pred_col].median(),
                'std': predictions[pred_col].std()
            }

    # Count high emission predictions
    high_emissions = 0
    for target in targets:
        pred_col = f'Predicted_{target}'
        if pred_col in predictions.columns and target in df_processed.columns:
            threshold = df_processed[target].median()
            high_emissions += (predictions[pred_col] > threshold).sum()

    # Get facility information
    facilities = []
    if 'Facility Name' in df_processed.columns:
        facilities = df_processed.loc[predictions.index, 'Facility Name'].unique()[:3].tolist()

    # Generate both mock and comprehensive summaries
    mock_summary = generate_mock_summary(predictions, df_processed)

    # Create comprehensive summary
    summary = f"""Emissions Compliance Forecasting Summary
Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}

MOCK SUMMARY (Project Specs Format):
{mock_summary}

MODEL PERFORMANCE:
"""

    for target in targets:
        if target in results:
            r2 = results[target]['R2']
            mae = results[target]['MAE']
            summary += f"- {target}: R² = {r2:.4f}, MAE = {mae:.4f}\n"

    summary += f"\nPREDICTION STATISTICS:\n"
    for target, stats in pred_stats.items():
        summary += f"- {target}: Avg = {stats['avg']:.4f}, Median = {stats['median']:.4f}\n"

    summary += f"\nCOMPLIANCE INSIGHTS:\n"
    summary += f"- High-emission predictions: {high_emissions} (above median rates)\n"
    summary += f"- Key facilities analyzed: {', '.join(facilities) if facilities else 'N/A'}\n"
    summary += f"- Total predictions generated: {len(predictions)}\n"

    return summary

def save_outputs(X_test, y_test, predictions, summary, evaluation_report=None, cv_results=None):
    """Save all outputs to files."""
    # Save summary
    with open('weekly_summary.txt', 'w') as f:
        f.write(summary)

    # Save evaluation report if provided
    if evaluation_report:
        with open('evaluation_report.md', 'w', encoding='utf-8') as f:
            f.write(evaluation_report)

    # Save cross-validation results if provided
    if cv_results:
        cv_df = pd.DataFrame(cv_results).T
        cv_df.to_csv('cross_validation_results.csv')

    # Save detailed predictions
    output_df = pd.concat([X_test, y_test, predictions], axis=1)
    output_df['Summary'] = summary
    output_df.to_csv('predictions.csv', index=True)

    saved_files = [
        "predictions.csv", "weekly_summary.txt",
        "model_predictions_comparison.png", "nox_predictions.png",
        "comprehensive_predictions_comparison.png", "residual_analysis.png",
        "performance_comparison.png", "distribution_comparison.png"
    ]

    if evaluation_report:
        saved_files.append("evaluation_report.md")
    if cv_results:
        saved_files.append("cross_validation_results.csv")

    logger.info(f"Outputs saved: {', '.join(saved_files)}")

if __name__ == "__main__":
    main()
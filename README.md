# Emissions Compliance Forecasting Module

## Overview

This project implements an AI/ML module to forecast emission rates (SO2, NOx, CO2) for power plant operations, ensuring compliance with EPA 40 CFR Part 75. The module predicts emission rates using quarterly CEMS data from EPA CAMPD, helping Rayfield Systems' energy compliance software automate monitoring and prevent regulatory violations.

## Workflow

The module addresses the pain point of manual emissions compliance monitoring by:

1. **Input**: Processing quarterly CEMS data (e.g., Gross Load, Heat Input, Fuel Type)
2. **Model**: Using improved RandomForestRegressor to predict SO2, NOx, and CO2 rates (lbs/mmBtu or short tons/mmBtu)
3. **Output**: Predicted emission rates, visualizations, and a compliance summary

## Model Performance

### Current Model Configuration
- **Algorithm**: RandomForestRegressor with improved hyperparameters
- **Features**: Gross Load, Heat Input, Operating Time, Load-to-Heat Ratio, Heat Efficiency, Operating Intensity, Primary Fuel Type, Unit Type, Quarter
- **Preprocessing**: Robust outlier removal, improved feature engineering, RobustScaler for numerical features

### Performance Metrics (Latest Results)
- **SO2 Rate**: MAE = 0.0000, R² = 1.0000 ⚠️ *May indicate overfitting due to aggressive outlier removal*
- **NOx Rate**: MAE = 0.0205, R² = 0.4371 ✅ *Moderate performance*
- **CO2 Rate**: MAE = 0.0001, R² = 0.4463 ✅ *Moderate performance*

## Project Structure

```
EnergyWorkflowAutomation/
├── ai_model_dev.ipynb          # Jupyter notebook with development steps
├── ai_module.py                # Core ML functions (preprocessing, training, prediction)
├── main_pipeline.py            # End-to-end pipeline script
├── requirements.txt            # Python dependencies
├── README.md                   # This file
├── data_analysis.py           # Data exploration utilities
├── quarterly-emissions-*.csv   # Input data (CEMS emissions data)
└── outputs/
    ├── predictions.csv         # Detailed predictions with features
    ├── weekly_summary.txt      # Performance summary
    ├── nox_predictions.png     # NOx predictions visualization
    └── model_predictions_comparison.png  # All targets comparison
```

## Key Features

### Data Preprocessing
- ✅ Robust outlier detection and removal using IQR method
- ✅ Improved missing value handling
- ✅ Feature engineering (efficiency metrics, operating intensity)
- ✅ Comprehensive logging and error handling

### Model Training
- ✅ Cross-validation for model evaluation
- ✅ Stratified train-test split by quarter
- ✅ RobustScaler for better handling of outliers
- ✅ Improved RandomForest hyperparameters

### Outputs
- ✅ Comprehensive visualizations
- ✅ Detailed performance metrics
- ✅ Compliance insights and facility analysis

## Installation and Setup

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Installation
```bash
# Clone or download the repository
cd EnergyWorkflowAutomation

# Install dependencies
pip install -r requirements.txt

# Ensure the CEMS data file is in the working directory
# File: quarterly-emissions-e59aa51f-1169-445d-918e-fb306c4f282a.csv
```

## Usage

### Quick Start
```bash
# Run the complete pipeline
python main_pipeline.py
```

### Development Workflow
```bash
# For interactive development, use the Jupyter notebook
jupyter notebook ai_model_dev.ipynb
```

### Expected Outputs
- `predictions.csv`: Test set features, actual rates, predicted rates, and summary
- `weekly_summary.txt`: Performance summary with compliance insights
- `nox_predictions.png`: Scatter plot of predicted vs. actual NOx rates
- `model_predictions_comparison.png`: Comprehensive visualization of all targets

## Limitations and Future Improvements

### Current Limitations
1. **Data Granularity**: Quarterly data limits hourly predictions; hourly CEMS data would align better with EPA requirements
2. **Overfitting Risk**: SO2 model shows perfect fit, indicating potential overfitting
3. **Compliance Thresholds**: Model predicts rates but doesn't flag specific EPA violation thresholds
4. **Temporal Dependencies**: Current model doesn't capture time-series patterns

### Planned Improvements
1. **Model Enhancements**:
   - Implement time-series specific models (LSTM, ARIMA)
   - Add ensemble methods combining multiple algorithms
   - Implement proper cross-validation for time-series data

2. **Data Quality**:
   - More sophisticated outlier detection methods
   - Integration with real-time data streams
   - Addition of weather and operational context data

3. **Production Readiness**:
   - Model drift monitoring
   - A/B testing framework
   - API endpoints for real-time predictions
   - Automated retraining pipeline

## Contributing

### Code Quality Standards
- Use `black` for code formatting
- Use `flake8` for linting
- Use `isort` for import sorting
- Maintain test coverage above 80%

### Testing
```bash
# Run tests
pytest tests/ --cov=ai_module

# Run specific test
pytest tests/test_preprocessing.py -v
```

## License

This project is developed for Rayfield Systems' energy compliance software.

## Contact

For questions or issues, please contact the development team.

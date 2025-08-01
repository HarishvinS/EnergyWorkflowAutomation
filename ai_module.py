import pandas as pd
import numpy as np
import logging
from sklearn.preprocessing import StandardScaler, OneHotEncoder, RobustScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def preprocess_data(df):
    """
    Preprocess CEMS data for modeling with improved data quality handling.

    Args:
        df (pd.DataFrame): Raw CEMS data

    Returns:
        tuple: (processed_df, features, targets)

    Raises:
        ValueError: If input data is invalid or missing required columns
        TypeError: If input is not a pandas DataFrame
    """
    # Input validation
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"Input must be a pandas DataFrame, got {type(df)}")

    if df.empty:
        raise ValueError("Input DataFrame is empty")

    # Check for required columns
    required_columns = [
        'Sum of the Operating Time', 'Gross Load (MWh)', 'Heat Input (mmBtu)',
        'Primary Fuel Type', 'Unit Type', 'Quarter',
        'SO2 Rate (lbs/mmBtu)', 'NOx Rate (lbs/mmBtu)', 'CO2 Rate (short tons/mmBtu)'
    ]

    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    logger.info(f"Starting preprocessing with {len(df)} rows")

    # Create a copy for processing
    df_processed = df.copy()

    # Filter out zero operating time (non-operational periods)
    initial_rows = len(df_processed)
    df_processed = df_processed[df_processed['Sum of the Operating Time'] > 0].copy()
    logger.info(f"Removed {initial_rows - len(df_processed)} rows with zero operating time")

    # Define targets
    targets = ['SO2 Rate (lbs/mmBtu)', 'NOx Rate (lbs/mmBtu)', 'CO2 Rate (short tons/mmBtu)']

    # Remove rows with missing target values
    for target in targets:
        before = len(df_processed)
        df_processed = df_processed.dropna(subset=[target])
        after = len(df_processed)
        if before != after:
            logger.info(f"Removed {before - after} rows with missing {target}")

    # Outlier removal using IQR method for each target
    for target in targets:
        before = len(df_processed)
        Q1 = df_processed[target].quantile(0.25)
        Q3 = df_processed[target].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 3 * IQR  # More conservative than 1.5 * IQR
        upper_bound = Q3 + 3 * IQR

        # Remove extreme outliers
        df_processed = df_processed[
            (df_processed[target] >= lower_bound) &
            (df_processed[target] <= upper_bound)
        ]
        after = len(df_processed)
        if before != after:
            logger.info(f"Removed {before - after} outliers from {target} (bounds: {lower_bound:.4f} to {upper_bound:.4f})")

    # Improved feature engineering
    logger.info("Applying feature engineering...")

    # Load to Heat Ratio with better handling (fix pandas warning)
    heat_input_safe = df_processed['Heat Input (mmBtu)'].replace(0, np.nan)
    df_processed['Load_to_Heat_Ratio'] = df_processed['Gross Load (MWh)'] / heat_input_safe
    median_ratio = df_processed['Load_to_Heat_Ratio'].median()
    df_processed['Load_to_Heat_Ratio'] = df_processed['Load_to_Heat_Ratio'].fillna(median_ratio)

    # Efficiency metric
    df_processed['Heat_Efficiency'] = df_processed['Gross Load (MWh)'] / (df_processed['Heat Input (mmBtu)'] + 1e-6)

    # Operating intensity
    df_processed['Operating_Intensity'] = df_processed['Sum of the Operating Time'] / (24 * 90)  # Fraction of quarter

    # Time-based features (as specified in project requirements)
    logger.info("Adding time-based features...")

    # Seasonal indicators
    df_processed['Is_Winter'] = (df_processed['Quarter'].isin([1, 4])).astype(int)  # Q1 (Jan-Mar) and Q4 (Oct-Dec)
    df_processed['Is_Summer'] = (df_processed['Quarter'].isin([2, 3])).astype(int)  # Q2 (Apr-Jun) and Q3 (Jul-Sep)

    # Quarter-based cyclical features (sine/cosine encoding for better ML representation)
    df_processed['Quarter_Sin'] = np.sin(2 * np.pi * df_processed['Quarter'] / 4)
    df_processed['Quarter_Cos'] = np.cos(2 * np.pi * df_processed['Quarter'] / 4)

    # Peak demand indicators (Q2 and Q3 typically have higher energy demand)
    df_processed['Is_Peak_Season'] = (df_processed['Quarter'].isin([2, 3])).astype(int)

    # Year-based features if available
    if 'Year' in df_processed.columns:
        # Normalize year to start from 0 for better model performance
        min_year = df_processed['Year'].min()
        df_processed['Years_Since_Start'] = df_processed['Year'] - min_year

        # Yearly cyclical features
        df_processed['Year_Sin'] = np.sin(2 * np.pi * df_processed['Years_Since_Start'] / 10)  # 10-year cycle
        df_processed['Year_Cos'] = np.cos(2 * np.pi * df_processed['Years_Since_Start'] / 10)

    logger.info("Time-based features added successfully")

    # Define features (including new time-based features)
    features = [
        'Gross Load (MWh)', 'Heat Input (mmBtu)', 'Sum of the Operating Time',
        'Load_to_Heat_Ratio', 'Heat_Efficiency', 'Operating_Intensity',
        'Primary Fuel Type', 'Unit Type', 'Quarter',
        # Time-based features
        'Is_Winter', 'Is_Summer', 'Quarter_Sin', 'Quarter_Cos', 'Is_Peak_Season'
    ]

    # Add year-based features if available
    if 'Year' in df_processed.columns:
        features.extend(['Years_Since_Start', 'Year_Sin', 'Year_Cos'])

    # Handle missing values in features
    numerical_cols = ['Gross Load (MWh)', 'Heat Input (mmBtu)', 'Sum of the Operating Time',
                      'Load_to_Heat_Ratio', 'Heat_Efficiency', 'Operating_Intensity',
                      'Is_Winter', 'Is_Summer', 'Quarter_Sin', 'Quarter_Cos', 'Is_Peak_Season']
    categorical_cols = ['Primary Fuel Type', 'Unit Type', 'Quarter']

    # Add year-based numerical features if available
    if 'Years_Since_Start' in df_processed.columns:
        numerical_cols.extend(['Years_Since_Start', 'Year_Sin', 'Year_Cos'])

    # Fix pandas warnings by avoiding inplace operations on potentially copied data
    for col in numerical_cols:
        if col in df_processed.columns:
            median_val = df_processed[col].median()
            df_processed[col] = df_processed[col].fillna(median_val)

    for col in categorical_cols:
        if col in df_processed.columns:
            mode_val = df_processed[col].mode()
            fill_val = mode_val[0] if len(mode_val) > 0 else 'Unknown'
            df_processed[col] = df_processed[col].fillna(fill_val)

    logger.info(f"Final dataset shape: {df_processed.shape}")
    logger.info(f"Remaining missing values: {df_processed[features + targets].isnull().sum().sum()}")

    return df_processed, features, targets

def train_model(X, y, target, use_robust_scaling=True):
    """
    Train an improved RandomForestRegressor model for a specific target.

    Args:
        X (pd.DataFrame): Feature matrix
        y (pd.DataFrame): Target matrix
        target (str): Target column name
        use_robust_scaling (bool): Whether to use RobustScaler instead of StandardScaler

    Returns:
        sklearn.pipeline.Pipeline: Trained model pipeline

    Raises:
        ValueError: If input data is invalid or target not found
        TypeError: If inputs are not pandas DataFrames
    """
    # Input validation
    if not isinstance(X, pd.DataFrame):
        raise TypeError(f"X must be a pandas DataFrame, got {type(X)}")

    if not isinstance(y, pd.DataFrame):
        raise TypeError(f"y must be a pandas DataFrame, got {type(y)}")

    if X.empty or y.empty:
        raise ValueError("Input DataFrames cannot be empty")

    if target not in y.columns:
        raise ValueError(f"Target '{target}' not found in y columns: {list(y.columns)}")

    if len(X) != len(y):
        raise ValueError(f"X and y must have same length: X={len(X)}, y={len(y)}")

    logger.info(f"Training model for {target}")

    # Define column types (including time-based features)
    numerical_cols = ['Gross Load (MWh)', 'Heat Input (mmBtu)', 'Sum of the Operating Time',
                      'Load_to_Heat_Ratio', 'Heat_Efficiency', 'Operating_Intensity',
                      'Is_Winter', 'Is_Summer', 'Quarter_Sin', 'Quarter_Cos', 'Is_Peak_Season']
    categorical_cols = ['Primary Fuel Type', 'Unit Type', 'Quarter']

    # Add year-based features if they exist in the data
    year_features = ['Years_Since_Start', 'Year_Sin', 'Year_Cos']
    for feature in year_features:
        if feature in X.columns:
            numerical_cols.append(feature)

    # Filter columns that actually exist in X
    numerical_cols = [col for col in numerical_cols if col in X.columns]
    categorical_cols = [col for col in categorical_cols if col in X.columns]

    # Choose scaler based on data distribution (RobustScaler is better for outliers)
    scaler = RobustScaler() if use_robust_scaling else StandardScaler()

    # Create preprocessing pipeline with improved categorical handling
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', scaler, numerical_cols),
            ('cat', OneHotEncoder(
                drop='first',
                handle_unknown='ignore',
                sparse_output=False  # Return dense arrays for consistency
            ), categorical_cols)
        ],
        remainder='drop'  # Drop any columns not specified
    )

    # Improved model with better hyperparameters
    model = RandomForestRegressor(
        n_estimators=200,  # More trees for better performance
        max_depth=15,      # Slightly deeper trees
        min_samples_split=10,  # Prevent overfitting
        min_samples_leaf=5,    # Prevent overfitting
        max_features='sqrt',   # Feature subsampling
        random_state=42,
        n_jobs=-1  # Use all available cores
    )

    # Create pipeline
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('model', model)
    ])

    # Fit the model
    try:
        pipeline.fit(X, y[target])
        logger.info(f"Successfully trained model for {target}")
    except Exception as e:
        logger.error(f"Error training model for {target}: {str(e)}")
        raise

    return pipeline

def predict(model, X):
    """
    Generate predictions using a trained model.

    Args:
        model (sklearn.pipeline.Pipeline): Trained model pipeline
        X (pd.DataFrame): Feature matrix

    Returns:
        np.ndarray: Predictions

    Raises:
        ValueError: If input data is invalid
        TypeError: If inputs are not of expected types
    """
    # Input validation
    if model is None:
        raise ValueError("Model cannot be None")

    if not hasattr(model, 'predict'):
        raise TypeError("Model must have a 'predict' method")

    if not isinstance(X, pd.DataFrame):
        raise TypeError(f"X must be a pandas DataFrame, got {type(X)}")

    if X.empty:
        raise ValueError("Input DataFrame X cannot be empty")

    try:
        predictions = model.predict(X)
        logger.info(f"Generated {len(predictions)} predictions")
        return predictions
    except Exception as e:
        logger.error(f"Error generating predictions: {str(e)}")
        raise

def evaluate_model(model, X_test, y_test, target):
    """
    Evaluate model performance with comprehensive metrics.

    Args:
        model: Trained model pipeline
        X_test: Test features
        y_test: Test targets
        target: Target column name

    Returns:
        dict: Evaluation metrics
    """
    from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error

    try:
        y_pred = model.predict(X_test)

        metrics = {
            'MAE': mean_absolute_error(y_test[target], y_pred),
            'RMSE': np.sqrt(mean_squared_error(y_test[target], y_pred)),
            'R2': r2_score(y_test[target], y_pred),
            'Mean_Actual': y_test[target].mean(),
            'Mean_Predicted': y_pred.mean(),
            'Std_Actual': y_test[target].std(),
            'Std_Predicted': y_pred.std()
        }

        logger.info(f"Model evaluation for {target}: R² = {metrics['R2']:.4f}, MAE = {metrics['MAE']:.4f}")
        return metrics

    except Exception as e:
        logger.error(f"Error evaluating model for {target}: {str(e)}")
        raise
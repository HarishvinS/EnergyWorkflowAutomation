import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor, IsolationForest

def preprocess_data(df):
    """Preprocess CEMS data for modeling."""
    # Filter out zero operating time
    df = df[df['Sum of the Operating Time'] > 0].copy()
    
    # Feature engineering
    df['Rolling_Heat_Input'] = df.groupby(['Facility ID', 'Unit ID'])['Heat Input (mmBtu)'].transform(
        lambda x: x.rolling(4, min_periods=1).mean())
    df['Load_to_Heat_Ratio'] = df['Gross Load (MWh)'] / df['Heat Input (mmBtu)'].replace(0, np.nan)
    
    # Handle missing values
    features = ['Gross Load (MWh)', 'Heat Input (mmBtu)', 'Sum of the Operating Time', 
                'Rolling_Heat_Input', 'Load_to_Heat_Ratio', 'Primary Fuel Type', 'Unit Type', 'Quarter']
    targets = ['SO2 Rate (lbs/mmBtu)', 'NOx Rate (lbs/mmBtu)', 'CO2 Rate (short tons/mmBtu)']
    
    for col in features + targets:
        if df[col].dtype in ['float64', 'int64']:
            df[col].fillna(df[col].median(), inplace=True)
        else:
            df[col].fillna(df[col].mode()[0], inplace=True)
    
    return df, features, targets

def train_model(X, y, target):
    """Train a RandomForestRegressor model for a specific target."""
    numerical_cols = ['Gross Load (MWh)', 'Heat Input (mmBtu)', 'Sum of the Operating Time', 
                      'Rolling_Heat_Input', 'Load_to_Heat_Ratio']
    categorical_cols = ['Primary Fuel Type', 'Unit Type', 'Quarter']
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_cols),
            ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), categorical_cols)
        ])
    
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('model', RandomForestRegressor(max_depth=10, n_estimators=100, random_state=42))
    ])
    
    pipeline.fit(X, y[target])
    return pipeline

def predict(model, X):
    """Generate predictions using a trained model."""
    return model.predict(X)

def detect_anomalies(df, targets):
    """Detect anomalies in emission rates using IsolationForest."""
    model = IsolationForest(contamination=0.05, random_state=42)
    # Focus on NOx Rate for anomaly detection
    features = ['NOx Rate (lbs/mmBtu)']
    df['Anomaly'] = model.fit_predict(df[features]) == -1
    return df
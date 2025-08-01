import pandas as pd
import numpy as np

# Load and analyze the data
df = pd.read_csv('quarterly-emissions-e59aa51f-1169-445d-918e-fb306c4f282a.csv')
print('=== DATA ANALYSIS ===')
print(f'Total rows: {len(df)}')
print(f'Rows with zero operating time: {len(df[df["Sum of the Operating Time"] == 0])}')

# Check target variables
targets = ['SO2 Rate (lbs/mmBtu)', 'NOx Rate (lbs/mmBtu)', 'CO2 Rate (short tons/mmBtu)']
for target in targets:
    print(f'\n{target}:')
    print(f'  Non-null values: {df[target].notna().sum()}')
    if df[target].notna().sum() > 0:
        print(f'  Mean: {df[target].mean():.6f}')
        print(f'  Std: {df[target].std():.6f}')
        print(f'  Min: {df[target].min():.6f}')
        print(f'  Max: {df[target].max():.6f}')
        print(f'  Zero values: {(df[target] == 0).sum()}')

# Check for data leakage issues
print('\n=== POTENTIAL DATA ISSUES ===')
df_active = df[df['Sum of the Operating Time'] > 0]
print(f'Active rows: {len(df_active)}')

# Check correlation between features and targets
features = ['Gross Load (MWh)', 'Heat Input (mmBtu)', 'Sum of the Operating Time']
print('\n=== CORRELATIONS ===')
for feature in features:
    for target in targets:
        if df_active[feature].notna().sum() > 0 and df_active[target].notna().sum() > 0:
            corr = df_active[feature].corr(df_active[target])
            print(f'{feature} vs {target}: {corr:.4f}')

# Check categorical variables
print('\n=== CATEGORICAL VARIABLES ===')
categorical_cols = ['Primary Fuel Type', 'Unit Type', 'Quarter']
for col in categorical_cols:
    print(f'\n{col} unique values: {df[col].nunique()}')
    print(df[col].value_counts().head())

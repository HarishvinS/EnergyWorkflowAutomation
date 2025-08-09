import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, r2_score
from ai_module import preprocess_data, train_model, predict, detect_anomalies

# Streamlit app layout
st.title("Emissions Compliance Forecasting App")
st.markdown("Upload a CEMS dataset to forecast SO2, NOx, and CO2 emission rates and detect anomalies.")

# Day 1: File uploader
uploaded_file = st.file_uploader("Upload CEMS CSV file", type="csv")

if uploaded_file:
    # Load and preprocess data
    df = pd.read_csv(uploaded_file)
    df, features, targets = preprocess_data(df)
    
    # Split data (use all data for training for simplicity in prototype)
    X = df[features]
    y = df[targets]
    
    # Train models and predict
    predictions = pd.DataFrame(index=X.index)
    results = {}
    for target in targets:
        model = train_model(X, y, target)
        y_pred = predict(model, X)
        predictions[f'Predicted_{target}'] = y_pred
        mae = mean_absolute_error(y[target], y_pred)
        r2 = r2_score(y[target], y_pred)
        results[target] = {'MAE': mae, 'R2': r2}
        st.write(f"{target} - MAE: {mae:.4f}, R²: {r2:.4f}")
    
    # Day 2: Anomaly detection
    df_with_anomalies = detect_anomalies(df, targets)
    
    # Day 1 & 3: Summary
    def generate_mock_summary(predictions, df):
        avg_so2 = predictions['Predicted_SO2 Rate (lbs/mmBtu)'].mean()
        avg_nox = predictions['Predicted_NOx Rate (lbs/mmBtu)'].mean()
        avg_co2 = predictions['Predicted_CO2 Rate (short tons/mmBtu)'].mean()
        high_emissions = len(predictions[
            (predictions['Predicted_SO2 Rate (lbs/mmBtu)'] > df['SO2 Rate (lbs/mmBtu)'].median()) |
            (predictions['Predicted_NOx Rate (lbs/mmBtu)'] > df['NOx Rate (lbs/mmBtu)'].median()) |
            (predictions['Predicted_CO2 Rate (short tons/mmBtu)'] > df['CO2 Rate (short tons/mmBtu)'].median())
        ])
        facilities = df.loc[predictions.index, 'Facility Name'].unique()[:3].tolist()
        anomalies = df_with_anomalies[df_with_anomalies['Anomaly'] == True][['Year', 'Quarter', 'Facility Name']].to_dict('records')
        anomaly_str = ', '.join([f"{a['Facility Name']} (Year {a['Year']}, Q{a['Quarter']})" for a in anomalies[:3]])
        summary = f"""**Weekly Summary:**
- Average SO2 Rate: {avg_so2:.4f} lbs/mmBtu
- Average NOx Rate: {avg_nox:.4f} lbs/mmBtu
- Average CO2 Rate: {avg_co2:.4f} short tons/mmBtu
- High-emission predictions: {high_emissions} (above median rates)
- Key facilities: {', '.join(facilities)}
- Anomalies detected: {anomaly_str if anomaly_str else 'None'}"""
        return summary
    
    summary = generate_mock_summary(predictions, df)
    st.markdown(summary)
    
    # Day 2 & 3: Visualize predictions and anomalies
    st.subheader("NOx Rate Predictions")
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(y['NOx Rate (lbs/mmBtu)'], predictions['Predicted_NOx Rate (lbs/mmBtu)'], alpha=0.5)
    ax.plot([y['NOx Rate (lbs/mmBtu)'].min(), y['NOx Rate (lbs/mmBtu)'].max()], 
            [y['NOx Rate (lbs/mmBtu)'].min(), y['NOx Rate (lbs/mmBtu)'].max()], 'r--')
    ax.set_xlabel('Actual NOx Rate (lbs/mmBtu)')
    ax.set_ylabel('Predicted NOx Rate (lbs/mmBtu)')
    ax.set_title('Predicted vs Actual NOx Rate')
    st.pyplot(fig)
    
    st.subheader("NOx Rate Anomalies")
    anomalies = df_with_anomalies[df_with_anomalies['Anomaly'] == True]
    if not anomalies.empty:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(df_with_anomalies.index, df_with_anomalies['NOx Rate (lbs/mmBtu)'], label='NOx Rate')
        ax.scatter(anomalies.index, anomalies['NOx Rate (lbs/mmBtu)'], color='red', label='Anomalies')
        ax.set_xlabel('Index')
        ax.set_ylabel('NOx Rate (lbs/mmBtu)')
        ax.set_title('NOx Rate with Anomalies')
        ax.legend()
        st.pyplot(fig)
    else:
        st.write("No anomalies detected.")
    
    # Day 4: Export alerts
    if not anomalies.empty:
        alerts = anomalies[['Year', 'Quarter', 'Facility Name', 'NOx Rate (lbs/mmBtu)']]
        alerts.to_csv('alerts_today.csv', index=False)
        st.write("Alerts exported to alerts_today.csv")
    else:
        st.write("No alerts to export (no anomalies).")
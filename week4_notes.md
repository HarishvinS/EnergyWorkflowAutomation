# Week 4 Notes: Emissions Compliance Forecasting App

## What Was Built

### Streamlit App (app.py): A web app that:

Allows users to upload a CEMS CSV file (quarterly-emissions-e59aa51f-1169-445d-918e-fb306c4f282a.csv).

Displays predicted SO2, NOx, and CO2 emission rates using the RandomForestRegressor model from Week 3.

Shows a scatter plot of predicted vs. actual NOx rates and a line plot highlighting NOx rate anomalies.

Generates a summary of average predicted rates, high-emission counts, key facilities, and detected anomalies.

Exports anomalies to alerts_today.csv for Zapier integration.

Updated ai_module.py: Added detect_anomalies function using IsolationForest to identify unusual NOx emission rates.

## What Didn't Get Finished

Zapier Screenshots: Unable to generate actual screenshots of the Zapier setup or email due to lack of access to a Zapier account or email client. Provided detailed setup instructions instead.

Real-Time Data: The app uses static quarterly data; real-time or hourly CEMS data integration was not implemented due to dataset limitations.

UI Polish: The Streamlit app is functional but minimal; additional styling or interactive features (e.g., filtering by facility) were not added due to time constraints.

## Next Steps

Incorporate Hourly Data: Obtain hourly CEMS data to align with EPA 40 CFR Part 75 requirements for more precise compliance monitoring.

Enhance UI: Add interactive filters (e.g., select facility or pollutant) and improve chart aesthetics in the Streamlit app.
# import pandas as pd
# import numpy as np


# df = pd.read_csv("TSLA.csv")
# print(df.head())

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Load the dataset
# file_path = '/path/to/tesla_stock_data.csv'  # Replace with the actual dataset path
file_path = "TSLA.csv"
data = pd.read_csv(file_path)
data['Date'] = pd.to_datetime(data['Date'])  # Convert Date to datetime format

# Basic Information
print(data.info())
print(data.describe())

# Time-Series Analysis
fig = px.line(data, x='Date', y='Adj Close', title='Tesla Adjusted Closing Price Over Time')
fig.show()

# Annual Trends
data['Year'] = data['Date'].dt.year
annual_summary = data.groupby('Year').agg({'High': 'max', 'Low': 'min', 'Volume': 'mean'}).reset_index()
fig = px.bar(annual_summary, x='Year', y=['High', 'Low'], title='Annual High and Low Prices', barmode='group')
fig.show()

# Monthly Seasonality
data['Month'] = data['Date'].dt.month
data['YearMonth'] = data['Date'].dt.to_period('M')
monthly_avg = data.groupby(['YearMonth']).mean().reset_index()
fig = px.line(monthly_avg, x='YearMonth', y='Adj Close', title='Monthly Average Adjusted Closing Price')
fig.show()

# Correlation Analysis
correlation = data.corr()
fig = px.imshow(correlation, text_auto=True, title='Correlation Heatmap')
fig.show()

# Volatility Analysis
data['Daily Change %'] = (data['Adj Close'].pct_change()) * 100
fig = px.histogram(data, x='Daily Change %', title='Daily Percentage Change Distribution', nbins=50)
fig.show()

# Highlight Significant Events
threshold = 10  # Define a threshold for significant daily changes
significant_changes = data[abs(data['Daily Change %']) > threshold]
fig = px.scatter(significant_changes, x='Date', y='Daily Change %', title='Significant Daily Percentage Changes')
fig.show()

# # Save processed data
# processed_file_path = '/path/to/processed_tesla_stock_data.csv'
# data.to_csv(processed_file_path, index=False)
# print(f"Processed data saved to {processed_file_path}")

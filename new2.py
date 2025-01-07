# gpt snippet
import streamlit as st
import pandas as pd
import plotly.graph_objs as go

# Load the Tesla stock data
df = pd.read_csv('TSLA.csv')
df['Date'] = pd.to_datetime(df['Date'])

# Title
st.title('Tesla Stock Price Dashboard')

# Sidebar for user inputs
st.sidebar.header('User Input Features')

# Custom Date Range Selector
start_date = st.sidebar.date_input("Start date", df['Date'].min())
end_date = st.sidebar.date_input("End date", df['Date'].max())
filtered_df = df[(df['Date'] >= pd.to_datetime(start_date)) & (df['Date'] <= pd.to_datetime(end_date))]

# Customizable Indicators
indicators = st.sidebar.multiselect('Select Indicators', ['SMA', 'EMA', 'Bollinger Bands', 'RSI', 'MACD'], default=['SMA', 'EMA'])

# Main Plot
st.subheader('Stock Price Over Time')
fig = go.Figure()

# Add traces based on selected indicators
fig.add_trace(go.Scatter(x=filtered_df['Date'], y=filtered_df['Close'], mode='lines', name='Close Price'))

# Calculate and add selected indicators
if 'SMA' in indicators:
    filtered_df['SMA'] = filtered_df['Close'].rolling(window=20).mean()
    fig.add_trace(go.Scatter(x=filtered_df['Date'], y=filtered_df['SMA'], mode='lines', name='SMA (20)'))

if 'EMA' in indicators:
    filtered_df['EMA'] = filtered_df['Close'].ewm(span=20, adjust=False).mean()
    fig.add_trace(go.Scatter(x=filtered_df['Date'], y=filtered_df['EMA'], mode='lines', name='EMA (20)'))

if 'Bollinger Bands' in indicators:
    filtered_df['BB_Upper'] = filtered_df['SMA'] + 2*filtered_df['Close'].rolling(window=20).std()
    filtered_df['BB_Lower'] = filtered_df['SMA'] - 2*filtered_df['Close'].rolling(window=20).std()
    fig.add_trace(go.Scatter(x=filtered_df['Date'], y=filtered_df['BB_Upper'], mode='lines', name='BB Upper'))
    fig.add_trace(go.Scatter(x=filtered_df['Date'], y=filtered_df['BB_Lower'], mode='lines', name='BB Lower'))

st.plotly_chart(fig)

# Scenario Analysis
st.subheader('Scenario Analysis')
st.write("Simulate future stock prices by adjusting the growth rate.")
growth_rate = st.slider("Select annual growth rate (%)", min_value=-50, max_value=50, value=10, step=1)

# Simulate future stock prices
future_days = 30
last_price = filtered_df['Close'].iloc[-1]
future_dates = pd.date_range(filtered_df['Date'].iloc[-1], periods=future_days+1).tolist()

simulated_prices = [last_price * (1 + growth_rate/100)**(i/252) for i in range(future_days+1)]
simulated_df = pd.DataFrame({'Date': future_dates, 'Simulated Price': simulated_prices})

# Plot the simulated prices
fig_simulation = go.Figure()
fig_simulation.add_trace(go.Scatter(x=simulated_df['Date'], y=simulated_df['Simulated Price'], mode='lines', name='Simulated Price'))
st.plotly_chart(fig_simulation)

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from datetime import datetime, timedelta
import numpy as np
from plotly.subplots import make_subplots
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

st.set_page_config(
    page_title="Tesla Stock Analytics Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    .main {
        background-color: #f0f2f6;
    }
    .stApp {
        max-width: 1400px;
        margin: 0 auto;
    }
    .plot-container {
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        border-radius: 8px;
        padding: 10px;
    }
    </style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    df = pd.read_csv('TSLA.csv')
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    df['Daily_Return'] = df['Close'].pct_change()
    df['Volatility'] = df['Daily_Return'].rolling(window=20).std() * np.sqrt(252)
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    df['Upper_BB'] = df['SMA_20'] + (df['Close'].rolling(window=20).std() * 2)
    df['Lower_BB'] = df['SMA_20'] - (df['Close'].rolling(window=20).std() * 2)
    return df

df = load_data()

st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/e/e8/Tesla_logo.png", width=200)
st.sidebar.title('Select Date Range')

default_end_date = df.index.max()
default_start_date = default_end_date - timedelta(days=180)
start_date = st.sidebar.date_input('Start Date', default_start_date, min_value=df.index.min(), max_value=df.index.max())
end_date = st.sidebar.date_input('End Date', default_end_date, min_value=df.index.min(), max_value=df.index.max())

filtered_df = df.loc[(df.index >= pd.to_datetime(start_date)) & (df.index <= pd.to_datetime(end_date))]

st.title('Tesla Stock Analytics Dashboard 📈')
st.markdown("""
    ### Understanding Key Metrics
    - **Latest Price**: The most recent stock closing price
    - **Trading Volume**: Number of shares traded in the latest session
    - **Volatility**: Measure of price fluctuation over time (higher % = more volatile)
    - **Price Range**: The difference between highest and lowest prices in selected period
""")

col1, col2, col3, col4 = st.columns(4)
latest_price = filtered_df['Close'].iloc[-1]
price_change = filtered_df['Close'].iloc[-1] - filtered_df['Close'].iloc[-2]
price_change_pct = (price_change / filtered_df['Close'].iloc[-2]) * 100
volatility = filtered_df['Volatility'].iloc[-1] * 100

with col1:
    st.metric("Latest Price", f"${latest_price:.2f}", f"{price_change:.2f} ({price_change_pct:.1f}%)")
with col2:
    st.metric("Trading Volume", f"{filtered_df['Volume'].iloc[-1]:,.0f}")
with col3:
    st.metric("Volatility", f"{volatility:.1f}%")
with col4:
    st.metric("Price Range", f"${filtered_df['High'].max()-filtered_df['Low'].min():.2f}")

st.subheader('1. Advanced Price Analysis')
st.markdown("""
    This section shows different ways to visualize Tesla's stock price movements. Each chart type offers unique insights:
    - **Area**: Shows the price movement with filled area below for trend visualization
    - **Candlestick**: Displays open, high, low, and close prices in a single bar
    - **OHLC**: Similar to candlestick but with a different visual representation
    - **Line with Markers**: Simple line chart with daily price points marked
""")

chart_types = ["Area", "Candlestick", "OHLC", "Line with Markers"]
chart_type = st.radio("Select Chart Type", chart_types, horizontal=True)

if chart_type == "Area":
    fig = px.area(filtered_df, y='Close', 
                  title='Tesla Stock Price',
                  template='plotly_dark')
    fig.update_layout(
        yaxis_title="Price (USD)",
        hovermode='x unified'
    )
elif chart_type == "Candlestick":
    fig = go.Figure(data=[go.Candlestick(
        x=filtered_df.index,
        open=filtered_df['Open'],
        high=filtered_df['High'],
        low=filtered_df['Low'],
        close=filtered_df['Close']
    )])
    fig.update_layout(
        title='Tesla Stock Price',
        yaxis_title='Price (USD)',
        template='plotly_dark',
        xaxis_rangeslider_visible=False
    )
elif chart_type == "Line with Markers":
    fig = px.line(filtered_df, y='Close',
                  title='Tesla Stock Price with Markers',
                  template='plotly_dark')
    fig.add_scatter(x=filtered_df.index, y=filtered_df['Close'],
                   mode='markers', name='Daily Points')
else:
    fig = go.Figure(data=[go.Ohlc(
        x=filtered_df.index,
        open=filtered_df['Open'],
        high=filtered_df['High'],
        low=filtered_df['Low'],
        close=filtered_df['Close']
    )])
    fig.update_layout(
        title='Tesla Stock Price',
        yaxis_title='Price (USD)',
        template='plotly_dark',
        xaxis_rangeslider_visible=False
    )

st.plotly_chart(fig, use_container_width=True)
st.subheader('2. Comprehensive Technical Analysis')
st.markdown("""
    Technical analysis helps predict future price movements based on historical data:
    - **Moving Averages**: Show average price over different time periods to identify trends
    - **RSI (Relative Strength Index)**: Measures momentum and identifies overbought/oversold conditions
""")

ta_col1, ta_col2 = st.columns(2)

with ta_col1:
    st.markdown("**Moving Averages**: Smooths out price data to show trends more clearly")
    ma_periods = st.multiselect(
        'Select Moving Average Periods',
        [5, 10, 20, 50, 100, 200],
        default=[20, 50]
    )
    
    fig_ma = go.Figure()
    fig_ma.add_trace(go.Scatter(x=filtered_df.index, y=filtered_df['Close'],
                               name='Close Price', line=dict(color='#00b4d8')))
    
    for period in ma_periods:
        ma = filtered_df['Close'].rolling(window=period).mean()
        fig_ma.add_trace(go.Scatter(x=filtered_df.index, y=ma,
                                  name=f'{period}-day MA'))
    
    fig_ma.update_layout(
        title='Moving Averages Analysis',
        template='plotly_dark',
        height=400,
        hovermode='x unified'
    )
    st.plotly_chart(fig_ma, use_container_width=True)

with ta_col2:
    st.markdown("**RSI**: Values above 70 indicate overbought conditions, below 30 indicate oversold")
    rsi_period = st.slider('RSI Period', 5, 30, 14)
    delta = filtered_df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=rsi_period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=rsi_period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    
    fig_rsi = go.Figure()
    fig_rsi.add_trace(go.Scatter(x=filtered_df.index, y=rsi, name='RSI',
                                line=dict(color='#00b4d8')))
    fig_rsi.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Overbought")
    fig_rsi.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="Oversold")
    fig_rsi.update_layout(
        title='RSI Analysis',
        template='plotly_dark',
        height=400,
        yaxis=dict(range=[0, 100])
    )
    st.plotly_chart(fig_rsi, use_container_width=True)

st.subheader('3. Bollinger Bands Analysis')
st.markdown("""
    Bollinger Bands are volatility bands placed above and below a moving average:
    - Upper and lower bands represent standard deviations from the middle band
    - Wider bands indicate higher volatility
    - Price touching or exceeding bands may signal potential reversal
""")

fig_bb = go.Figure()
fig_bb.add_trace(go.Scatter(x=filtered_df.index, y=filtered_df['Close'], name='Close Price'))
fig_bb.add_trace(go.Scatter(x=filtered_df.index, y=filtered_df['Upper_BB'], name='Upper BB',
                           line=dict(dash='dash')))
fig_bb.add_trace(go.Scatter(x=filtered_df.index, y=filtered_df['Lower_BB'], name='Lower BB',
                           line=dict(dash='dash')))
fig_bb.update_layout(title='Bollinger Bands', template='plotly_dark')
st.plotly_chart(fig_bb, use_container_width=True)

st.subheader('4. Volume Analysis')
st.markdown("""
    Trading volume is a crucial indicator of market interest:
    - Higher volume often validates price movements
    - Volume spikes may indicate important market events
    - The 20-day average volume helps identify unusual trading activity
""")

fig_vol = make_subplots(specs=[[{"secondary_y": True}]])
fig_vol.add_trace(go.Bar(
    x=filtered_df.index,
    y=filtered_df['Volume'],
    name='Volume',
    marker_color='rgba(0, 180, 216, 0.5)'
), secondary_y=False)
fig_vol.add_trace(go.Scatter(
    x=filtered_df.index,
    y=filtered_df['Volume'].rolling(window=20).mean(),
    name='20-day Average Volume',
    line=dict(color='orange')
), secondary_y=True)
fig_vol.update_layout(
    title='Trading Volume Analysis',
    template='plotly_dark',
    hovermode='x unified'
)
st.plotly_chart(fig_vol, use_container_width=True)

# 5 returns analysis  ye wala 2 try --- > Multiple Metrics
st.subheader('5. Returns Analysis')
st.markdown("""
    Returns analysis shows how much profit or loss the stock has generated:
    - Daily Returns: Shows the percentage change in price each day
    - Distribution: Helps understand the range and frequency of returns
    - Cumulative Returns: Shows total returns over time if reinvested
""")

returns_col1, returns_col2 = st.columns(2)

with returns_col1:
    daily_returns = filtered_df['Daily_Return'].dropna()
    fig_returns = go.Figure()
    fig_returns.add_trace(go.Histogram(
        x=daily_returns,
        nbinsx=50,
        name='Daily Returns',
        showlegend=True
    ))
    fig_returns.add_trace(go.Histogram(
        x=daily_returns.rolling(window=5).mean(),
        nbinsx=50,
        name='5-day Rolling Returns',
        opacity=0.7
    ))
    fig_returns.update_layout(
        title='Distribution of Returns',
        template='plotly_dark',
        barmode='overlay',
        height=400
    )
    st.plotly_chart(fig_returns, use_container_width=True)

with returns_col2:
    cumulative_returns = (1 + daily_returns).cumprod()
    fig_cum_returns = px.line(
        cumulative_returns,
        title='Cumulative Returns Analysis',
        template='plotly_dark'
    )
    fig_cum_returns.update_layout(height=400)
    st.plotly_chart(fig_cum_returns, use_container_width=True)

# 6. Volatility Analysis
st.subheader('6. Volatility Analysis')
st.markdown("""
    Volatility measures the degree of price fluctuation:
    - Higher volatility indicates greater price swings and risk
    - Lower volatility suggests more stable price movement
    - 20-day historical volatility is annualized for comparison
""")

fig_vol = px.line(filtered_df['Volatility'],
                  title='Historical Volatility (20-day)',
                  template='plotly_dark')
st.plotly_chart(fig_vol, use_container_width=True)

# 7. Price Momentum
st.subheader('7. Price Momentum Analysis')
st.markdown("""
    Momentum shows the speed of price changes:
    - Positive momentum indicates upward price pressure
    - Negative momentum suggests downward pressure
    - Multiple timeframes help confirm momentum strength
""")

momentum_periods = [5, 10, 20]
fig_momentum = go.Figure()
for period in momentum_periods:
    momentum = filtered_df['Close'].diff(period)
    fig_momentum.add_trace(go.Scatter(x=filtered_df.index, y=momentum,
                                    name=f'{period}-day Momentum'))
fig_momentum.update_layout(title='Price Momentum', template='plotly_dark')
st.plotly_chart(fig_momentum, use_container_width=True)

# 8. Correlation Analysis
if st.checkbox('Show Correlation Analysis'):
    st.subheader('8. Correlation Analysis')
    st.markdown("""
        Correlation shows relationships between different price metrics:
        - Values range from -1 (inverse relationship) to +1 (direct relationship)
        - Helps understand how different price components move together
        - Useful for identifying trading patterns
    """)
    
    corr = filtered_df[['Open', 'High', 'Low', 'Close', 'Volume']].corr()
    fig_corr = go.Figure(data=go.Heatmap(
        z=corr.values,
        x=corr.columns,
        y=corr.columns,
        text=np.round(corr.values, 2),
        texttemplate='%{text}',
        textfont={"size": 10},
        hoverongaps=False,
        colorscale='RdBu'
    ))
    fig_corr.update_layout(height=500, title='Correlation Matrix')
    st.plotly_chart(fig_corr, use_container_width=True)

# 9. Price Trends
st.subheader('9. Price Trend Analysis')
st.markdown("""
    Trend analysis helps identify the overall price direction:
    - 20-day trend shows short-term price direction
    - 50-day trend indicates intermediate-term movement
    - Crossing of trend lines may signal trend changes
""")

fig_trends = go.Figure()
fig_trends.add_trace(go.Scatter(x=filtered_df.index, y=filtered_df['Close'],
                               name='Close Price'))
fig_trends.add_trace(go.Scatter(x=filtered_df.index, y=filtered_df['SMA_20'],
                               name='20-day Trend'))
fig_trends.add_trace(go.Scatter(x=filtered_df.index, y=filtered_df['SMA_50'],
                               name='50-day Trend'))
fig_trends.update_layout(title='Price Trends', template='plotly_dark')
st.plotly_chart(fig_trends, use_container_width=True)

# 10. Trading Range Analysis
st.subheader('10. Trading Range Analysis')
st.markdown("""
    Trading range shows the daily price spread:
    - Larger ranges indicate higher intraday volatility
    - Smaller ranges suggest more stable trading
    - Unusual ranges may signal important market events
""")

fig_range = go.Figure()
fig_range.add_trace(go.Scatter(x=filtered_df.index, y=filtered_df['High']-filtered_df['Low'],
                              name='Daily Trading Range'))
fig_range.update_layout(title='Daily Trading Range', template='plotly_dark')
st.plotly_chart(fig_range, use_container_width=True)

# 11. Volume-Price Relationship
st.subheader('11. Volume-Price Relationship')
st.markdown("""
    This analysis shows how volume relates to price:
    - Higher volumes at certain price levels may indicate support/resistance
    - Trend line shows the general relationship between volume and price
    - Outliers may indicate significant market events
""")

fig_vol_price = px.scatter(filtered_df, x='Volume', y='Close',
                          title='Volume vs Price',
                          template='plotly_dark',
                          trendline="ols")
st.plotly_chart(fig_vol_price, use_container_width=True)

# 12. Price Distribution
st.subheader('12. Price Distribution Analysis')
st.markdown("""
    Price distribution shows the range and frequency of price levels:
    - Wider distribution indicates more price volatility
    - The box shows the main trading range
    - The violin shape shows where prices tend to cluster
""")

fig_dist = go.Figure()
fig_dist.add_trace(go.Violin(y=filtered_df['Close'], box_visible=True,
                            meanline_visible=True))
fig_dist.update_layout(title='Price Distribution', template='plotly_dark')
st.plotly_chart(fig_dist, use_container_width=True)


def predict_future_prices(df, days_ahead=5):
    # Use 'Close' price as feature and target
    df['Day'] = np.arange(len(df))
    X = df[['Day']]
    y = df['Close']
    
    # Train/Test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    
    # Train the model
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Predict future prices
    future_days = np.array(range(len(df), len(df) + days_ahead)).reshape(-1, 1)
    future_predictions = model.predict(future_days)
    
    return future_predictions

st.subheader('Future Price Prediction')
st.markdown("Predict future closing prices for Tesla stock using a simple linear regression model.")

days_ahead = st.slider("Days ahead to predict", min_value=1, max_value=30, value=5)

future_predictions = predict_future_prices(df, days_ahead=days_ahead)
future_dates = pd.date_range(df.index.max() + timedelta(1), periods=days_ahead)

fig_pred = go.Figure()
fig_pred.add_trace(go.Scatter(x=df.index, y=df['Close'], name='Actual Prices', line=dict(color='blue')))
fig_pred.add_trace(go.Scatter(x=future_dates, y=future_predictions, name='Predicted Prices', line=dict(color='red', dash='dash')))
fig_pred.update_layout(title='Future Price Prediction', template='plotly_dark', xaxis_title='Date', yaxis_title='Price (USD)')

st.plotly_chart(fig_pred, use_container_width=True)






st.markdown('---')



st.markdown("""
    <div style='text-align: center; padding: 20px;'>
        <h4>Created by Subhan Ali</h4>
        <p>Advanced Technical Analysis Dashboard for Tesla Stock</p>
    </div>
""", unsafe_allow_html=True)


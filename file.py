import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Load the dataset
data = pd.read_csv("TSLA.csv")
data['Date'] = pd.to_datetime(data['Date'])

# Helper Functions
def plot_candlestick(data):
    fig = go.Figure(data=[go.Candlestick(
        x=data['Date'],
        open=data['Open'],
        high=data['High'],
        low=data['Low'],
        close=data['Close']
    )])
    fig.update_layout(title='Tesla Stock Candlestick Chart', xaxis_title='Date', yaxis_title='Price')
    return fig

def plot_moving_averages(data):
    data['SMA_50'] = data['Adj Close'].rolling(window=50).mean()
    data['SMA_200'] = data['Adj Close'].rolling(window=200).mean()
    fig = px.line(data, x='Date', y=['Adj Close', 'SMA_50', 'SMA_200'], 
                  title='Tesla Stock Prices with Moving Averages')
    return fig

def plot_price_distribution(data):
    fig = px.histogram(data, x='Adj Close', nbins=50, title='Distribution of Adjusted Close Prices')
    return fig

def plot_trading_volume(data):
    fig = px.bar(data, x='Date', y='Volume', title='Tesla Trading Volume Over Time', 
                 labels={'Volume': 'Trading Volume'})
    return fig

def plot_volatility(data):
    data['Daily Change (%)'] = data['Adj Close'].pct_change() * 100
    fig = px.histogram(data, x='Daily Change (%)', nbins=50, title='Daily Percentage Change Distribution')
    return fig

def plot_monthly_avg(data):
    data['Month'] = data['Date'].dt.month
    monthly_avg = data.groupby('Month')['Adj Close'].mean().reset_index()
    fig = px.bar(monthly_avg, x='Month', y='Adj Close', 
                 title='Average Adjusted Close Price by Month', 
                 labels={'Adj Close': 'Adjusted Close Price', 'Month': 'Month'})
    return fig

def plot_yearly_avg(data):
    data['Year'] = data['Date'].dt.year
    yearly_avg = data.groupby('Year')['Adj Close'].mean().reset_index()
    fig = px.line(yearly_avg, x='Year', y='Adj Close', 
                  title='Yearly Average Adjusted Close Price', 
                  labels={'Adj Close': 'Adjusted Close Price', 'Year': 'Year'})
    return fig

def plot_correlation_heatmap(data):
    numeric_columns = data.select_dtypes(include='number')
    correlation = numeric_columns.corr()
    fig = px.imshow(correlation, text_auto=True, title='Correlation Heatmap', labels={'color': 'Correlation'})
    return fig


# Streamlit App
st.set_page_config(page_title="Tesla Stock Analysis", layout="wide")
st.title("Tesla Stock Analysis Dashboard")


data['Daily Change (%)'] = data['Adj Close'].pct_change() * 100
# Sidebar for navigation
st.sidebar.title("Navigation")
options = [
    "Overview",
    "Time Series Analysis",
    "Candlestick Chart",
    "Moving Averages",
    "Price Distribution",
    "Trading Volume",
    "Volatility Analysis",
    "Monthly Seasonality",
    "Yearly Trends",
    "Correlation Heatmap",
    "Significant Events"
]
choice = st.sidebar.selectbox("Choose Analysis", options)


if choice == "Overview":
    st.header("Tesla Stock Analysis Overview")
    st.subheader("📊 Dataset Overview")
    st.write(f"**📅 Total Records**: {len(data)}")
    st.write(f"**📈 Average Adjusted Close Price**: ${data['Adj Close'].mean():.2f}")
    st.write(f"**📊 Average Trading Volume**: {data['Volume'].mean():,.0f}")
    st.write(f"**📉 Average Daily Percentage Change**: {data['Daily Change (%)'].mean():.2f}%")
    st.write(f"**📅 Date Range**: {data['Date'].min().date()} to {data['Date'].max().date()}")
    st.subheader("📑 Dataset Preview")
    st.write(data.head(10))

elif choice == "Time Series Analysis":
    fig = px.line(data, x='Date', y='Adj Close', title='Tesla Adjusted Closing Price Over Time')
    st.plotly_chart(fig, use_container_width=True)

elif choice == "Candlestick Chart":
    fig = plot_candlestick(data)
    st.plotly_chart(fig, use_container_width=True)

elif choice == "Moving Averages":
    fig = plot_moving_averages(data)
    st.plotly_chart(fig, use_container_width=True)

elif choice == "Price Distribution":
    fig = plot_price_distribution(data)
    st.plotly_chart(fig, use_container_width=True)

elif choice == "Trading Volume":
    fig = plot_trading_volume(data)
    st.plotly_chart(fig, use_container_width=True)

elif choice == "Volatility Analysis":
    fig = plot_volatility(data)
    st.plotly_chart(fig, use_container_width=True)

elif choice == "Monthly Seasonality":
    fig = plot_monthly_avg(data)
    st.plotly_chart(fig, use_container_width=True)

elif choice == "Yearly Trends":
    fig = plot_yearly_avg(data)
    st.plotly_chart(fig, use_container_width=True)

elif choice == "Correlation Heatmap":
    fig = plot_correlation_heatmap(data)
    st.plotly_chart(fig, use_container_width=True)

elif choice == "Significant Events":
    data['Daily Change (%)'] = data['Adj Close'].pct_change() * 100
    threshold = 10  # Define a threshold for significant daily changes
    significant_changes = data[abs(data['Daily Change (%)']) > threshold]
    fig = px.scatter(significant_changes, x='Date', y='Daily Change (%)', 
                     title='Significant Daily Percentage Changes')
    st.plotly_chart(fig, use_container_width=True)

st.sidebar.info("Select a tab to explore various analyses of Tesla's stock performance.")

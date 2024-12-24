import pandas as pd
import matplotlib.pyplot as plt

# Load your dataset
df = pd.read_csv('TSLA.csv')

# Ensure the Date column (if present) is in datetime format
df['Date'] = pd.to_datetime(df['Date'])

# Set the Date as the index for better visualization
df.set_index('Date', inplace=True)

# Plot all numerical columns for each row
df[['Open', 'High', 'Low', 'Close', 'Adj Close']].plot(figsize=(15, 7))
plt.title('Tesla Stock Data (2010-2020)')
plt.ylabel('Price')
plt.xlabel('Date')
plt.legend(loc='best')
plt.show()



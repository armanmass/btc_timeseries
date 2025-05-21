#%%
import ccxt, pandas as pd
import mplfinance as mpf
import matplotlib.pyplot as plt
from datetime import datetime

# Download data
exchange = ccxt.binance({
    "enableRateLimit": True,
})
symbol = "BTC/USD"
timeframe = "1h"
start_date = "2020-05-16T00:00:00Z"
end_date = "2024-03-16T00:00:00Z"
since = exchange.parse8601(start_date)
end_timestamp = exchange.parse8601(end_date)
all_ohlcv = []

print(f"Downloading BTC/USD data from {start_date} to {end_date}")
while since < end_timestamp:
    try:
        batch = exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=1000)
        if not batch: 
            print("No more data available")
            break
        all_ohlcv += batch
        since = batch[-1][0] + 1
        print(f"Downloaded data until {pd.to_datetime(since, unit='ms')}")
    except Exception as e:
        print(f"Error downloading data: {str(e)}")
        break

if not all_ohlcv:
    raise ValueError("No data was downloaded")

df = pd.DataFrame(all_ohlcv, columns=[
    "time","open","high","low","close","volume"
])
df["time"] = pd.to_datetime(df["time"], unit="ms")
df.set_index("time", inplace=True)
df.to_csv("btc_usd_5y_hourly_binance.csv")
print(f"Downloaded {len(df)} data points from {df.index[0]} to {df.index[-1]}")

#%%
# Calculate technical indicators
df['SMA20'] = df['close'].rolling(window=20).mean()
df['SMA50'] = df['close'].rolling(window=50).mean()

# Create the candlestick chart
mpf_style = mpf.make_mpf_style(base_mpf_style='charles', 
                              gridstyle='', 
                              y_on_right=True,
                              marketcolors=mpf.make_marketcolors(up='#26a69a',
                                                               down='#ef5350',
                                                               edge='inherit',
                                                               wick='inherit',
                                                               volume='in'))

# Add volume and moving averages
add_plots = [
    mpf.make_addplot(df['SMA20'], color='blue', width=0.7),
    mpf.make_addplot(df['SMA50'], color='red', width=0.7),
]

# Plot the chart
fig, axes = mpf.plot(df,
                    type='candle',
                    style=mpf_style,
                    title='BTC/USD Hourly Price Chart',
                    ylabel='Price (USD)',
                    volume=True,
                    addplot=add_plots,
                    figsize=(15, 10),
                    panel_ratios=(3, 1),
                    returnfig=True)

# Add legend
axes[0].legend(['SMA20', 'SMA50'])

# Save the plot
plt.savefig('btc_price_chart.png', bbox_inches='tight', dpi=300)
plt.close()

#%%
# Create returns distribution plot
plt.figure(figsize=(12, 6))
df['returns'] = df['close'].pct_change()
df['returns'].hist(bins=100)
plt.title('Distribution of Hourly Returns')
plt.xlabel('Return')
plt.ylabel('Frequency')
plt.savefig('btc_returns_distribution.png', bbox_inches='tight', dpi=300)
plt.close()

print("Charts have been saved as 'btc_price_chart.png' and 'btc_returns_distribution.png'")
# %%

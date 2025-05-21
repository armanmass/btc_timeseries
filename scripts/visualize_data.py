import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Set style
sns.set(style="whitegrid")

# Path to data (change if needed)
data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
parquet_files = [f for f in os.listdir(data_dir) if f.endswith('.parquet')]
if not parquet_files:
    print("No Parquet files found in data directory.")
    sys.exit(1)

# Use the first Parquet file found
parquet_path = os.path.join(data_dir, parquet_files[0])
df = pd.read_parquet(parquet_path)

# --- 1. Price over time ---
plt.figure(figsize=(14, 5))
plt.plot(df.index, df['close'], label='Close Price')
plt.title('BTC Close Price Over Time')
plt.xlabel('Time')
plt.ylabel('Price (USD)')
plt.legend()
plt.tight_layout()
plt.savefig('btc_price_over_time.png')
plt.close()

# --- 2. Volume over time ---
plt.figure(figsize=(14, 5))
plt.plot(df.index, df['volume'], color='orange', label='Volume')
plt.title('BTC Volume Over Time')
plt.xlabel('Time')
plt.ylabel('Volume')
plt.legend()
plt.tight_layout()
plt.savefig('btc_volume_over_time.png')
plt.close()

# --- 3. Distribution of returns (1h log returns) ---
df['log_return'] = (df['close'] / df['close'].shift(1)).apply(lambda x: np.log(x) if x > 0 else 0)
plt.figure(figsize=(8, 5))
sns.histplot(df['log_return'].dropna(), bins=100, kde=True)
plt.title('Distribution of 1h Log Returns')
plt.xlabel('Log Return')
plt.ylabel('Frequency')
plt.tight_layout()
plt.savefig('btc_log_return_distribution.png')
plt.close()

# --- 4. Correlation heatmap of features ---
plt.figure(figsize=(12, 10))
corr = df[[col for col in df.columns if df[col].dtype != 'O']].corr()
sns.heatmap(corr, annot=False, cmap='coolwarm', center=0)
plt.title('Feature Correlation Heatmap')
plt.tight_layout()
plt.savefig('btc_feature_correlation_heatmap.png')
plt.close()

# --- 5. Technical indicators (RSI, MACD) ---
if 'rsi' in df.columns:
    plt.figure(figsize=(14, 3))
    plt.plot(df.index, df['rsi'], label='RSI')
    plt.title('RSI Over Time')
    plt.xlabel('Time')
    plt.ylabel('RSI')
    plt.legend()
    plt.tight_layout()
    plt.savefig('btc_rsi_over_time.png')
    plt.close()
if 'macd' in df.columns and 'macd_signal' in df.columns:
    plt.figure(figsize=(14, 3))
    plt.plot(df.index, df['macd'], label='MACD')
    plt.plot(df.index, df['macd_signal'], label='MACD Signal')
    plt.title('MACD and Signal Over Time')
    plt.xlabel('Time')
    plt.ylabel('MACD')
    plt.legend()
    plt.tight_layout()
    plt.savefig('btc_macd_over_time.png')
    plt.close()

# --- 6. Seasonality: Hour, Day of Week, Month ---
if 'hour' in df.columns:
    plt.figure(figsize=(8, 4))
    sns.boxplot(x='hour', y='close', data=df.reset_index())
    plt.title('BTC Price by Hour of Day')
    plt.xlabel('Hour')
    plt.ylabel('Close Price')
    plt.tight_layout()
    plt.savefig('btc_price_by_hour.png')
    plt.close()
if 'day_of_week' in df.columns:
    plt.figure(figsize=(8, 4))
    sns.boxplot(x='day_of_week', y='close', data=df.reset_index())
    plt.title('BTC Price by Day of Week')
    plt.xlabel('Day of Week (0=Mon)')
    plt.ylabel('Close Price')
    plt.tight_layout()
    plt.savefig('btc_price_by_dayofweek.png')
    plt.close()
if 'month' in df.columns:
    plt.figure(figsize=(8, 4))
    sns.boxplot(x='month', y='close', data=df.reset_index())
    plt.title('BTC Price by Month')
    plt.xlabel('Month')
    plt.ylabel('Close Price')
    plt.tight_layout()
    plt.savefig('btc_price_by_month.png')
    plt.close()

print("Saved key BTC data visualizations to current directory.") 
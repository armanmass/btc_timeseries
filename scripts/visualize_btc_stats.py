import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Set style
sns.set(style="whitegrid")

# Output directory
output_dir = os.path.join(os.path.dirname(__file__), 'visualizations')
os.makedirs(output_dir, exist_ok=True)

# Path to data (change if needed)
data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')

# Look for the full processed parquet file
full_processed_parquet_files = [f for f in os.listdir(data_dir) if f.endswith('_full_processed.parquet')]

if not full_processed_parquet_files:
    print("Error: No full processed Parquet files found in data directory. Please run the preprocessing script (e.g., test_preprocessing.py) first to generate the cached data.")
    sys.exit(1)

# Use the first full processed Parquet file found
parquet_path = os.path.join(data_dir, full_processed_parquet_files[0])

try:
    print(f"Attempting to load data from: {parquet_path}")
    df = pd.read_parquet(parquet_path)
    print(f"Successfully loaded data with columns: {list(df.columns)}")
except Exception as e:
    print(f"Error loading {parquet_path}: {e}")
    sys.exit(1)

# --- Summary statistics table ---
# Exclude non-numeric columns for describe()
numeric_df = df.select_dtypes(include=np.number)
stats = numeric_df.describe().T[['mean', 'min', 'max']]
stats.to_csv(os.path.join(output_dir, 'btc_feature_stats.csv'))
print(f"Saved summary statistics to {os.path.join(output_dir, 'btc_feature_stats.csv')}")

# --- Identify raw and engineered features ---
raw_features = ['open', 'high', 'low', 'close', 'volume']
# Ensure columns exist before trying to remove them
valid_raw_features = [f for f in raw_features if f in df.columns]
engineered_features = [col for col in df.columns if col not in valid_raw_features and df[col].dtype != 'O']

print(f"Identified raw features: {valid_raw_features}")
print(f"Identified engineered features ({len(engineered_features)}): {engineered_features}")

# --- Save lists of features ---
with open(os.path.join(output_dir, 'btc_raw_features.txt'), 'w') as f:
    f.write('\n'.join(valid_raw_features))
with open(os.path.join(output_dir, 'btc_engineered_features.txt'), 'w') as f:
    f.write('\n'.join(engineered_features))
print(f"Saved feature lists to {output_dir}")

# --- 1. Price over time ---
# Check if 'close' column exists
if 'close' in df.columns:
    plt.figure(figsize=(14, 5))
    plt.plot(df.index, df['close'], label='Close Price')
    plt.title('BTC Close Price Over Time')
    plt.xlabel('Time')
    plt.ylabel('Price (USD)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'btc_price_over_time.png'))
    plt.close()
    print("Saved BTC Price Over Time plot.")
else:
    print("'close' column not found for Price Over Time plot.")

# --- 2. Volume over time ---
# Check if 'volume' column exists
if 'volume' in df.columns:
    plt.figure(figsize=(14, 5))
    plt.plot(df.index, df['volume'], color='orange', label='Volume')
    plt.title('BTC Volume Over Time')
    plt.xlabel('Time')
    plt.ylabel('Volume')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'btc_volume_over_time.png'))
    plt.close()
    print("Saved BTC Volume Over Time plot.")
else:
     print("'volume' column not found for Volume Over Time plot.")

# --- 3. Distribution of returns (1h log returns) ---
# Check if 'close' column exists for log return calculation
if 'close' in df.columns:
    df['log_return'] = (df['close'] / df['close'].shift(1)).apply(lambda x: np.log(x) if x > 0 else np.nan)
    plt.figure(figsize=(8, 5))
    sns.histplot(df['log_return'].dropna(), bins=100, kde=True)
    plt.title('Distribution of 1h Log Returns')
    plt.xlabel('Log Return')
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'btc_log_return_distribution.png'))
    plt.close()
    print("Saved Distribution of 1h Log Returns plot.")
else:
    print("'close' column not found for Log Return Distribution plot.")

# --- Visualize engineered features over time (first 5) ---
print(f"Visualizing the first {min(5, len(engineered_features))} engineered features over time.")
for i, feat in enumerate(engineered_features[:5]):
    plt.figure(figsize=(14, 3))
    plt.plot(df.index, df[feat], label=feat)
    plt.title(f'{feat} Over Time')
    plt.xlabel('Time')
    plt.ylabel(feat)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'btc_{feat}_over_time.png'))
    plt.close()
    print(f"Saved {feat} Over Time plot.")

# --- Enhanced correlation heatmap (all numeric features) ---
plt.figure(figsize=(16, 14))
# Ensure 'log_return' is excluded if it exists, otherwise get all numeric columns
columns_for_corr = [col for col in df.columns if df[col].dtype != 'O' and col != 'log_return']

if columns_for_corr:
    corr = df[columns_for_corr].corr()
    sns.heatmap(corr, annot=False, cmap='coolwarm', center=0)
    plt.title('Correlation Heatmap: All Numeric Features')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'btc_all_feature_correlation_heatmap.png'))
    plt.close()
    print("Saved Correlation Heatmap.")
else:
    print("No numeric features available for correlation heatmap.")

# --- Pairplot for selected engineered features (if not too many and exist) ---
selected_engineered = [f for f in engineered_features if f in df.columns and df[f].dtype != 'O'][:5]
if len(selected_engineered) > 1:
    # Sample data for pairplot if dataset is too large
    sample_df = df[selected_engineered].dropna()
    if len(sample_df) > 10000:
        sample_df = sample_df.sample(n=10000, random_state=42)
    if len(sample_df) > 1:
        sns.pairplot(sample_df)
        plt.suptitle('Pairplot of Selected Engineered Features', y=1.02)
        plt.tight_layout(rect=[0, 0.03, 1, 0.98]) # Adjust layout to prevent suptitle overlap
        plt.savefig(os.path.join(output_dir, 'btc_engineered_features_pairplot.png'))
        plt.close()
        print("Saved Engineered Features Pairplot.")
    else:
        print("Not enough valid data points for Engineered Features Pairplot after dropping NaNs.")
else:
    print("Not enough selected engineered features for Pairplot.")

# --- 6. Seasonality: Hour, Day of Week, Month ---
# Check if time features exist before plotting
if all(col in df.columns for col in ['hour', 'day_of_week', 'month']):
    # Ensure index is datetime for plotting
    if not isinstance(df.index, pd.DatetimeIndex):
         print("DataFrame index is not a DatetimeIndex, skipping seasonality plots.")
    else:
        df_reset = df.reset_index()
        plt.figure(figsize=(8, 4))
        sns.boxplot(x='hour', y='close', data=df_reset)
        plt.title('BTC Price by Hour of Day')
        plt.xlabel('Hour')
        plt.ylabel('Close Price')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'btc_price_by_hour.png'))
        plt.close()
        print("Saved Price by Hour of Day plot.")

        plt.figure(figsize=(8, 4))
        sns.boxplot(x='day_of_week', y='close', data=df_reset)
        plt.title('BTC Price by Day of Week')
        plt.xlabel('Day of Week (0=Mon)')
        plt.ylabel('Close Price')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'btc_price_by_dayofweek.png'))
        plt.close()
        print("Saved Price by Day of Week plot.")

        plt.figure(figsize=(8, 4))
        sns.boxplot(x='month', y='close', data=df_reset)
        plt.title('BTC Price by Month')
        plt.xlabel('Month')
        plt.ylabel('Close Price')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'btc_price_by_month.png'))
        plt.close()
        print("Saved Price by Month plot.")
else:
    print("Time features (hour, day_of_week, or month) not found for seasonality plots.")


print(f"Finished generating BTC data visualizations and summary statistics in {output_dir}")
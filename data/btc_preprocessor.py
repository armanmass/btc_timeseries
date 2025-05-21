from typing import Tuple, List, Optional
import pandas as pd
import numpy as np
from pydantic import BaseModel, Field
from sklearn.preprocessing import StandardScaler
import ta
from datetime import datetime, timedelta
import logging
from pathlib import Path
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PreprocessingConfig(BaseModel):
    """Configuration for data preprocessing"""
    input_window: int = Field(default=168, description="Number of hours to use as input")
    prediction_horizons: List[int] = Field(
        default=[24, 168, 720], 
        description="Prediction horizons in hours (1d, 1w, 1m)"
    )
    train_split: float = Field(default=0.7, description="Training set proportion")
    val_split: float = Field(default=0.15, description="Validation set proportion")
    # test_split will be 1 - train_split - val_split

class BTCData(BaseModel):
    """Processed BTC data structure"""
    features: np.ndarray
    targets: np.ndarray
    feature_names: List[str]
    target_names: List[str]
    scaler: StandardScaler
    timestamps: np.ndarray

    model_config = {
        "arbitrary_types_allowed": True
    }

def load_and_preprocess_data(
    file_path: str,
    config: PreprocessingConfig
) -> Tuple[BTCData, BTCData, BTCData]:
    """
    Load and preprocess BTC data from CSV file, with Parquet caching.
    """
    try:
        # Determine cache path (global for every file in data folder)
        base_name = os.path.splitext(os.path.basename(file_path))[0]
        cache_path = os.path.join(os.path.dirname(file_path), base_name + '.parquet')
        use_cache = False
        if os.path.exists(cache_path):
            csv_mtime = os.path.getmtime(file_path)
            parquet_mtime = os.path.getmtime(cache_path)
            if parquet_mtime > csv_mtime:
                use_cache = True
        if use_cache:
            logger.info(f"Loading data from cache: {cache_path}")
            df = pd.read_parquet(cache_path)
        else:
            # Read data in chunks to handle large file
            logger.info(f"Loading data from {file_path}")
            chunks = pd.read_csv(
                file_path,
                chunksize=100000  # Adjust based on available memory
            )
            # Process first chunk to get column names
            first_chunk = next(chunks)
            # Lowercase all column names to handle capitalization issues
            first_chunk.columns = [col.lower() for col in first_chunk.columns]
            required_columns = ['open', 'high', 'low', 'close', 'volume']
            if not all(col in first_chunk.columns for col in required_columns):
                raise ValueError(f"CSV must contain columns: {required_columns}")
            # Combine all chunks, ensuring all columns are lowercase
            df = pd.concat([
                first_chunk
            ] + [chunk.rename(columns={col: col.lower() for col in chunk.columns}) for chunk in chunks])
            logger.info(f"Loaded {len(df)} rows of data")
            # Convert Timestamp to datetime and set as index
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
            df = df.set_index('timestamp')
            # Save to Parquet cache
            df.to_parquet(cache_path)
            logger.info(f"Cached data to {cache_path}")
        # Sort by timestamp
        df = df.sort_index()
        # Handle missing values
        df = handle_missing_values(df)
        # Add technical indicators
        df = add_technical_indicators(df)
        # Create time features
        df = add_time_features(df)
        # Create target variables for different horizons
        df = create_target_variables(df, config.prediction_horizons)
        # Split data
        train_data, val_data, test_data = split_data(df, config)
        return train_data, val_data, test_data
    except Exception as e:
        logger.error(f"Error preprocessing data: {str(e)}")
        raise

def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Handle missing values in the dataset"""
    # Forward fill for price data
    price_cols = ['open', 'high', 'low', 'close']
    df[price_cols] = df[price_cols].ffill()
    
    # Fill remaining missing values with 0 for volume
    df['volume'] = df['volume'].fillna(0)
    
    # Drop any remaining rows with missing values
    df = df.dropna()
    
    return df

def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add technical indicators to the dataset"""
    # RSI
    df['rsi'] = ta.momentum.RSIIndicator(df['close']).rsi()
    
    # MACD
    macd = ta.trend.MACD(df['close'])
    df['macd'] = macd.macd()
    df['macd_signal'] = macd.macd_signal()
    df['macd_diff'] = macd.macd_diff()
    
    # Bollinger Bands
    bollinger = ta.volatility.BollingerBands(df['close'])
    df['bb_high'] = bollinger.bollinger_hband()
    df['bb_low'] = bollinger.bollinger_lband()
    df['bb_mid'] = bollinger.bollinger_mavg()
    
    # ATR
    df['atr'] = ta.volatility.AverageTrueRange(
        df['high'], df['low'], df['close']
    ).average_true_range()
    
    # Volume indicators
    df['volume_sma'] = ta.volume.volume_weighted_average_price(
        df['high'], df['low'], df['close'], df['volume']
    )
    
    return df

def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add time-based features"""
    # Ensure index is a DatetimeIndex
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    df['hour'] = df.index.hour
    df['day_of_week'] = df.index.dayofweek
    df['day_of_month'] = df.index.day
    df['month'] = df.index.month
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    
    # Cyclical encoding of time features
    df['hour_sin'] = np.sin(2 * np.pi * df['hour']/24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour']/24)
    df['day_sin'] = np.sin(2 * np.pi * df['day_of_week']/7)
    df['day_cos'] = np.cos(2 * np.pi * df['day_of_week']/7)
    
    return df

def create_target_variables(
    df: pd.DataFrame,
    horizons: List[int]
) -> pd.DataFrame:
    """Create target variables for different prediction horizons"""
    for horizon in horizons:
        # Future price
        df[f'price_{horizon}h'] = df['close'].shift(-horizon)
        
        # Future returns
        df[f'return_{horizon}h'] = (
            df[f'price_{horizon}h'] / df['close'] - 1
        )
        
        # Future volatility (using high-low range)
        df[f'volatility_{horizon}h'] = (
            df['high'].rolling(horizon).max() / 
            df['low'].rolling(horizon).min() - 1
        ).shift(-horizon)
    
    return df

def split_data(
    df: pd.DataFrame,
    config: PreprocessingConfig
) -> Tuple[BTCData, BTCData, BTCData]:
    """Split data into train, validation, and test sets"""
    # Calculate split indices
    n = len(df)
    train_end = int(n * config.train_split)
    val_end = int(n * (config.train_split + config.val_split))
    
    # Split data
    train_df = df.iloc[:train_end]
    val_df = df.iloc[train_end:val_end]
    test_df = df.iloc[val_end:]
    
    # Prepare feature and target columns
    feature_cols = [col for col in df.columns if not any(
        col.startswith(prefix) for prefix in ['price_', 'return_', 'volatility_']
    )]
    target_cols = [col for col in df.columns if any(
        col.startswith(prefix) for prefix in ['price_', 'return_', 'volatility_']
    )]
    
    # Scale features
    scaler = StandardScaler()
    train_features = scaler.fit_transform(train_df[feature_cols])
    val_features = scaler.transform(val_df[feature_cols])
    test_features = scaler.transform(test_df[feature_cols])
    
    # Create BTCData objects
    train_data = BTCData(
        features=train_features,
        targets=train_df[target_cols].values,
        feature_names=feature_cols,
        target_names=target_cols,
        scaler=scaler,
        timestamps=train_df.index.values
    )
    
    val_data = BTCData(
        features=val_features,
        targets=val_df[target_cols].values,
        feature_names=feature_cols,
        target_names=target_cols,
        scaler=scaler,
        timestamps=val_df.index.values
    )
    
    test_data = BTCData(
        features=test_features,
        targets=test_df[target_cols].values,
        feature_names=feature_cols,
        target_names=target_cols,
        scaler=scaler,
        timestamps=test_df.index.values
    )
    
    return train_data, val_data, test_data 
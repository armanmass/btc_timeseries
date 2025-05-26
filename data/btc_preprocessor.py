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
import torch
from torch.utils.data import Dataset # Import PyTorch Dataset

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

class BTCDataset(Dataset):
    """Custom Dataset for BTC time series data."""
    def __init__(self, features: np.ndarray, targets: np.ndarray, input_window: int, prediction_horizons: List[int]):
        # Store the full feature and target arrays
        self.features = features
        self.targets = targets
        self.input_window = input_window
        self.prediction_horizons = prediction_horizons
        self.num_target_types = targets.shape[1] // len(prediction_horizons)

        # Calculate the number of possible sequences
        # A sequence ends at index i + input_window - 1
        # The furthest target is at index i + input_window + max(horizons) - 1
        # So the last valid starting index `i` is when i + input_window + max(horizons) - 1 < len(features)
        # i < len(features) - input_window - max(horizons) + 1
        # The number of sequences is this value
        self._num_sequences = len(features) - self.input_window - max(self.prediction_horizons) + 1
        
        # Ensure we don't have negative number of sequences if data is too short
        if self._num_sequences < 0:
            self._num_sequences = 0
            logger.warning("Data is too short to create any sequences with the given window and horizons.")

    def __len__(self):
        return self._num_sequences

    def __getitem__(self, idx):
        # For a given sequence index `idx`,
        # the input window is from `idx` to `idx + input_window`
        input_seq = self.features[idx:(idx + self.input_window)]
        
        # The targets are for the steps AFTER the input window ends
        # For horizon H, the target is at index (idx + input_window + H - 1)
        current_targets = []
        for horizon in self.prediction_horizons:
             target_idx = idx + self.input_window + horizon - 1
             # This index should always be valid based on how _num_sequences is calculated
             
             # Add targets for this horizon (e.g., price, return, volatility)
             start_col = self.prediction_horizons.index(horizon) * self.num_target_types
             end_col = start_col + self.num_target_types
             current_targets.extend(self.targets[target_idx, start_col:end_col])
        
        # Convert to torch tensors
        input_seq_tensor = torch.FloatTensor(input_seq)
        targets_tensor = torch.FloatTensor(current_targets)

        # Add validation to check for NaN/Inf in tensors
        if not torch.isfinite(input_seq_tensor).all():
            logger.error(f"NaN or Inf found in input sequence at index {idx}")
            # Depending on severity, you might want to raise an error or handle it differently
            # For now, we'll print an error and continue, but this needs investigation if it occurs
        if not torch.isfinite(targets_tensor).all():
            logger.error(f"NaN or Inf found in target tensor at index {idx}")
            # Same as above, investigate if this occurs

        return input_seq_tensor, targets_tensor

def load_and_preprocess_data(
    file_path: str,
    config: PreprocessingConfig
) -> Tuple[pd.DataFrame, BTCData, BTCData, BTCData]:
    """
    Load and preprocess BTC data from CSV file, with Parquet caching.
    Returns the full preprocessed DataFrame and the split train/val/test sets.
    """
    try:
        # Determine cache path for the initial DataFrame (global for every file in data folder)
        base_name = os.path.splitext(os.path.basename(file_path))[0]
        initial_cache_path = os.path.join(os.path.dirname(file_path), base_name + '_initial.parquet')
        full_processed_cache_path = os.path.join(os.path.dirname(file_path), base_name + '_full_processed.parquet')
        
        df = None # Initialize df

        # Try loading from full processed cache first
        if os.path.exists(full_processed_cache_path):
            csv_mtime = os.path.getmtime(file_path)
            parquet_mtime = os.path.getmtime(full_processed_cache_path)
            if parquet_mtime > csv_mtime:
                logger.info(f"Loading full processed data from cache: {full_processed_cache_path}")
                df = pd.read_parquet(full_processed_cache_path)

        if df is None: # If not loaded from full processed cache
             # Try loading initial DataFrame from cache
            if os.path.exists(initial_cache_path):
                csv_mtime = os.path.getmtime(file_path)
                parquet_mtime = os.path.getmtime(initial_cache_path)
                if parquet_mtime > csv_mtime:
                    logger.info(f"Loading initial data from cache: {initial_cache_path}")
                    df = pd.read_parquet(initial_cache_path)
            
            if df is None: # If not loaded from any cache
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
                # Save the initial processed DataFrame to cache
                df.to_parquet(initial_cache_path)
                logger.info(f"Cached initial data to {initial_cache_path}")
        
        # If df was loaded from initial cache or just processed, continue with feature engineering
        if full_processed_cache_path is not None and not os.path.exists(full_processed_cache_path): # Only proceed if full processed cache didn't exist

            logger.info("Proceeding with feature engineering...")
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

            # Save the full processed DataFrame to cache
            df.to_parquet(full_processed_cache_path)
            logger.info(f"Cached full processed data to {full_processed_cache_path}")


        # Split data
        train_data, val_data, test_data = split_data(df, config)
        
        return df, train_data, val_data, test_data # Return full df as well
        
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
        # Only create future price target for each horizon
        df[f'price_{horizon}h'] = df['close'].shift(-horizon)
    
    return df

def split_data(
    df: pd.DataFrame,
    config: PreprocessingConfig
) -> Tuple[torch.utils.data.Dataset, torch.utils.data.Dataset, torch.utils.data.Dataset]:
    """
    Split data into train, validation, and test sets and create PyTorch Datasets.
    """
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
    # Ensure target columns are in a consistent order across splits
    # Extract all unique target prefixes and sort them
    target_prefixes = sorted(list(set(
        col.split('_')[0] for col in df.columns if col.startswith('price_')
    )))
    
    target_cols = []
    # Add target columns for each horizon in the specified order of prefixes and horizons
    for horizon in config.prediction_horizons:
        for prefix in target_prefixes:
             col_name = f'{prefix}_{horizon}h'
             if col_name in df.columns: # Check if column exists (e.g., if data was too short for a horizon)
                 target_cols.append(col_name)

    # Drop rows where target values are NaN (happens at the end due to shifting)
    df_cleaned = df.dropna(subset=target_cols).copy()

    # Re-split data after dropping NaNs
    train_df_cleaned = df_cleaned.iloc[:train_end]
    val_df_cleaned = df_cleaned.iloc[train_end:val_end]
    test_df_cleaned = df_cleaned.iloc[val_end:]

    # Scale features using scaler fitted ONLY on the training data (cleaned)
    scaler = StandardScaler()
    train_features_scaled = scaler.fit_transform(train_df_cleaned[feature_cols])
    val_features_scaled = scaler.transform(val_df_cleaned[feature_cols])
    test_features_scaled = scaler.transform(test_df_cleaned[feature_cols])
    
    # Scale targets as well
    # It's generally better to scale features and targets independently or use a multi-output scaler
    # For simplicity here, let's scale targets using the same StandardScaler, but be mindful
    # A dedicated target scaler (like MinMaxScaler) might be more appropriate depending on the target distribution.
    # Let's use a separate scaler for targets for better practice.
    target_scaler = StandardScaler()
    train_targets_scaled = target_scaler.fit_transform(train_df_cleaned[target_cols])
    val_targets_scaled = target_scaler.transform(val_df_cleaned[target_cols])
    test_targets_scaled = target_scaler.transform(test_df_cleaned[target_cols])

    # Create sequences
    # Now, the sequences are created within the Dataset __getitem__ method on the fly.

    # Create PyTorch Datasets
    train_dataset = BTCDataset(train_features_scaled, train_targets_scaled, config.input_window, config.prediction_horizons)
    val_dataset = BTCDataset(val_features_scaled, val_targets_scaled, config.input_window, config.prediction_horizons)
    test_dataset = BTCDataset(test_features_scaled, test_targets_scaled, config.input_window, config.prediction_horizons)

    logger.info(f"Train sequences shape: {len(train_dataset)} sequences")
    logger.info(f"Validation sequences shape: {len(val_dataset)} sequences")
    logger.info(f"Test sequences shape: {len(test_dataset)} sequences")

    # Create PyTorch Datasets
    # train_dataset = BTCDataset(train_X, train_y)
    # val_dataset = BTCDataset(val_X, val_y)
    # test_dataset = BTCDataset(test_X, test_y)
    
    # Note: We are not returning the scaler here, but you might need it for inference
    # to inverse transform predictions. Consider returning it if needed.
    
    return train_dataset, val_dataset, test_dataset 
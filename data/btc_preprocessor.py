from typing import Dict, List, Tuple, Optional, Union
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import ta
from ta.trend import SMAIndicator, EMAIndicator, MACD
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.volatility import BollingerBands
from ta.volume import VolumeWeightedAveragePrice
import logging
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BTCDataPreprocessor:
    """Preprocesses BTC price data for time series forecasting."""
    
    def __init__(
        self,
        data_path: Union[str, Path],
        sequence_length: int = 168,  # 1 week of hourly data
        prediction_horizons: List[int] = [24, 168, 720],  # 1d, 1w, 1m
        train_split: float = 0.7,
        val_split: float = 0.15,
        test_split: float = 0.15,
        random_state: int = 42
    ) -> None:
        """
        Initialize the preprocessor.
        
        Args:
            data_path: Path to the CSV file containing BTC price data
            sequence_length: Number of time steps to use as input
            prediction_horizons: List of prediction horizons in hours
            train_split: Proportion of data to use for training
            val_split: Proportion of data to use for validation
            test_split: Proportion of data to use for testing
            random_state: Random seed for reproducibility
        """
        self.data_path = Path(data_path)
        self.sequence_length = sequence_length
        self.prediction_horizons = prediction_horizons
        self.train_split = train_split
        self.val_split = val_split
        self.test_split = test_split
        self.random_state = random_state
        
        # Validate splits
        if not np.isclose(train_split + val_split + test_split, 1.0):
            raise ValueError("Train, validation, and test splits must sum to 1.0")
        
        # Initialize scalers
        self.price_scaler = StandardScaler()
        self.feature_scaler = StandardScaler()
        
        # Store processed data
        self.raw_data: Optional[pd.DataFrame] = None
        self.processed_data: Optional[pd.DataFrame] = None
        self.feature_columns: List[str] = []
        self.target_columns: List[str] = []
        
    def load_data(self) -> pd.DataFrame:
        """Load and validate the raw data."""
        logger.info(f"Loading data from {self.data_path}")
        
        try:
            df = pd.read_csv(self.data_path, index_col="time", parse_dates=True)
        except Exception as e:
            raise ValueError(f"Failed to load data from {self.data_path}: {str(e)}")
        
        # Validate required columns
        required_columns = ["open", "high", "low", "close", "volume"]
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        # Sort by time and remove duplicates
        df = df.sort_index()
        df = df[~df.index.duplicated(keep='first')]
        
        # Check for missing values
        missing_values = df[required_columns].isnull().sum()
        if missing_values.any():
            logger.warning(f"Found missing values:\n{missing_values}")
            # Forward fill missing values
            df = df.fillna(method='ffill')
            # If still missing values at the start, backfill
            df = df.fillna(method='bfill')
        
        self.raw_data = df
        logger.info(f"Loaded {len(df)} data points from {df.index[0]} to {df.index[-1]}")
        return df
    
    def add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add technical indicators to the dataset."""
        logger.info("Adding technical indicators")
        
        # Price-based indicators
        df['returns'] = df['close'].pct_change()
        df['log_returns'] = np.log1p(df['returns'])
        
        # Moving averages
        for window in [20, 50, 100, 200]:
            df[f'sma_{window}'] = SMAIndicator(close=df['close'], window=window).sma_indicator()
            df[f'ema_{window}'] = EMAIndicator(close=df['close'], window=window).ema_indicator()
        
        # MACD
        macd = MACD(close=df['close'])
        df['macd'] = macd.macd()
        df['macd_signal'] = macd.macd_signal()
        df['macd_diff'] = macd.macd_diff()
        
        # RSI
        df['rsi'] = RSIIndicator(close=df['close']).rsi()
        
        # Bollinger Bands
        bb = BollingerBands(close=df['close'])
        df['bb_high'] = bb.bollinger_hband()
        df['bb_low'] = bb.bollinger_lband()
        df['bb_mid'] = bb.bollinger_mavg()
        df['bb_width'] = (df['bb_high'] - df['bb_low']) / df['bb_mid']
        
        # Stochastic Oscillator
        stoch = StochasticOscillator(high=df['high'], low=df['low'], close=df['close'])
        df['stoch_k'] = stoch.stoch()
        df['stoch_d'] = stoch.stoch_signal()
        
        # Volume indicators
        df['vwap'] = VolumeWeightedAveragePrice(
            high=df['high'],
            low=df['low'],
            close=df['close'],
            volume=df['volume']
        ).volume_weighted_average_price()
        
        # Volatility
        df['volatility'] = df['returns'].rolling(window=24).std()  # 24-hour volatility
        
        # Time features
        df['hour'] = df.index.hour
        df['day_of_week'] = df.index.dayofweek
        df['day_of_month'] = df.index.day
        df['month'] = df.index.month
        
        # Remove rows with NaN values (from indicator calculation)
        df = df.dropna()
        
        logger.info(f"Added {len(df.columns) - len(self.raw_data.columns)} technical indicators")
        return df
    
    def create_sequences(
        self,
        df: pd.DataFrame
    ) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """
        Create sequences for time series forecasting.
        
        Returns:
            Tuple of (X, y) where:
            - X: Input sequences of shape (n_samples, sequence_length, n_features)
            - y: Dictionary of target sequences for each horizon
        """
        logger.info("Creating sequences for time series forecasting")
        
        # Define feature and target columns
        price_columns = ['open', 'high', 'low', 'close']
        volume_columns = ['volume']
        feature_columns = [col for col in df.columns 
                         if col not in price_columns + volume_columns + ['returns', 'log_returns']]
        
        # Scale the data
        price_data = self.price_scaler.fit_transform(df[price_columns])
        volume_data = self.feature_scaler.fit_transform(df[volume_columns])
        feature_data = self.feature_scaler.fit_transform(df[feature_columns])
        
        # Combine all features
        all_features = np.hstack([price_data, volume_data, feature_data])
        
        # Create sequences
        X, y = [], {}
        for horizon in self.prediction_horizons:
            y[horizon] = []
        
        for i in range(len(df) - self.sequence_length - max(self.prediction_horizons)):
            # Input sequence
            X.append(all_features[i:i + self.sequence_length])
            
            # Target sequences for each horizon
            for horizon in self.prediction_horizons:
                target_idx = i + self.sequence_length + horizon - 1
                if target_idx < len(df):
                    y[horizon].append(price_data[target_idx, 3])  # Use close price as target
        
        X = np.array(X)
        for horizon in self.prediction_horizons:
            y[horizon] = np.array(y[horizon])
        
        logger.info(f"Created {len(X)} sequences")
        logger.info(f"Input shape: {X.shape}")
        for horizon, arr in y.items():
            logger.info(f"Target shape for {horizon}h: {arr.shape}")
        
        return X, y
    
    def split_data(
        self,
        X: np.ndarray,
        y: Dict[str, np.ndarray]
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Dict[str, np.ndarray]]]:
        """
        Split data into train, validation, and test sets.
        
        Returns:
            Tuple of (X_split, y_split) where each is a dictionary containing
            train, validation, and test sets.
        """
        logger.info("Splitting data into train, validation, and test sets")
        
        # Calculate split indices
        n_samples = len(X)
        train_end = int(n_samples * self.train_split)
        val_end = train_end + int(n_samples * self.val_split)
        
        # Split input data
        X_split = {
            'train': X[:train_end],
            'val': X[train_end:val_end],
            'test': X[val_end:]
        }
        
        # Split target data for each horizon
        y_split = {
            'train': {},
            'val': {},
            'test': {}
        }
        
        for horizon in self.prediction_horizons:
            y_split['train'][horizon] = y[horizon][:train_end]
            y_split['val'][horizon] = y[horizon][train_end:val_end]
            y_split['test'][horizon] = y[horizon][val_end:]
        
        # Log split sizes
        for split in ['train', 'val', 'test']:
            logger.info(f"{split.capitalize()} set size: {len(X_split[split])}")
            for horizon in self.prediction_horizons:
                logger.info(f"{split.capitalize()} target size for {horizon}h: {len(y_split[split][horizon])}")
        
        return X_split, y_split
    
    def process(self) -> Tuple[Dict[str, np.ndarray], Dict[str, Dict[str, np.ndarray]]]:
        """
        Run the complete preprocessing pipeline.
        
        Returns:
            Tuple of (X_split, y_split) containing the processed and split data.
        """
        # Load and validate data
        df = self.load_data()
        
        # Add technical indicators
        df = self.add_technical_indicators(df)
        self.processed_data = df
        
        # Create sequences
        X, y = self.create_sequences(df)
        
        # Split data
        X_split, y_split = self.split_data(X, y)
        
        logger.info("Data preprocessing completed successfully")
        return X_split, y_split
    
    def save_processed_data(
        self,
        X_split: Dict[str, np.ndarray],
        y_split: Dict[str, Dict[str, np.ndarray]],
        output_dir: Union[str, Path]
    ) -> None:
        """Save processed data to disk."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save scalers
        np.save(output_dir / 'price_scaler_mean.npy', self.price_scaler.mean_)
        np.save(output_dir / 'price_scaler_scale.npy', self.price_scaler.scale_)
        np.save(output_dir / 'feature_scaler_mean.npy', self.feature_scaler.mean_)
        np.save(output_dir / 'feature_scaler_scale.npy', self.feature_scaler.scale_)
        
        # Save processed data
        for split in ['train', 'val', 'test']:
            np.save(output_dir / f'X_{split}.npy', X_split[split])
            for horizon in self.prediction_horizons:
                np.save(
                    output_dir / f'y_{split}_{horizon}h.npy',
                    y_split[split][horizon]
                )
        
        # Save metadata
        metadata = {
            'sequence_length': self.sequence_length,
            'prediction_horizons': self.prediction_horizons,
            'feature_columns': self.feature_columns,
            'target_columns': self.target_columns,
            'train_split': self.train_split,
            'val_split': self.val_split,
            'test_split': self.test_split,
            'random_state': self.random_state
        }
        pd.Series(metadata).to_json(output_dir / 'metadata.json')
        
        logger.info(f"Saved processed data to {output_dir}")

if __name__ == '__main__':
    # Example usage
    preprocessor = BTCDataPreprocessor(
        data_path="../btc_usd_5y_hourly_kraken.csv",  # Updated path to look in parent directory
        sequence_length=168,  # 1 week
        prediction_horizons=[24, 168, 720],  # 1d, 1w, 1m
    )
    
    # Process data
    X_split, y_split = preprocessor.process()
    
    # Save processed data
    preprocessor.save_processed_data(X_split, y_split, "processed")  # Save in data/processed directory 
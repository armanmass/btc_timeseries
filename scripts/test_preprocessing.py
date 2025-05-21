import sys
from pathlib import Path
import logging
from data.btc_preprocessor import (
    load_and_preprocess_data,
    PreprocessingConfig,
    BTCData
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """Test the BTC data preprocessing pipeline"""
    try:
        # Get the data file path (always relative to project root)
        data_file = Path(__file__).parent.parent / "data" / "btcusd_1-min_data.csv"
        if not data_file.exists():
            raise FileNotFoundError(f"Data file not found: {data_file}")
        
        # Create preprocessing config
        config = PreprocessingConfig(
            input_window=168,  # 1 week of hourly data
            prediction_horizons=[24, 168, 720],  # 1d, 1w, 1m
            train_split=0.7,
            val_split=0.15
        )
        
        # Load and preprocess data
        logger.info("Starting data preprocessing...")
        train_data, val_data, test_data = load_and_preprocess_data(
            str(data_file),
            config
        )
        
        # Print dataset statistics
        logger.info("\nDataset Statistics:")
        logger.info(f"Training set size: {len(train_data.features)} samples")
        logger.info(f"Validation set size: {len(val_data.features)} samples")
        logger.info(f"Test set size: {len(test_data.features)} samples")
        logger.info(f"\nNumber of features: {len(train_data.feature_names)}")
        logger.info(f"Number of targets: {len(train_data.target_names)}")
        
        # Print feature names
        logger.info("\nFeature names:")
        for i, name in enumerate(train_data.feature_names):
            logger.info(f"{i+1}. {name}")
        
        # Print target names
        logger.info("\nTarget names:")
        for i, name in enumerate(train_data.target_names):
            logger.info(f"{i+1}. {name}")
        
        # Print data ranges
        logger.info("\nFeature value ranges (training set):")
        for i, name in enumerate(train_data.feature_names):
            min_val = train_data.features[:, i].min()
            max_val = train_data.features[:, i].max()
            logger.info(f"{name}: [{min_val:.2f}, {max_val:.2f}]")
        
        logger.info("\nPreprocessing completed successfully!")
        
    except Exception as e:
        logger.error(f"Error in preprocessing pipeline: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main() 
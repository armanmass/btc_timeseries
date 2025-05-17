# BTC Price Prediction Plan using Transformer Models

## 1. Current State
- We have hourly BTC/USD price data from Kraken (2015-present)
- Data includes: OHLCV (Open, High, Low, Close, Volume)
- Single RTX 2080 GPU (8GB VRAM) available for training
- Need to predict prices for 1 day (24h), 1 week (168h), and 1 month (720h) horizons

## 2. Target State
- A transformer-based model that can predict BTC prices at multiple time horizons
- Model should be efficient enough to run on RTX 2080
- Predictions should include confidence intervals
- Model should be retrainable with new data
- Evaluation metrics: RMSE, MAE, and directional accuracy

## 3. Files to Create/Modify

### New Files:
1. `data/btc_preprocessor.py`
   - Data preprocessing and feature engineering
   - Time series window creation
   - Train/val/test split with proper time ordering
   - Feature scaling and normalization

2. `models/btc_transformer.py`
   - Custom transformer model architecture
   - Based on Time Series Transformer (TST) architecture
   - Optimized for GPU memory usage
   - Multi-horizon prediction heads

3. `training/trainer.py`
   - Training loop with gradient accumulation
   - Mixed precision training (FP16)
   - Early stopping and model checkpointing
   - Learning rate scheduling

4. `evaluation/metrics.py`
   - Custom evaluation metrics
   - Backtesting framework
   - Confidence interval calculation
   - Visualization tools

5. `config/model_config.py`
   - Model hyperparameters
   - Training configuration
   - Data processing parameters

6. `scripts/train.py`
   - Main training script
   - Command line interface
   - Experiment logging

7. `scripts/predict.py`
   - Inference script
   - Real-time prediction pipeline
   - Model serving

### Modify:
1. `btcproject.py`
   - Add data export functionality for ML pipeline
   - Add prediction visualization

## 4. Implementation Plan

### Phase 1: Data Preparation (2 days)
- [ ] Create data preprocessing pipeline
  - [ ] Handle missing values and outliers
  - [ ] Create technical indicators (RSI, MACD, Bollinger Bands)
  - [ ] Add market sentiment features (if available)
  - [ ] Create proper time series windows
  - [ ] Implement train/val/test split (70/15/15)
  - [ ] Add data augmentation techniques

### Phase 2: Model Development (3 days)
- [ ] Implement custom transformer architecture
  - [ ] Use Time Series Transformer (TST) as base
  - [ ] Optimize for GPU memory usage
  - [ ] Implement multi-horizon prediction
  - [ ] Add attention visualization
  - [ ] Implement confidence intervals

### Phase 3: Training Pipeline (2 days)
- [ ] Implement training infrastructure
  - [ ] Set up mixed precision training
  - [ ] Implement gradient accumulation
  - [ ] Add learning rate scheduling
  - [ ] Set up early stopping
  - [ ] Implement model checkpointing
  - [ ] Add experiment logging (Weights & Biases)

### Phase 4: Evaluation & Optimization (2 days)
- [ ] Implement evaluation metrics
  - [ ] RMSE and MAE for point predictions
  - [ ] Directional accuracy
  - [ ] Confidence interval coverage
  - [ ] Backtesting framework
- [ ] Model optimization
  - [ ] Hyperparameter tuning
  - [ ] Architecture optimization
  - [ ] Memory usage optimization

### Phase 5: Deployment & Monitoring (1 day)
- [ ] Create inference pipeline
- [ ] Implement real-time predictions
- [ ] Add model monitoring
- [ ] Create visualization dashboard

## 5. Technical Details

### Model Architecture
- Base: Time Series Transformer (TST)
- Input: 168 hours of historical data (1 week)
- Output: Predictions for 24h, 168h, and 720h horizons
- Features:
  - Price data (OHLCV)
  - Technical indicators
  - Time features (hour, day, week, month)
  - Market sentiment (if available)

### Training Strategy
- Mixed precision training (FP16)
- Gradient accumulation (batch size = 32)
- Learning rate: 1e-4 with cosine decay
- Optimizer: AdamW with weight decay
- Loss function: Custom loss combining:
  - MSE for point predictions
  - Quantile loss for confidence intervals
  - Directional loss for trend prediction

### Memory Optimization
- Gradient checkpointing
- Attention optimization
- Efficient data loading
- Model pruning after training

### Evaluation Strategy
- Walk-forward validation
- Multiple time horizons
- Confidence intervals
- Backtesting on historical data
- Comparison with baseline models (ARIMA, Prophet)

## 6. Additional Considerations
- Model interpretability
- Real-time prediction latency
- Regular retraining schedule
- Model versioning
- Error handling and logging
- Documentation

## 7. Success Metrics
- RMSE < 2% of current price
- Directional accuracy > 60%
- Training time < 4 hours on RTX 2080
- Inference time < 100ms
- Memory usage < 6GB VRAM

## 8. Potential Challenges
- Market volatility
- Limited training data
- GPU memory constraints
- Overfitting to recent trends
- Model interpretability

## 9. Future Improvements
- Add more data sources
- Implement ensemble methods
- Add market sentiment analysis
- Implement online learning
- Add more sophisticated features 
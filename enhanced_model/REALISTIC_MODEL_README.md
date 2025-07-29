# Realistic Enhanced Model for Cryptocurrency Prediction

## Overview

This is a **professional and realistic enhanced model** specifically designed for cryptocurrency price prediction with emphasis on capturing real-world market patterns. The model addresses the limitations of the previous "enhanced model" by incorporating time-of-day variations, market microstructure features, and realistic constraints that make predictions more aligned with actual market behavior.

## Key Improvements Over Previous Model

### 🕐 Time-Aware Features
- **Trading Hour Patterns**: Captures US trading hours (14:00-21:00 UTC), Asian trading hours (00:00-08:00 UTC), and European trading hours (08:00-16:00 UTC)
- **Weekend Effects**: Differentiates between weekday and weekend behavior
- **Time-of-Day Categories**: Morning, afternoon, evening, and night patterns
- **Market Session Overlaps**: Identifies periods when multiple markets are active

### 🏪 Market Microstructure
- **Bid-Ask Spread Proxies**: High-low ratios and price efficiency measures
- **Volume Patterns**: Volume-weighted average price (VWAP) and volume ratios
- **Price Momentum**: Multi-timeframe momentum indicators
- **Gap Analysis**: Identifies gap-up and gap-down patterns

### 🔄 Regime Detection
- **Volatility Regimes**: High, low, and normal volatility periods
- **Trend vs Mean-Reversion**: Identifies trending vs range-bound markets
- **Momentum Strength**: Measures the strength of price momentum
- **Market State Classification**: Above/below moving averages

### 📊 Multi-Scale Analysis
- **Short-term Patterns**: 5-15 minute patterns
- **Medium-term Patterns**: 30-60 minute patterns  
- **Long-term Patterns**: 2-4 hour patterns
- **Ultra-long-term Patterns**: Daily patterns

### 🎯 Realistic Constraints
- **Volatility Bounds**: 0.001 to 0.5 (realistic ranges)
- **Skewness Bounds**: -0.8 to 0.8 (realistic asymmetry)
- **Kurtosis Bounds**: 0.1 to 10.0 (realistic tail behavior)
- **Time-Aware Constraints**: Higher volatility during US trading hours

## Model Architecture

### Core Components

1. **TimeAwareAttention**: Multi-head attention mechanism that considers time-of-day patterns
2. **MarketRegimeDetector**: Detects and processes different market regimes
3. **MultiScaleProcessor**: Processes features at different temporal scales
4. **TemporalConsistencyLayer**: Ensures smooth predictions over time
5. **RealisticMomentPredictor**: Specialized predictor with realistic constraints

### Architecture Features

- **Input Size**: 15 features (increased from 10)
- **Hidden Size**: 256 (increased from 128)
- **Layers**: 4 (increased from 3)
- **Attention Heads**: 8
- **Dropout**: 0.2

## Feature Engineering

### Time-Based Features
```python
# Trading hour indicators
'us_trading_hours'          # US market hours (14:00-21:00 UTC)
'asian_trading_hours'       # Asian market hours (00:00-08:00 UTC)
'european_trading_hours'    # European market hours (08:00-16:00 UTC)
'is_weekend'               # Weekend indicator
'us_european_overlap'      # Market overlap periods
'asian_european_overlap'   # Market overlap periods

# Cyclical time encoding
'hour_sin', 'hour_cos'     # Hour of day encoding
'dow_sin', 'dow_cos'       # Day of week encoding
'month_sin', 'month_cos'   # Month encoding
```

### Market Microstructure Features
```python
# Spread and efficiency
'spread_proxy'             # High-low ratio
'price_efficiency'         # Price movement efficiency
'volume_ratio'            # Volume relative to average
'vwap'                    # Volume-weighted average price
'price_vwap_ratio'        # Price relative to VWAP

# Momentum indicators
'price_momentum_5'        # 5-period momentum
'price_momentum_12'       # 12-period momentum
'price_momentum_24'       # 24-period momentum
'gap_up', 'gap_down'      # Gap indicators
```

### Volatility Features
```python
# Multi-window volatility
'realized_vol_6'          # 30-minute volatility
'realized_vol_12'         # 1-hour volatility
'realized_vol_24'         # 2-hour volatility
'realized_vol_48'         # 4-hour volatility
'realized_vol_96'         # 8-hour volatility

# Advanced volatility measures
'parkinson_vol'           # Parkinson volatility
'garman_klass_vol'        # Garman-Klass volatility
'vol_of_vol'              # Volatility of volatility
'vol_clustering'          # Volatility clustering
```

### Regime Features
```python
# Volatility regimes
'high_vol_regime'         # High volatility periods
'low_vol_regime'          # Low volatility periods

# Price regimes
'above_ma'                # Above moving average
'below_ma'                # Below moving average

# Momentum regimes
'strong_momentum'         # Strong momentum periods
'weak_momentum'           # Weak momentum periods
'range_bound'             # Range-bound periods
'trending'                # Trending periods
```

## Usage

### Training the Model

```python
from enhanced_model.trainer import RealisticModelTrainer
from enhanced_model.config import RealisticConfig

# Initialize configuration
config = RealisticConfig()

# Train for specific cryptocurrency
trainer = RealisticModelTrainer(config, crypto_symbol='BTC')
df = trainer.load_and_preprocess_data()
X_train, X_val, y_train, y_val, time_train, time_val, feature_cols = trainer.prepare_training_data(df)
trainer.train(X_train, X_val, y_train, y_val, time_train, time_val, feature_cols)
```

### Making Predictions

```python
from enhanced_model.continuous_predictor import RealisticContinuousPredictor
from enhanced_model.config import RealisticConfig

# Initialize predictor
config = RealisticConfig()
predictor = RealisticContinuousPredictor(config, crypto_symbol='BTC')

# Generate 288 predictions (24 hours)
predictions_data = predictor.generate_288_predictions()

# Access predictions
for prediction in predictions_data['predictions']:
    volatility = prediction['predicted_volatility']
    skewness = prediction['predicted_skewness']
    kurtosis = prediction['predicted_kurtosis']
    print(f"Vol: {volatility:.4f}, Skew: {skewness:.4f}, Kurt: {kurtosis:.4f}")
```

### Continuous Prediction

```python
# Run continuous prediction every 5 minutes
predictor.run_continuous_prediction(interval_minutes=5)
```

## Configuration

### Model Parameters
```python
class RealisticConfig:
    # Model architecture
    INPUT_SIZE = 15           # Number of input features
    HIDDEN_SIZE = 256         # Hidden layer size
    NUM_LAYERS = 4           # Number of LSTM layers
    DROPOUT = 0.2            # Dropout rate
    
    # Training parameters
    BATCH_SIZE = 32          # Batch size
    LEARNING_RATE = 0.0005   # Learning rate
    NUM_EPOCHS = 150         # Number of epochs
    EARLY_STOPPING_PATIENCE = 20
    
    # Loss function weights
    VOLATILITY_WEIGHT = 2.0
    SKEWNESS_WEIGHT = 1.0
    KURTOSIS_WEIGHT = 1.5
    UNCERTAINTY_WEIGHT = 0.1
    CONSISTENCY_WEIGHT = 0.3
    REGIME_WEIGHT = 0.2
    
    # Realistic constraints
    MIN_VOLATILITY = 0.001
    MAX_VOLATILITY = 0.5
    MAX_SKEWNESS = 0.8
    MIN_KURTOSIS = 0.1
    MAX_KURTOSIS = 10.0
```

## Supported Cryptocurrencies

- **BTC**: Bitcoin
- **ETH**: Ethereum  
- **SOL**: Solana
- **XAU**: Gold

## Key Features

### ✅ Realistic Predictions
- Captures time-of-day variations in volatility
- Higher volatility during US trading hours
- Lower volatility during Asian trading hours
- Weekend effects properly modeled

### ✅ Market-Aware Constraints
- Volatility bounds based on real market data
- Skewness constraints for realistic asymmetry
- Kurtosis constraints for realistic tail behavior
- Time-aware constraints for trading hours

### ✅ Multi-Scale Analysis
- Short-term patterns (5-15 minutes)
- Medium-term patterns (30-60 minutes)
- Long-term patterns (2-4 hours)
- Ultra-long-term patterns (daily)

### ✅ Regime Detection
- Automatic detection of market regimes
- Different processing for different regimes
- Regime-specific feature weighting

### ✅ Temporal Consistency
- Smooth predictions over time
- Prevents zigzag patterns
- Adaptive smoothing based on market conditions

## Performance Improvements

### Training Efficiency
- **Faster Convergence**: Better loss function design
- **Stable Training**: Improved weight initialization
- **Early Stopping**: Adaptive patience based on validation loss
- **Learning Rate Scheduling**: ReduceLROnPlateau with patience

### Prediction Quality
- **Realistic Ranges**: All predictions within realistic bounds
- **Time Consistency**: Smooth transitions between predictions
- **Market Awareness**: Predictions reflect actual market behavior
- **Uncertainty Quantification**: Confidence measures for predictions

## Comparison with Previous Model

| Feature | Previous Model | Realistic Model |
|---------|---------------|-----------------|
| Time Awareness | Basic | Advanced (trading hours, weekends) |
| Market Microstructure | Limited | Comprehensive |
| Regime Detection | None | Automatic |
| Multi-Scale Analysis | Basic | Advanced |
| Realistic Constraints | Basic | Comprehensive |
| Temporal Consistency | Basic | Advanced |
| Model Capacity | 128 hidden | 256 hidden |
| Training Stability | Moderate | High |

## File Structure

```
enhanced_model/
├── enhanced_model.py          # Realistic model architecture
├── feature_engineering.py     # Realistic feature engineering
├── trainer.py                 # Realistic model trainer
├── continuous_predictor.py    # Realistic continuous predictor
├── config.py                  # Realistic configuration
├── database_manager.py        # Database operations
├── data_processor.py          # Data preprocessing
├── utils.py                   # Utility functions
├── kurtosis_smoothing.py      # Kurtosis smoothing utilities
└── models/                    # Trained models
    ├── BTC_realistic_model.pth
    ├── ETH_realistic_model.pth
    ├── SOL_realistic_model.pth
    └── XAU_realistic_model.pth
```

## Getting Started

1. **Install Dependencies**:
   ```bash
   pip install torch pandas numpy scikit-learn matplotlib seaborn pymongo python-dotenv
   ```

2. **Train the Model**:
   ```bash
   cd enhanced_model
   python trainer.py
   ```

3. **Run Continuous Prediction**:
   ```bash
   python continuous_predictor.py
   ```

4. **Use in Your Code**:
   ```python
   from enhanced_model.continuous_predictor import RealisticContinuousPredictor
   from enhanced_model.config import RealisticConfig
   
   config = RealisticConfig()
   predictor = RealisticContinuousPredictor(config, 'BTC')
   predictions = predictor.generate_288_predictions()
   ```

## Monitoring and Validation

### Quality Metrics
- **Zigzag Score**: Measures prediction smoothness
- **Realistic Bounds**: Ensures predictions are within realistic ranges
- **Time Consistency**: Validates temporal smoothness
- **Market Alignment**: Verifies predictions match market patterns

### Performance Tracking
- Training loss curves
- Validation metrics
- Prediction quality analysis
- Model performance over time

## Troubleshooting

### Common Issues

1. **Model Not Loading**:
   - Ensure model is trained first
   - Check file paths in config
   - Verify model file exists

2. **Feature Mismatch**:
   - Retrain model with current feature set
   - Check feature engineering pipeline
   - Verify data preprocessing

3. **Poor Predictions**:
   - Check data quality
   - Verify realistic constraints
   - Monitor training metrics

### Performance Optimization

1. **GPU Usage**: Model automatically uses GPU if available
2. **Batch Size**: Adjust based on available memory
3. **Sequence Length**: Optimize for your data size
4. **Feature Selection**: Remove unnecessary features

## Future Enhancements

- **Online Learning**: Continuous model updates
- **Ensemble Methods**: Multiple model combination
- **Advanced Regimes**: More sophisticated regime detection
- **External Data**: News sentiment, social media
- **Cross-Asset**: Multi-asset correlation modeling

## Contributing

This model is designed to be easily extensible. Key areas for contribution:

1. **New Features**: Additional market microstructure features
2. **Model Architecture**: Improved neural network designs
3. **Loss Functions**: Better training objectives
4. **Data Sources**: Additional data integration
5. **Validation**: Enhanced evaluation metrics

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

For questions and support:
1. Check the troubleshooting section
2. Review the configuration options
3. Examine the example usage
4. Monitor the training logs

---

**Note**: This realistic model represents a significant improvement over the previous enhanced model, specifically addressing the time-of-day variations and market microstructure patterns that make predictions more aligned with real-world cryptocurrency market behavior. 
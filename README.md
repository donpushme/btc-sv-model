# Bitcoin Volatility Prediction for Monte Carlo Simulation

A comprehensive AI-powered system for predicting Bitcoin volatility, skewness, and kurtosis using deep learning techniques. This project is specifically designed for Monte Carlo simulations with real-time predictions that capture the cyclical nature of Bitcoin trading patterns.

## 🚀 Features

- **LSTM-based Neural Network** with attention mechanism for time series prediction
- **Real-time Volatility Prediction** with 24-hour horizon and 5-minute intervals
- **Monte Carlo Simulation** capabilities using predicted statistical moments
- **Intraday Pattern Recognition** capturing US/Asian trading hours effects
- **Advanced Feature Engineering** with 80+ technical indicators and market microstructure features
- **Real-time Risk Assessment** with market regime classification
- **MongoDB Integration** for persistent storage of predictions and training data
- **Continuous Learning** with automatic model retraining on new data
- **Production-Ready Architecture** with graceful shutdown and error handling
- **Comprehensive Analytics** and performance monitoring dashboard

## 📊 What It Predicts

The model predicts three key statistical moments for Bitcoin price changes:

1. **Volatility (σ)** - Price movement intensity
2. **Skewness** - Asymmetry in returns distribution
3. **Kurtosis** - Tail heaviness (fat-tail events)

These predictions enable realistic Monte Carlo simulations that capture:
- Higher volatility during US trading hours (9:30 AM - 4:00 PM EST)
- Lower volatility during Asian night hours
- Weekend effects and holiday patterns
- Market stress periods and regime changes

## 🛠️ Installation

### Prerequisites

- Python 3.8 or higher
- CUDA-compatible GPU (optional, for faster training)

### Setup

1. **Install Python (Windows users):**
   If you get "Python was not found" error on Windows:
   - Download Python from [python.org](https://www.python.org/downloads/)
   - During installation, check "Add Python to PATH"
   - Or install via Microsoft Store: `python3`
   - Verify installation: `python --version` or `python3 --version`

2. **Clone the repository:**
```bash
git clone <repository-url>
cd bitcoin-volatility-prediction
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
# or on some systems:
python -m pip install -r requirements.txt
```

4. **Create project directories and check system:**
```bash
python utils.py
# or if python command doesn't work:
python3 utils.py
```

This will:
- Create the necessary project directories
- Check your Python and package versions
- Verify PyTorch installation and GPU availability

5. **Setup MongoDB (for database features):**

**Option A: Local MongoDB**
```bash
# Windows (using Chocolatey)
choco install mongodb

# macOS (using Homebrew)
brew install mongodb-community

# Ubuntu/Debian
sudo apt install mongodb

# Start MongoDB service
mongod
```

**Option B: MongoDB Atlas (Cloud)**
- Create free account at [MongoDB Atlas](https://www.mongodb.com/atlas)
- Create a cluster and get connection string
- Update `.env` file with your connection string

6. **Configure database connection:**
```bash
python database_manager.py
```

This will create a `.env` file where you can configure your MongoDB connection.

This will create the following directory structure:
```
├── data/           # Bitcoin price data
├── models/         # Trained model checkpoints
├── results/        # Training results and plots
├── logs/           # Training logs
└── .env           # Database configuration
```

## 📁 Project Structure

```
bitcoin-volatility-prediction/
├── config.py                  # Configuration parameters
├── data_processor.py          # Data loading and preprocessing
├── feature_engineering.py     # Advanced feature creation
├── model.py                   # Neural network architecture
├── trainer.py                 # Training pipeline
├── predictor.py               # Basic real-time prediction interface
├── realtime_predictor.py      # Enhanced predictor with DB & online learning
├── database_manager.py        # MongoDB operations and data persistence
├── realtime_example.py        # Production-ready real-time system
├── utils.py                   # Utility functions and Monte Carlo simulation
├── requirements.txt           # Python dependencies
├── README.md                 # This file
├── example_usage.py          # Complete basic usage example
└── .env                      # Database configuration (created automatically)
```

## 📋 Data Requirements

Your Bitcoin price data should be in CSV format with the following columns:

| Column    | Description                    | Example                  |
|-----------|--------------------------------|--------------------------|
| timestamp | DateTime in any standard format | 2024-01-01 12:00:00    |
| open      | Opening price                  | 45000.50                |
| close     | Closing price                  | 45150.25                |
| high      | Highest price                  | 45200.00                |
| low       | Lowest price                   | 44980.75                |

**Recommended data characteristics:**
- **Frequency**: 5-minute intervals
- **Duration**: Minimum 30 days for training
- **Quality**: Clean data with minimal gaps

You can obtain Bitcoin data from:
- [CoinGecko API](https://www.coingecko.com/en/api)
- [Binance API](https://binance-docs.github.io/apidocs/)
- [Yahoo Finance](https://finance.yahoo.com/) (using `yfinance` library)

## 🚀 Quick Start

### 1. Prepare Your Data

Place your Bitcoin price data as `data/bitcoin_price_data.csv`:

```python
import pandas as pd

# Example: Download Bitcoin data using yfinance
import yfinance as yf

# Download 5-minute Bitcoin data
btc = yf.download("BTC-USD", interval="5m", period="60d")
btc.reset_index(inplace=True)
btc.columns = ['timestamp', 'open', 'high', 'low', 'close', 'adj_close', 'volume']
btc = btc[['timestamp', 'open', 'close', 'high', 'low']]
btc.to_csv('data/bitcoin_price_data.csv', index=False)
```

### 2. Train the Model

```bash
python trainer.py
```

This will:
- Load and preprocess your data
- Engineer 80+ features
- Train the LSTM model with early stopping
- Save the best model to `models/best_model.pth`
- Generate training plots in `results/`

### 3. Make Predictions

```python
from predictor import RealTimeVolatilityPredictor
import pandas as pd

# Load your latest Bitcoin data
data = pd.read_csv('data/bitcoin_price_data.csv')

# Initialize predictor
predictor = RealTimeVolatilityPredictor()

# Make prediction
prediction = predictor.predict_next_period(data)

print(f"Predicted Volatility: {prediction['predicted_volatility']:.4f}")
print(f"Market Regime: {prediction['market_regime']}")
print(f"Risk Level: {prediction['risk_assessment']}")
```

### 4. Run Monte Carlo Simulation

```python
from utils import monte_carlo_simulation, plot_monte_carlo_results

# Use predicted values for simulation
simulation_results, summary_stats = monte_carlo_simulation(
    volatility=prediction['predicted_volatility'],
    skewness=prediction['predicted_skewness'],
    kurtosis=prediction['predicted_kurtosis'],
    initial_price=prediction['current_price'],
    intervals=288,  # 24 hours
    num_simulations=1000
)

# Visualize results
plot_monte_carlo_results(
    simulation_results, 
    summary_stats, 
    prediction['current_price'],
    save_path='results/monte_carlo_simulation.png'
)

print(f"Expected final price: ${summary_stats['mean_final_price']:,.2f}")
print(f"Probability of profit: {summary_stats['probability_profit']:.2%}")
```

### 5. MongoDB Integration & Continuous Learning

```python
from realtime_predictor import EnhancedRealTimePredictor

# Initialize enhanced predictor with database and online learning
predictor = EnhancedRealTimePredictor(
    enable_database=True,
    enable_online_learning=True
)

# Make prediction and save to database
prediction = predictor.predict_and_save(data)

# Get prediction history from database
history = predictor.get_prediction_history(hours=24)
print(f"Retrieved {len(history)} predictions from database")

# Get analytics
analytics = predictor.get_prediction_analytics(hours=24)
print(f"Average volatility: {analytics['average_volatility']:.4f}")

# Force model retraining with new data
success = predictor.force_retrain()

# Start background tasks (monitoring, cleanup, retraining)
predictor.start_scheduled_tasks()
```

### 6. Production Deployment

```python
from realtime_example import ProductionBitcoinPredictor

# Initialize production predictor
production = ProductionBitcoinPredictor()

# Start real-time monitoring (fetches live data every 30 minutes)
production.start_realtime_monitoring(prediction_interval_minutes=30)

# The system will:
# - Fetch real-time Bitcoin data from Yahoo Finance
# - Make predictions and save to database
# - Automatically retrain model when needed
# - Monitor performance and cleanup old data
# - Handle graceful shutdown on Ctrl+C
```

## ⚙️ Configuration

Modify `config.py` to adjust model parameters:

```python
class Config:
    # Model architecture
    HIDDEN_SIZE = 128        # LSTM hidden units
    NUM_LAYERS = 3          # LSTM layers
    SEQUENCE_LENGTH = 144   # Input sequence length (12 hours)
    
    # Training
    BATCH_SIZE = 32
    LEARNING_RATE = 0.001
    NUM_EPOCHS = 100
    
    # Prediction
    PREDICTION_HORIZON = 288  # 24 hours output
```

## 🔄 Real-time Usage

For real-time predictions, integrate the predictor into your trading system:

```python
import schedule
import time
from predictor import RealTimeVolatilityPredictor

predictor = RealTimeVolatilityPredictor()

def make_hourly_prediction():
    # Fetch latest Bitcoin data
    data = get_latest_bitcoin_data()  # Your data fetching function
    
    # Make prediction
    prediction = predictor.predict_next_period(data)
    
    # Store or use prediction
    save_prediction(prediction)  # Your storage function

# Schedule predictions every hour
schedule.every().hour.do(make_hourly_prediction)

while True:
    schedule.run_pending()
    time.sleep(60)
```

## 📊 Model Performance

The model tracks several metrics during training:

- **Validation Loss**: Overall prediction accuracy
- **R² Scores**: Individual performance for volatility, skewness, kurtosis
- **Component Losses**: Weighted loss for each target variable

Example training results:
```
Epoch 90/100
  Train Loss: 0.023456
  Val Loss: 0.028901
  Val R² - Vol: 0.7543, Skew: 0.4892, Kurt: 0.3567
```

## 🎯 Advanced Features

### MongoDB Database Integration

Persistent storage for all predictions and model data:

```python
from database_manager import DatabaseManager

# Initialize database
db = DatabaseManager()

# Save predictions automatically
prediction = predictor.predict_and_save(data)

# Query prediction history
recent_predictions = db.get_recent_predictions(hours=24)

# Get analytics
analytics = predictor.get_prediction_analytics(hours=24)
print(f"Average volatility: {analytics['average_volatility']:.4f}")

# Database cleanup
db.cleanup_old_data()
```

**Database Collections:**
- `predictions`: All volatility predictions with timestamps
- `training_data`: Historical price data for model retraining  
- `models`: Model metadata and training metrics
- `performance`: Model performance tracking over time

### Continuous Learning & Auto-Retraining

The system automatically improves with new data:

```python
# Enhanced predictor with online learning
predictor = EnhancedRealTimePredictor(enable_online_learning=True)

# System automatically:
# - Collects new price data from predictions
# - Triggers retraining when conditions are met
# - Updates model with latest market patterns
# - Maintains model version history

# Manual retraining
success = predictor.force_retrain()

# Check retraining status
status = predictor.get_system_status()
print(f"Last retrain: {status['last_retrain']}")
print(f"New data points: {status['new_data_count']}")
```

**Retraining Triggers:**
- Time-based: Every 24 hours (configurable)
- Data-based: After 288 new data points (24 hours worth)
- Performance-based: When accuracy degrades significantly

### Production-Ready Deployment

Real-time system with fault tolerance:

```python
# Production system
python realtime_example.py

# Features:
# - Automatic data fetching from Yahoo Finance
# - Graceful error handling and recovery
# - Background task scheduling
# - Performance monitoring
# - Database connection pooling
# - Memory management and cleanup
```

### Intraday Pattern Prediction

Generate volatility patterns for specific time periods:

```python
# Predict next 4 hours with 5-minute resolution
pattern = predictor.predict_intraday_pattern(data, intervals=48)
print(pattern.head())
```

### Batch Predictions

Process multiple datasets:

```python
datasets = [data1, data2, data3]
predictions = predictor.batch_predict(datasets)
```

### Market Regime Classification

The model automatically classifies market conditions:

- `high_volatility_skewed`: High volatility with significant skewness
- `high_volatility_normal`: High volatility, normal distribution
- `low_volatility_stable`: Low volatility, stable conditions
- `medium_volatility_fat_tails`: Medium volatility with fat tails

## 🚨 Risk Assessment

Automated risk level classification:

- **Low**: Normal market conditions
- **Medium**: Elevated volatility or moderate skewness
- **High**: High volatility with extreme moments
- **Very High**: Extreme market stress conditions

## 🚀 Production Commands

### Available Scripts

```bash
# Basic training and prediction
python trainer.py                    # Train the model
python predictor.py                  # Demo basic prediction
python example_usage.py             # Complete example workflow

# Enhanced real-time system
python realtime_predictor.py        # Demo enhanced predictor
python realtime_example.py          # Production system
python realtime_example.py demo     # Analytics demo

# Database management
python database_manager.py          # Setup database and test connection
python utils.py                     # System check and project setup
```

### Configuration Files

```bash
# Core configuration
config.py                           # Model and training parameters

# Database configuration  
.env                                 # MongoDB connection and settings

# Dependencies
requirements.txt                     # Python packages
```

## 🔧 Troubleshooting

### Common Issues

1. **"No model found" error**
   ```bash
   # Ensure you've trained a model first
   python trainer.py
   ```

2. **CUDA out of memory**
   ```python
   # Reduce batch size in config.py
   BATCH_SIZE = 16  # or smaller
   ```

3. **MongoDB connection failed**
   ```bash
   # Check MongoDB is running
   mongod --version
   
   # Start MongoDB service
   # Windows: net start MongoDB
   # macOS: brew services start mongodb-community  
   # Linux: sudo systemctl start mongod
   
   # Check connection string in .env file
   MONGODB_URI=mongodb://localhost:27017/
   ```

4. **Database features disabled**
   ```bash
   # Install MongoDB driver
   pip install pymongo
   
   # Test database connection
   python database_manager.py
   
   # Enable in config.py
   ENABLE_DATABASE = True
   ```

5. **Data validation errors**
   ```python
   from utils import validate_bitcoin_data
   validation = validate_bitcoin_data(your_data)
   print(validation['errors'])
   ```

6. **Feature mismatch during prediction**
   - Ensure your prediction data has the same format as training data
   - Check for missing timestamps or data gaps

7. **Data download errors with yfinance**
   ```python
   # If you get "Length mismatch" errors when downloading data
   # The example_usage.py script now handles different yfinance formats automatically
   # But you can also manually download data:
   
   import yfinance as yf
   import pandas as pd
   
   btc = yf.download("BTC-USD", interval="5m", period="30d")
   btc.reset_index(inplace=True)
   print("Available columns:", btc.columns.tolist())
   # Adjust column selection based on what's available
   ```

8. **Python not found on Windows**
   - Install Python from Microsoft Store or python.org
   - Ensure "Add Python to PATH" is checked during installation
   - Use `python3` instead of `python` if needed
   - Restart command prompt after installation

9. **PyTorch version compatibility issues**
   ```bash
   # If you get "ReduceLROnPlateau.__init__() got an unexpected keyword argument 'verbose'"
   # This has been fixed in the latest version, but if you encounter it:
   pip install torch>=2.0.0
   # or update your PyTorch installation:
   pip install --upgrade torch torchvision
   ```

10. **PyTorch gradient computation errors**
    ```bash
    # If you get "one of the variables needed for gradient computation has been modified by an inplace operation"
    # This error occurs when tensors are modified in-place during forward pass
    # This has been fixed in the model architecture, but if you encounter similar issues:
    # - Avoid using operations like tensor[index] = value
    # - Use tensor = torch.cat([...]) instead of in-place modifications
    # - Set torch.autograd.set_detect_anomaly(True) to debug gradient issues
    ```

11. **PyTorch 2.6+ weights_only loading error**
   ```bash
   # ERROR: "Weights only load failed" or "Unsupported global: sklearn.preprocessing"
   # CAUSE: PyTorch 2.6+ changed default security for torch.load() to prevent code execution
   # REASON: Our model checkpoints contain sklearn scalers (RobustScaler, StandardScaler)
   
   # ✅ SOLUTION: Already fixed in the codebase using safe_torch_load() utility
   # If you encounter this elsewhere in your code:
   
   # Option 1: Use our utility function
   from utils import safe_torch_load
   checkpoint = safe_torch_load('model.pth', map_location='cpu')
   
   # Option 2: Explicitly allow all objects (for trusted files only!)
   checkpoint = torch.load('model.pth', weights_only=False)
   
   # Option 3: Add safe globals for specific sklearn classes
   import torch.serialization
   from sklearn.preprocessing import RobustScaler, StandardScaler
   torch.serialization.add_safe_globals([RobustScaler, StandardScaler])
   checkpoint = torch.load('model.pth')
   
   # This error is common when:
   # - Using PyTorch 2.6+ with models trained on older versions
   # - Loading checkpoints that contain sklearn preprocessing objects
   # - Switching between different PyTorch versions
   ```

12. **Datetime accessor error during prediction**
     ```bash
     # ERROR: "Can only use .dt accessor with datetimelike values"
     # CAUSE: Timestamp column not properly converted to datetime format
     # REASON: Input data has timestamps as strings instead of datetime objects
     
     # ✅ SOLUTION: Already fixed in predictor.py and data_processor.py
     # If you encounter this with custom data:
     
     # Ensure timestamps are properly formatted
     import pandas as pd
     data['timestamp'] = pd.to_datetime(data['timestamp'])
     
     # Common timestamp formats that work:
     # - "2024-01-01 12:00:00"
     # - "2024-01-01T12:00:00Z"  
     # - Unix timestamps (will be auto-converted)
     
     # This error occurs when:
     # - Input CSV has timestamps as strings
     # - Datetime conversion fails due to invalid format
     # - Missing or corrupt timestamp data
     ```

13. **Online learning not working**
     ```bash
     # Check configuration
     ENABLE_ONLINE_LEARNING = True
     
     # Verify database is connected
     python database_manager.py
     
     # Check retraining conditions
     status = predictor.get_system_status()
     print(f"New data points: {status['new_data_count']}")
     
     # Force retraining manually
     success = predictor.force_retrain()
     ```

### Performance Optimization

- **GPU Training**: Ensure PyTorch detects your GPU
- **Data Size**: Use at least 30 days of 5-minute data for good results
- **Memory**: 8GB+ RAM recommended for large datasets

## 📈 Model Architecture

The system uses a sophisticated LSTM architecture:

```
Input → Feature Engineering → LSTM Layers → Attention → Separate Heads → Output
  ↓            ↓                  ↓            ↓           ↓              ↓
OHLC Data → 80+ Features → Temporal Learning → Focus → Vol/Skew/Kurt → Predictions
```

**Key Components:**
- **Feature Engineering**: Technical indicators, volatility measures, time features
- **LSTM Backbone**: 3-layer bidirectional LSTM with dropout
- **Attention Mechanism**: Focuses on relevant time periods
- **Multi-head Output**: Separate prediction heads for each target
- **Custom Loss**: Weighted combination optimizing all three targets

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🔗 References

- [LSTM Networks for Time Series](https://arxiv.org/abs/1506.02025)
- [Attention Mechanisms](https://arxiv.org/abs/1706.03762)
- [Bitcoin Volatility Modeling](https://www.sciencedirect.com/science/article/pii/S0378426619302791)
- [Monte Carlo Methods in Finance](https://link.springer.com/book/10.1007/978-3-662-05071-6)

## 💬 Support

For questions and support:
- Create an issue on GitHub
- Check the troubleshooting section above
- Review the example usage script

---

**⚡ Ready to predict Bitcoin volatility like a pro? Start with the Quick Start guide above!** 
import torch
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class EnhancedConfig:
    # Multi-crypto support
    SUPPORTED_CRYPTOS = {
        'BTC': {
            'name': 'Bitcoin',
            'pyth_symbol': 'Crypto.BTC/USD',
            'db_table': 'btc_enhanced',
            'data_file': 'bitcoin_5min.csv'
        },
        'ETH': {
            'name': 'Ethereum', 
            'pyth_symbol': 'Crypto.ETH/USD',
            'db_table': 'eth_enhanced',
            'data_file': 'ethereum_5min.csv'
        },
        'XAU': {
            'name': 'Gold',
            'pyth_symbol': 'Metal.XAU/USD',
            'db_table': 'xau_enhanced',
            'data_file': 'xau_5min.csv'
        },
        'SOL': {
            'name': 'Solana',
            'pyth_symbol': 'Crypto.SOL/USD', 
            'db_table': 'sol_enhanced',
            'data_file': 'solana_5min.csv'
        }
    }
    
    # Default crypto for backward compatibility
    DEFAULT_CRYPTO = 'BTC'
    
    # Data parameters
    SEQUENCE_LENGTH = 144  # 12 hours of 5-minute intervals for context
    PREDICTION_HORIZON = 288  # 24 hours of 5-minute intervals to predict
    INTERVAL_MINUTES = 5
    
    # Enhanced model parameters
    INPUT_SIZE = 10  # OHLC + derived features + time features
    HIDDEN_SIZE = 128
    NUM_LAYERS = 3
    OUTPUT_SIZE = 3  # volatility, skewness, kurtosis
    DROPOUT = 0.2
    
    # Training parameters
    BATCH_SIZE = 16  # Reduced for better memory efficiency
    LEARNING_RATE = 0.001
    NUM_EPOCHS = 100
    VALIDATION_SPLIT = 0.2
    EARLY_STOPPING_PATIENCE = 15
    
    # Feature engineering
    VOLATILITY_WINDOW = 24  # 2 hours for rolling volatility
    RETURN_WINDOWS = [6, 12, 24, 48]  # Different windows for rolling features
    
    # Time features
    USE_TIME_FEATURES = True
    USE_CYCLICAL_ENCODING = True
    
    # Device
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Enhanced paths (point to root directories)
    MODEL_SAVE_PATH = 'models/'  # Enhanced models saved in enhanced_model/models directory
    DATA_PATH = '../training_data/'  # Use root training_data directory
    RESULTS_PATH = 'results/'  # Enhanced results in enhanced_model directory
    
    # Real-time prediction
    MIN_HISTORICAL_DATA_HOURS = 24
    
    # Database settings (read from environment)
    ENABLE_DATABASE = os.getenv('ENABLE_DATABASE', 'true').lower() == 'true'
    DATABASE_NAME = os.getenv('DATABASE_NAME', 'synth_prediction_enhanced')
    
    # Online learning settings (read from environment)
    ENABLE_ONLINE_LEARNING = os.getenv('ENABLE_ONLINE_LEARNING', 'true').lower() == 'true'
    RETRAIN_INTERVAL_HOURS = int(os.getenv('RETRAIN_INTERVAL_HOURS', '24'))
    MIN_NEW_DATA_POINTS = int(os.getenv('MIN_NEW_DATA_POINTS', '288'))
    PERFORMANCE_THRESHOLD = float(os.getenv('PERFORMANCE_THRESHOLD', '0.05'))
    MAX_TRAINING_DATA_HOURS = int(os.getenv('MAX_TRAINING_DATA_HOURS', '720'))
    
    # Retraining settings for limited data
    RETRAIN_WITH_LIMITED_DATA = os.getenv('RETRAIN_WITH_LIMITED_DATA', 'true').lower() == 'true'
    RETRAIN_MIN_DATA_POINTS = int(os.getenv('RETRAIN_MIN_DATA_POINTS', '20'))
    RETRAIN_SMALL_WINDOWS = [4, 6, 12]  # Minimum 4 for kurtosis calculation
    RETRAIN_NORMAL_WINDOWS = [6, 12, 24, 48]  # Normal windows for sufficient data
    
    # Data retention settings (read from environment)
    PREDICTION_RETENTION_DAYS = int(os.getenv('PREDICTION_RETENTION_DAYS', '90'))
    TRAINING_DATA_RETENTION_DAYS = int(os.getenv('TRAINING_DATA_RETENTION_DAYS', '180'))
    PERFORMANCE_DATA_RETENTION_DAYS = int(os.getenv('PERFORMANCE_DATA_RETENTION_DAYS', '365'))
    
    # Model versioning (read from environment) - Disabled to use simple naming
    AUTO_MODEL_VERSIONING = os.getenv('AUTO_MODEL_VERSIONING', 'false').lower() == 'true'
    KEEP_MODEL_VERSIONS = int(os.getenv('KEEP_MODEL_VERSIONS', '1')) 
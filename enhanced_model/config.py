import torch
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class RealisticConfig:
    # Multi-crypto support
    SUPPORTED_CRYPTOS = {
        'BTC': {
            'name': 'Bitcoin',
            'pyth_symbol': 'Crypto.BTC/USD',
            'db_table': 'btc_realistic',
            'data_file': 'bitcoin_5min.csv'
        },
        'ETH': {
            'name': 'Ethereum', 
            'pyth_symbol': 'Crypto.ETH/USD',
            'db_table': 'eth_realistic',
            'data_file': 'ethereum_5min.csv'
        },
        'XAU': {
            'name': 'Gold',
            'pyth_symbol': 'Metal.XAU/USD',
            'db_table': 'xau_realistic',
            'data_file': 'xau_5min.csv'
        },
        'SOL': {
            'name': 'Solana',
            'pyth_symbol': 'Crypto.SOL/USD', 
            'db_table': 'sol_realistic',
            'data_file': 'solana_5min.csv'
        }
    }
    
    # Default crypto for backward compatibility
    DEFAULT_CRYPTO = 'BTC'
    
    # Data parameters
    SEQUENCE_LENGTH = 144  # 12 hours of 5-minute intervals for context
    PREDICTION_HORIZON = 288  # 24 hours of 5-minute intervals to predict
    INTERVAL_MINUTES = 5
    
    # Realistic model parameters (increased capacity for better performance)
    INPUT_SIZE = 15  # Increased for more features
    HIDDEN_SIZE = 256  # Increased for better capacity
    NUM_LAYERS = 4  # Increased for deeper model
    OUTPUT_SIZE = 3  # volatility, skewness, kurtosis
    DROPOUT = 0.2
    
    # Training parameters
    BATCH_SIZE = 32  # Increased for better training
    LEARNING_RATE = 0.0005  # Reduced for more stable training
    NUM_EPOCHS = 150  # Increased for better convergence
    VALIDATION_SPLIT = 0.2
    EARLY_STOPPING_PATIENCE = 20  # Increased patience
    
    # Feature engineering
    VOLATILITY_WINDOW = 24  # 2 hours for rolling volatility
    RETURN_WINDOWS = [6, 12, 24, 48, 96]  # Different windows for rolling features
    
    # Time features
    USE_TIME_FEATURES = True
    USE_CYCLICAL_ENCODING = True
    
    # Device
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Realistic paths (point to root directories)
    MODEL_SAVE_PATH = 'models/'  # Realistic models saved in enhanced_model/models directory
    DATA_PATH = '../training_data/'  # Use root training_data directory
    RESULTS_PATH = 'results/'  # Realistic results in enhanced_model directory
    
    # Real-time prediction
    MIN_HISTORICAL_DATA_HOURS = 24
    
    # Database settings (read from environment)
    ENABLE_DATABASE = os.getenv('ENABLE_DATABASE', 'true').lower() == 'true'
    DATABASE_NAME = os.getenv('DATABASE_NAME', 'synth_prediction_realistic')
    
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
    
    # Realistic model specific settings
    USE_TIME_AWARE_ATTENTION = True
    USE_MARKET_REGIME_DETECTION = True
    USE_MULTI_SCALE_PROCESSING = True
    USE_TEMPORAL_CONSISTENCY = True
    
    # Loss function weights
    VOLATILITY_WEIGHT = 2.0
    SKEWNESS_WEIGHT = 1.0
    KURTOSIS_WEIGHT = 1.5
    UNCERTAINTY_WEIGHT = 0.1
    CONSISTENCY_WEIGHT = 0.3
    REGIME_WEIGHT = 0.2
    
    # Time-aware constraints
    US_TRADING_HOURS_START = 14  # 14:00 UTC (9:30 EST)
    US_TRADING_HOURS_END = 21    # 21:00 UTC (16:00 EST)
    ASIAN_TRADING_HOURS_START = 0  # 00:00 UTC
    ASIAN_TRADING_HOURS_END = 8    # 08:00 UTC
    
    # Realistic constraints
    MIN_VOLATILITY = 0.001
    MAX_VOLATILITY = 0.5
    MAX_SKEWNESS = 0.8
    MIN_KURTOSIS = 0.1
    MAX_KURTOSIS = 10.0

# Backward compatibility
EnhancedConfig = RealisticConfig 
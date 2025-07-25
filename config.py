import torch

class Config:
    # Data parameters
    SEQUENCE_LENGTH = 144  # 12 hours of 5-minute intervals for context
    PREDICTION_HORIZON = 288  # 24 hours of 5-minute intervals to predict
    INTERVAL_MINUTES = 5
    
    # Model parameters
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
    
    # Paths
    MODEL_SAVE_PATH = 'models/'
    DATA_PATH = 'data/'
    RESULTS_PATH = 'results/'
    
    # Real-time prediction
    MIN_HISTORICAL_DATA_HOURS = 24  # Minimum hours of data needed for prediction 
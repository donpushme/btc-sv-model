import pandas as pd
import numpy as np
from typing import Tuple, Optional
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class BitcoinDataProcessor:
    """
    Handles loading and preprocessing of Bitcoin price data for model training.
    """
    
    def __init__(self, csv_path: str):
        """
        Initialize with path to CSV file containing Bitcoin price data.
        Expected columns: timestamp, open, close, high, low
        """
        self.csv_path = csv_path
        self.data = None
        self.processed_data = None
        
    def load_data(self) -> pd.DataFrame:
        """Load Bitcoin price data from CSV file."""
        try:
            self.data = pd.read_csv(self.csv_path)
            print(f"Loaded {len(self.data)} records from {self.csv_path}")
            
            # Validate required columns
            required_cols = ['timestamp', 'open', 'close', 'high', 'low']
            missing_cols = [col for col in required_cols if col not in self.data.columns]
            if missing_cols:
                raise ValueError(f"Missing required columns: {missing_cols}")
            
            # Convert timestamp to datetime
            self.data['timestamp'] = pd.to_datetime(self.data['timestamp'])
            self.data = self.data.sort_values('timestamp').reset_index(drop=True)
            
            return self.data
            
        except Exception as e:
            raise Exception(f"Error loading data: {str(e)}")
    
    def calculate_returns(self) -> pd.DataFrame:
        """Calculate price returns and basic statistics."""
        if self.data is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        df = self.data.copy()
        
        # Calculate returns
        df['return'] = df['close'].pct_change()
        df['log_return'] = np.log(df['close'] / df['close'].shift(1))
        
        # Calculate price range and volatility proxy
        df['price_range'] = (df['high'] - df['low']) / df['open']
        df['true_range'] = np.maximum(
            df['high'] - df['low'],
            np.maximum(
                abs(df['high'] - df['close'].shift(1)),
                abs(df['low'] - df['close'].shift(1))
            )
        )
        
        # Volume proxy (using price range as we don't have volume data)
        df['volume_proxy'] = df['true_range'] * df['close']
        
        return df
    
    def calculate_rolling_statistics(self, df: pd.DataFrame, windows: list) -> pd.DataFrame:
        """Calculate rolling statistics for different time windows."""
        
        for window in windows:
            # Rolling volatility (standard deviation of returns)
            df[f'volatility_{window}'] = df['return'].rolling(window=window).std()
            
            # Rolling skewness
            df[f'skewness_{window}'] = df['return'].rolling(window=window).skew()
            
            # Rolling kurtosis
            df[f'kurtosis_{window}'] = df['return'].rolling(window=window).kurt()
            
            # Rolling mean and std
            df[f'return_mean_{window}'] = df['return'].rolling(window=window).mean()
            df[f'return_std_{window}'] = df['return'].rolling(window=window).std()
            
            # Price momentum
            df[f'price_momentum_{window}'] = df['close'] / df['close'].shift(window) - 1
        
        return df
    
    def add_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add time-based features to capture cyclical patterns."""
        
        # Ensure timestamp is datetime
        if 'timestamp' in df.columns and not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Extract time components
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['day_of_month'] = df['timestamp'].dt.day
        df['month'] = df['timestamp'].dt.month
        
        # Cyclical encoding for time features
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        
        # US trading hours indicator (high volatility periods)
        # US market hours: 9:30 AM - 4:00 PM EST (14:30 - 21:00 UTC)
        df['us_trading_hours'] = ((df['hour'] >= 14) & (df['hour'] <= 21)).astype(int)
        
        # Asian trading hours indicator
        df['asian_trading_hours'] = ((df['hour'] >= 0) & (df['hour'] <= 8)).astype(int)
        
        # Weekend indicator
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        
        return df
    
    def calculate_target_statistics(self, df: pd.DataFrame, 
                                  prediction_horizon: int = 288) -> pd.DataFrame:
        """
        Calculate future volatility, skewness, and kurtosis as targets.
        prediction_horizon: number of future periods to calculate statistics for
        """
        
        # Initialize target columns
        df['target_volatility'] = np.nan
        df['target_skewness'] = np.nan
        df['target_kurtosis'] = np.nan
        
        # Calculate targets for each time point
        for i in range(len(df) - prediction_horizon):
            future_returns = df['return'].iloc[i+1:i+1+prediction_horizon]
            
            if len(future_returns) == prediction_horizon and not future_returns.isna().any():
                volatility = future_returns.std()
                skewness = future_returns.skew()
                kurtosis = future_returns.kurt()  # This is excess kurtosis
                
                # Convert excess kurtosis to absolute kurtosis and apply bounds
                absolute_kurtosis = kurtosis + 3  # Convert to absolute kurtosis
                
                # Apply reasonable bounds for kurtosis (3 to 30)
                # Normal distribution has kurtosis = 3, extreme values capped at 30
                absolute_kurtosis = max(min(absolute_kurtosis, 30.0), 3.0)
                
                # Convert back to excess kurtosis for consistency
                excess_kurtosis = absolute_kurtosis - 3
                
                df.loc[i, 'target_volatility'] = volatility
                df.loc[i, 'target_skewness'] = skewness
                df.loc[i, 'target_kurtosis'] = excess_kurtosis
        
        return df
    
    def preprocess_data(self, return_windows: list = [6, 12, 24, 48],
                       prediction_horizon: int = 288) -> pd.DataFrame:
        """
        Complete preprocessing pipeline.
        """
        if self.data is None:
            self.load_data()
        
        print("Calculating returns...")
        df = self.calculate_returns()
        
        print("Adding time features...")
        df = self.add_time_features(df)
        
        print("Calculating rolling statistics...")
        df = self.calculate_rolling_statistics(df, return_windows)
        
        print("Calculating target statistics...")
        df = self.calculate_target_statistics(df, prediction_horizon)
        
        # Remove rows with NaN values
        initial_rows = len(df)
        df = df.dropna().reset_index(drop=True)
        final_rows = len(df)
        
        print(f"Removed {initial_rows - final_rows} rows with NaN values")
        print(f"Final dataset shape: {df.shape}")
        
        self.processed_data = df
        return df
    
    def get_feature_columns(self) -> list:
        """Return list of feature column names for model training."""
        if self.processed_data is None:
            raise ValueError("Data not processed. Call preprocess_data() first.")
        
        # Exclude timestamp and target columns
        exclude_cols = ['timestamp', 'target_volatility', 'target_skewness', 'target_kurtosis']
        feature_cols = [col for col in self.processed_data.columns if col not in exclude_cols]
        
        return feature_cols
    
    def get_target_columns(self) -> list:
        """Return list of target column names."""
        return ['target_volatility', 'target_skewness', 'target_kurtosis']
    
    def save_processed_data(self, path: str):
        """Save processed data to CSV file."""
        if self.processed_data is None:
            raise ValueError("No processed data to save.")
        
        self.processed_data.to_csv(path, index=False)
        print(f"Processed data saved to {path}") 
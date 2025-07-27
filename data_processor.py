import pandas as pd
import numpy as np
from typing import Tuple, Optional
from datetime import datetime, timedelta
import warnings
from config import Config
warnings.filterwarnings('ignore')

class CryptoDataProcessor:
    """
    Handles loading and preprocessing of cryptocurrency price data for model training.
    Supports multiple cryptocurrencies: BTC, ETH, XAU, SOL
    """
    
    def __init__(self, csv_path: str, crypto_symbol: str = 'BTC'):
        """
        Initialize with path to CSV file containing cryptocurrency price data.
        Expected columns: timestamp, open, close, high, low
        
        Args:
            csv_path: Path to CSV file
            crypto_symbol: Cryptocurrency symbol (BTC, ETH, XAU, SOL)
        """
        # Validate crypto symbol
        if crypto_symbol not in Config.SUPPORTED_CRYPTOS:
            raise ValueError(f"Unsupported crypto symbol: {crypto_symbol}. Supported: {list(Config.SUPPORTED_CRYPTOS.keys())}")
        
        self.csv_path = csv_path
        self.crypto_symbol = crypto_symbol
        self.crypto_config = Config.SUPPORTED_CRYPTOS[crypto_symbol]
        self.data = None
        self.processed_data = None
        
    def load_data(self) -> pd.DataFrame:
        """Load cryptocurrency price data from CSV file."""
        try:
            self.data = pd.read_csv(self.csv_path)
            print(f"Loaded {len(self.data)} records for {self.crypto_config['name']} ({self.crypto_symbol}) from {self.csv_path}")
            
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
            raise Exception(f"Error loading {self.crypto_symbol} data: {str(e)}")
    
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
        
        # For limited data, use a much shorter prediction horizon
        available_data = len(df)
        if available_data < prediction_horizon * 2:
            # If we have very limited data, use a much shorter horizon
            adaptive_horizon = max(12, min(available_data // 3, 48))  # Between 12 and 48 periods
            print(f"⚠️ Limited data ({available_data} points). Using adaptive prediction horizon: {adaptive_horizon}")
            prediction_horizon = adaptive_horizon
        
        # Calculate targets for each time point
        for i in range(len(df) - prediction_horizon):
            future_returns = df['return'].iloc[i+1:i+1+prediction_horizon]
            
            # More lenient tolerance for limited data
            min_required = max(6, prediction_horizon // 4)  # At least 6 periods or 25% of horizon
            
            if len(future_returns) >= min_required:
                # Remove any NaN values from future returns
                future_returns = future_returns.dropna()
                
                if len(future_returns) >= min_required:
                    volatility = future_returns.std()
                    skewness = future_returns.skew()
                    kurtosis = future_returns.kurt()  # This is excess kurtosis
                    
                    # Handle NaN values in statistics
                    if pd.isna(volatility) or pd.isna(skewness) or pd.isna(kurtosis):
                        continue
                    
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
        Complete preprocessing pipeline with enhanced robustness for limited data.
        """
        if self.data is None:
            self.load_data()
        
        print("Calculating returns...")
        df = self.calculate_returns()
        
        # Handle NaN values in returns more aggressively
        initial_rows = len(df)
        df['return'] = df['return'].fillna(method='ffill').fillna(method='bfill').fillna(0)
        df['log_return'] = df['log_return'].fillna(method='ffill').fillna(method='bfill').fillna(0)
        print(f"📊 Filled NaN values in returns. Rows: {initial_rows} -> {len(df)}")
        
        print("Adding time features...")
        df = self.add_time_features(df)
        
        # Adjust rolling windows for limited data
        available_data = len(df)
        if available_data < 500:
            print(f"⚠️ Limited data detected ({available_data} points). Using smaller rolling windows.")
            # Use smaller windows for limited data
            adjusted_windows = [w for w in return_windows if w <= available_data // 4]
            if not adjusted_windows:
                adjusted_windows = [6, 12]  # Fallback to very small windows
            return_windows = adjusted_windows
            print(f"📊 Adjusted rolling windows: {return_windows}")
        
        print("Calculating rolling statistics...")
        df = self.calculate_rolling_statistics(df, return_windows)
        
        print("Calculating target statistics...")
        df = self.calculate_target_statistics(df, prediction_horizon)
        
        # Check for NaN values before removal
        initial_rows = len(df)
        nan_counts = df.isna().sum()
        print(f"📊 NaN counts by column:")
        for col, count in nan_counts[nan_counts > 0].items():
            print(f"  {col}: {count} NaN values")
        
        # More aggressive NaN handling for limited data
        if initial_rows < 500:
            print(f"⚠️ Limited data detected. Using aggressive NaN handling...")
            
            # Fill NaN values in rolling statistics with forward/backward fill
            rolling_cols = [col for col in df.columns if any(window in col for window in ['volatility_', 'skewness_', 'kurtosis_', 'return_mean_', 'return_std_', 'price_momentum_'])]
            for col in rolling_cols:
                df[col] = df[col].fillna(method='ffill').fillna(method='bfill').fillna(0)
            
            # Fill target NaN values with reasonable defaults
            df['target_volatility'] = df['target_volatility'].fillna(0.02)  # 2% volatility
            df['target_skewness'] = df['target_skewness'].fillna(0.0)  # No skew
            df['target_kurtosis'] = df['target_kurtosis'].fillna(0.0)  # Normal kurtosis
            
            # Fill any remaining NaN values with 0
            df = df.fillna(0)
            
            print(f"📊 After aggressive NaN handling: {len(df)} data points")
        else:
            # Remove rows with NaN values for larger datasets
            df = df.dropna().reset_index(drop=True)
        
        final_rows = len(df)
        
        print(f"📊 Data preprocessing summary:")
        print(f"  Initial rows: {initial_rows}")
        print(f"  Final rows: {final_rows}")
        print(f"  Rows removed: {initial_rows - final_rows}")
        print(f"  Final dataset shape: {df.shape}")
        
        if final_rows == 0:
            raise ValueError(f"❌ No valid data points after preprocessing. Initial: {initial_rows}, Final: {final_rows}")
        
        if final_rows < 10:
            print(f"⚠️ Very limited data after preprocessing ({final_rows} points). This may affect model performance.")
        
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
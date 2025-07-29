#!/usr/bin/env python3
"""
Enhanced Data Processor for Monte Carlo Simulation

This module handles data preprocessing for the enhanced model architecture,
specifically designed for Monte Carlo simulation with better statistical moment prediction.
"""

import pandas as pd
import numpy as np
from typing import List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

class EnhancedCryptoDataProcessor:
    """
    Enhanced data processor for cryptocurrency volatility prediction.
    Optimized for Monte Carlo simulation with better statistical moment handling.
    """
    
    def __init__(self, csv_path: str, crypto_symbol: str = 'BTC'):
        """
        Initialize the data processor.
        
        Args:
            csv_path: Path to the CSV file containing OHLC data
            crypto_symbol: Cryptocurrency symbol (BTC, ETH, XAU, SOL)
        """
        self.csv_path = csv_path
        self.crypto_symbol = crypto_symbol
        self.df = None
        
    def load_data(self) -> pd.DataFrame:
        """
        Load and validate the CSV data.
        
        Returns:
            DataFrame with OHLC data
        """
        try:
            # Load CSV file
            df = pd.read_csv(self.csv_path)
            
            # Validate required columns
            required_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
            missing_columns = [col for col in required_columns if col not in df.columns]
            
            if missing_columns:
                raise ValueError(f"Missing required columns: {missing_columns}")
            
            # Convert timestamp to datetime
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            # Sort by timestamp
            df = df.sort_values('timestamp').reset_index(drop=True)
            
            # Remove duplicates
            df = df.drop_duplicates(subset=['timestamp']).reset_index(drop=True)
            
            self.df = df
            return df
            
        except Exception as e:
            raise Exception(f"Error loading data from {self.csv_path}: {str(e)}")
    
    def calculate_returns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate various return measures.
        
        Args:
            df: DataFrame with OHLC data
            
        Returns:
            DataFrame with return columns added
        """
        # Calculate returns (same as original)
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
    
    def calculate_rolling_statistics(self, df: pd.DataFrame, windows: List[int]) -> pd.DataFrame:
        """
        Calculate rolling statistics for different windows.
        
        Args:
            df: DataFrame with return data
            windows: List of window sizes for rolling calculations
            
        Returns:
            DataFrame with rolling statistics
        """
        for window in windows:
            # Ensure window is at least 4 for kurtosis calculation
            min_window = max(4, window)
            
            # Rolling volatility (standard deviation of returns)
            df[f'volatility_{window}'] = df['return'].rolling(window=window).std()
            
            # Rolling skewness (needs at least 3 points)
            if window >= 3:
                df[f'skewness_{window}'] = df['return'].rolling(window=window).skew()
            else:
                df[f'skewness_{window}'] = 0.0  # Default to no skew for very small windows
            
            # Rolling kurtosis (needs at least 4 points)
            if window >= 4:
                df[f'kurtosis_{window}'] = df['return'].rolling(window=window).kurt()
            else:
                df[f'kurtosis_{window}'] = 0.0  # Default to normal kurtosis for very small windows
            
            # Rolling momentum
            if window >= 2:
                df[f'momentum_{window}'] = df['close'].pct_change(window)
            else:
                df[f'momentum_{window}'] = 0.0
        
        return df
    
    def calculate_target_statistics(self, df: pd.DataFrame, prediction_horizon: int = 288) -> pd.DataFrame:
        """
        Calculate future volatility, skewness, and kurtosis as targets.
        prediction_horizon: number of future periods to calculate statistics for
        """
        print(f"Calculating target statistics with prediction_horizon={prediction_horizon}")
        
        # Initialize target columns
        df['target_volatility'] = np.nan
        df['target_skewness'] = np.nan
        df['target_kurtosis'] = np.nan
        
        # For limited data, use a much shorter prediction horizon
        available_data = len(df)
        if available_data < prediction_horizon + 100:  # Need at least horizon + 100 points for safety
            # If we have very limited data, use a much shorter horizon
            adaptive_horizon = max(6, min(available_data // 4, 24))  # Between 6 and 24 periods
            print(f"⚠️ Limited data ({available_data} points). Using adaptive prediction horizon: {adaptive_horizon}")
            prediction_horizon = adaptive_horizon
        else:
            print(f"✅ Sufficient data ({available_data} points). Using full prediction horizon: {prediction_horizon}")
        
        # Calculate targets for each time point
        # Use a more conservative approach for limited data
        max_start_idx = len(df) - prediction_horizon
        
        if max_start_idx <= 0:
            print(f"⚠️ Data too short for prediction horizon. Using all available data for targets.")
            # If data is too short, use all available data for each target
            for i in range(len(df)):
                if i < len(df) - 6:  # Need at least 6 points for statistics
                    future_returns = df['return'].iloc[i+1:].dropna()
                    if len(future_returns) >= 6:
                        volatility = future_returns.std()
                        skewness = future_returns.skew()
                        kurtosis = future_returns.kurt()
                        
                        if not pd.isna(volatility) and not pd.isna(skewness) and not pd.isna(kurtosis):
                            # Apply bounds to kurtosis
                            absolute_kurtosis = kurtosis + 3
                            absolute_kurtosis = max(min(absolute_kurtosis, 15.0), 3.0)
                            excess_kurtosis = absolute_kurtosis - 3
                            
                            df.loc[i, 'target_volatility'] = volatility
                            df.loc[i, 'target_skewness'] = skewness
                            df.loc[i, 'target_kurtosis'] = excess_kurtosis
        else:
            # Normal case: calculate targets for each time point
            for i in range(max_start_idx):
                future_returns = df['return'].iloc[i+1:i+1+prediction_horizon]
                
                # More lenient tolerance for limited data
                min_required = max(6, prediction_horizon // 3)  # At least 6 periods or 33% of horizon
                
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
                        
                        # Apply reasonable bounds for kurtosis (3 to 15)
                        # Normal distribution has kurtosis = 3, extreme values capped at 15 (more reasonable)
                        absolute_kurtosis = max(min(absolute_kurtosis, 15.0), 3.0)
                        
                        # Convert back to excess kurtosis for consistency
                        excess_kurtosis = absolute_kurtosis - 3
                        
                        df.loc[i, 'target_volatility'] = volatility
                        df.loc[i, 'target_skewness'] = skewness
                        df.loc[i, 'target_kurtosis'] = excess_kurtosis
        
        # Fill NaN targets with reasonable defaults
        df['target_volatility'] = df['target_volatility'].fillna(0.02)  # 2% volatility
        df['target_skewness'] = df['target_skewness'].fillna(0.0)  # No skew
        df['target_kurtosis'] = df['target_kurtosis'].fillna(0.0)  # Normal kurtosis
        
        print(f"Target statistics calculated. Sample values:")
        print(f"  Volatility sample: {df['target_volatility'].dropna().head(3).tolist()}")
        print(f"  Skewness sample: {df['target_skewness'].dropna().head(3).tolist()}")
        print(f"  Kurtosis sample: {df['target_kurtosis'].dropna().head(3).tolist()}")
        
        return df
    
    def add_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add time-based features for enhanced prediction.
        
        Args:
            df: DataFrame with timestamp column
            
        Returns:
            DataFrame with time features
        """
        # Extract time components
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['day_of_month'] = df['timestamp'].dt.day
        df['month'] = df['timestamp'].dt.month
        
        # Cyclical encoding for hour
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        
        # Cyclical encoding for day of week
        df['dow_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['dow_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        
        # US trading hours (14:00-21:00 UTC)
        df['us_trading_hours'] = ((df['hour'] >= 14) & (df['hour'] <= 21)).astype(int)
        
        # Asian trading hours (22:00-09:00 UTC)
        df['asian_trading_hours'] = ((df['hour'] >= 22) | (df['hour'] <= 9)).astype(int)
        
        # Weekend
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        
        return df
    
    def preprocess_data(self, return_windows: List[int] = [6, 12, 24, 48], 
                       prediction_horizon: int = 288) -> pd.DataFrame:
        """
        Complete preprocessing pipeline with enhanced robustness for limited data.
        """
        print("Loading data...")
        df = self.load_data()
        
        print("Calculating returns...")
        df = self.calculate_returns(df)
        
        # Handle NaN values in returns more aggressively
        initial_rows = len(df)
        df['return'] = df['return'].fillna(method='ffill').fillna(method='bfill').fillna(0)
        df['log_return'] = df['log_return'].fillna(method='ffill').fillna(method='bfill').fillna(0)
        
        print("Adding time features...")
        df = self.add_time_features(df)
        
        print("Calculating rolling statistics...")
        df = self.calculate_rolling_statistics(df, return_windows)
        
        print("Calculating target statistics...")
        df = self.calculate_target_statistics(df, prediction_horizon)
        
        # Handle NaN values aggressively for limited data
        if len(df) < 500:
            # For small datasets, be more aggressive with NaN handling
            numeric_columns = df.select_dtypes(include=[np.number]).columns
            df[numeric_columns] = df[numeric_columns].fillna(method='ffill').fillna(method='bfill').fillna(0)
        else:
            # For larger datasets, use standard NaN handling
            df = df.fillna(method='ffill').fillna(method='bfill')
        
        final_rows = len(df)
        print(f"Preprocessing complete. Initial rows: {initial_rows}, Final rows: {final_rows}")
        
        return df 
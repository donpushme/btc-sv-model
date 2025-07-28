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
        # Log returns
        df['log_return'] = np.log(df['close'] / df['close'].shift(1))
        
        # Simple returns
        df['simple_return'] = (df['close'] - df['close'].shift(1)) / df['close'].shift(1)
        
        # High-Low range
        df['hl_range'] = (df['high'] - df['low']) / df['close'].shift(1)
        
        # Open-Close range
        df['oc_range'] = (df['close'] - df['open']) / df['open']
        
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
            if len(df) < window:
                continue
                
            # Rolling volatility (standard deviation of returns)
            df[f'volatility_{window}'] = df['log_return'].rolling(window=window).std()
            
            # Rolling mean return
            df[f'mean_return_{window}'] = df['log_return'].rolling(window=window).mean()
            
            # Rolling skewness
            if window >= 3:
                df[f'skewness_{window}'] = df['log_return'].rolling(window=window).skew()
            else:
                df[f'skewness_{window}'] = 0.0
            
            # Rolling kurtosis (excess kurtosis)
            if window >= 4:
                df[f'kurtosis_{window}'] = df['log_return'].rolling(window=window).kurt()
            else:
                df[f'kurtosis_{window}'] = 0.0
            
            # Rolling momentum
            if window >= 2:
                df[f'momentum_{window}'] = df['close'].pct_change(window)
            else:
                df[f'momentum_{window}'] = 0.0
        
        return df
    
    def calculate_target_statistics(self, df: pd.DataFrame, prediction_horizon: int = 288) -> pd.DataFrame:
        """
        Calculate target statistics for future periods.
        
        Args:
            df: DataFrame with return data
            prediction_horizon: Number of periods to predict ahead
            
        Returns:
            DataFrame with target columns
        """
        # Calculate future returns for the prediction horizon
        future_returns = df['log_return'].shift(-prediction_horizon).rolling(window=prediction_horizon).apply(
            lambda x: x.dropna().tolist() if len(x.dropna()) >= prediction_horizon // 2 else np.nan
        )
        
        # Calculate target statistics
        df['target_volatility'] = future_returns.apply(
            lambda x: np.std(x) if isinstance(x, list) and len(x) >= prediction_horizon // 2 else np.nan
        )
        
        df['target_skewness'] = future_returns.apply(
            lambda x: float(pd.Series(x).skew()) if isinstance(x, list) and len(x) >= 3 else np.nan
        )
        
        df['target_kurtosis'] = future_returns.apply(
            lambda x: float(pd.Series(x).kurt()) if isinstance(x, list) and len(x) >= 4 else np.nan
        )
        
        # Ensure all target columns are numeric before applying bounds
        df['target_volatility'] = pd.to_numeric(df['target_volatility'], errors='coerce')
        df['target_skewness'] = pd.to_numeric(df['target_skewness'], errors='coerce')
        df['target_kurtosis'] = pd.to_numeric(df['target_kurtosis'], errors='coerce')
        
        # Convert kurtosis from excess to absolute, apply bounds, then back to excess
        # This ensures realistic kurtosis values for Monte Carlo simulation
        df['target_kurtosis'] = df['target_kurtosis'].apply(
            lambda x: np.clip(x + 3, 3.0, 15.0) - 3 if pd.notna(x) else np.nan
        )
        
        # Apply bounds to skewness
        df['target_skewness'] = df['target_skewness'].clip(-2.0, 2.0)
        
        # Apply bounds to volatility
        df['target_volatility'] = df['target_volatility'].clip(0.001, 0.1)
        
        # Fill NaN values with reasonable defaults
        df['target_volatility'].fillna(0.02, inplace=True)
        df['target_skewness'].fillna(0.0, inplace=True)
        df['target_kurtosis'].fillna(0.0, inplace=True)
        
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
        Complete data preprocessing pipeline.
        
        Args:
            return_windows: Windows for rolling statistics
            prediction_horizon: Number of periods to predict ahead
            
        Returns:
            Preprocessed DataFrame ready for training
        """
        # Load data
        df = self.load_data()
        
        # Calculate returns
        df = self.calculate_returns(df)
        
        # Add time features
        df = self.add_time_features(df)
        
        # Calculate rolling statistics
        df = self.calculate_rolling_statistics(df, return_windows)
        
        # Calculate target statistics
        df = self.calculate_target_statistics(df, prediction_horizon)
        
        # Handle NaN values aggressively for limited data
        if len(df) < 500:
            # For small datasets, be more aggressive with NaN handling
            numeric_columns = df.select_dtypes(include=[np.number]).columns
            df[numeric_columns] = df[numeric_columns].fillna(method='ffill').fillna(method='bfill').fillna(0)
        else:
            # For larger datasets, use standard NaN handling
            df = df.fillna(method='ffill').fillna(method='bfill')
        
        return df 
#!/usr/bin/env python3
"""
Enhanced Feature Engineering for Monte Carlo Simulation

This module provides advanced feature engineering capabilities specifically
designed for the enhanced model architecture and Monte Carlo simulation.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler, StandardScaler
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple, Dict, Optional
import torch
import warnings
warnings.filterwarnings('ignore')

class EnhancedFeatureEngineer:
    """
    Enhanced feature engineer for Monte Carlo simulation.
    """
    
    def __init__(self):
        self.feature_scaler = RobustScaler()
        self.target_scaler = RobustScaler()
        self.is_fitted = False
        
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Engineer advanced features for enhanced prediction.
        
        Args:
            df: Preprocessed DataFrame
            
        Returns:
            DataFrame with engineered features
        """
        # Technical indicators
        df = self.add_technical_indicators(df)
        
        # Volatility features
        df = self.add_volatility_features(df)
        
        # Market microstructure features
        df = self.add_microstructure_features(df)
        
        # Interaction features
        df = self.add_interaction_features(df)
        
        return df
    
    def add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add technical indicators for enhanced prediction.
        """
        # RSI
        df['rsi'] = self.calculate_rsi(df['close'], window=14)
        
        # MACD
        df['macd'], df['macd_signal'] = self.calculate_macd(df['close'])
        
        # Bollinger Bands
        df['bb_upper'], df['bb_lower'], df['bb_width'] = self.calculate_bollinger_bands(df['close'])
        
        # Stochastic Oscillator
        df['stoch_k'], df['stoch_d'] = self.calculate_stochastic(df['high'], df['low'], df['close'])
        
        # Williams %R
        df['williams_r'] = self.calculate_williams_r(df['high'], df['low'], df['close'])
        
        # Average True Range (ATR)
        df['atr'] = self.calculate_atr(df['high'], df['low'], df['close'])
        
        return df
    
    def add_volatility_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add volatility-related features.
        """
        # Realized volatility (different windows)
        for window in [6, 12, 24, 48]:
            df[f'realized_vol_{window}'] = df['log_return'].rolling(window=window).std()
        
        # Parkinson volatility (using high-low range)
        df['parkinson_vol'] = np.sqrt(
            (1 / (4 * np.log(2))) * 
            ((np.log(df['high'] / df['low']) ** 2).rolling(window=24).mean())
        )
        
        # Garman-Klass volatility
        df['garman_klass_vol'] = np.sqrt(
            (0.5 * (np.log(df['high'] / df['low']) ** 2) - 
             (2 * np.log(2) - 1) * (np.log(df['close'] / df['open']) ** 2)).rolling(window=24).mean()
        )
        
        # Volatility of volatility
        df['vol_of_vol'] = df['realized_vol_24'].rolling(window=48).std()
        
        return df
    
    def add_microstructure_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add market microstructure features.
        """
        # Volume-weighted features
        df['vwap'] = (df['close'] * df['volume']).rolling(window=24).sum() / df['volume'].rolling(window=24).sum()
        df['volume_ratio'] = df['volume'] / df['volume'].rolling(window=24).mean()
        
        # Price efficiency
        df['price_efficiency'] = abs(df['close'] - df['close'].shift(1)) / df['atr']
        
        # Spread proxy (high-low ratio)
        df['spread_proxy'] = (df['high'] - df['low']) / df['close']
        
        # Tick size effect
        df['tick_effect'] = (df['close'] % 0.01) / 0.01  # Assuming $0.01 tick size
        
        return df
    
    def add_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add interaction features between different indicators.
        """
        # Volatility-regime interactions
        df['vol_rsi_interaction'] = df['realized_vol_24'] * df['rsi']
        df['vol_macd_interaction'] = df['realized_vol_24'] * df['macd']
        
        # Time-volatility interactions
        df['hour_vol_interaction'] = df['hour'] * df['realized_vol_24']
        df['weekend_vol_interaction'] = df['is_weekend'] * df['realized_vol_24']
        
        # Volume-volatility interactions
        df['volume_vol_interaction'] = df['volume_ratio'] * df['realized_vol_24']
        
        return df
    
    def calculate_rsi(self, prices: pd.Series, window: int = 14) -> pd.Series:
        """Calculate RSI indicator."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def calculate_macd(self, prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series]:
        """Calculate MACD indicator."""
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        macd = ema_fast - ema_slow
        macd_signal = macd.ewm(span=signal).mean()
        return macd, macd_signal
    
    def calculate_bollinger_bands(self, prices: pd.Series, window: int = 20, std_dev: int = 2) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate Bollinger Bands."""
        sma = prices.rolling(window=window).mean()
        std = prices.rolling(window=window).std()
        upper_band = sma + (std * std_dev)
        lower_band = sma - (std * std_dev)
        bb_width = (upper_band - lower_band) / sma
        return upper_band, lower_band, bb_width
    
    def calculate_stochastic(self, high: pd.Series, low: pd.Series, close: pd.Series, k_window: int = 14, d_window: int = 3) -> Tuple[pd.Series, pd.Series]:
        """Calculate Stochastic Oscillator."""
        lowest_low = low.rolling(window=k_window).min()
        highest_high = high.rolling(window=k_window).max()
        k_percent = 100 * ((close - lowest_low) / (highest_high - lowest_low))
        d_percent = k_percent.rolling(window=d_window).mean()
        return k_percent, d_percent
    
    def calculate_williams_r(self, high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
        """Calculate Williams %R."""
        highest_high = high.rolling(window=window).max()
        lowest_low = low.rolling(window=window).min()
        williams_r = -100 * ((highest_high - close) / (highest_high - lowest_low))
        return williams_r
    
    def calculate_atr(self, high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
        """Calculate Average True Range."""
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=window).mean()
        return atr
    
    def fit_scalers(self, df: pd.DataFrame, feature_cols: List[str], target_cols: List[str]):
        """
        Fit scalers for features and targets.
        
        Args:
            df: Training DataFrame
            feature_cols: List of feature column names
            target_cols: List of target column names
        """
        # Fit feature scaler
        self.feature_scaler.fit(df[feature_cols])
        
        # Fit target scaler with enhanced transformation
        df_processed = df[target_cols].copy()
        
        # Apply enhanced transformations for targets
        df_processed['target_volatility'] = df_processed['target_volatility']  # Already in good range
        df_processed['target_skewness'] = df_processed['target_skewness']  # Already bounded
        df_processed['target_kurtosis'] = np.sqrt(df_processed['target_kurtosis'] + 1)  # Enhanced transformation
        
        self.target_scaler.fit(df_processed)
        self.is_fitted = True
    
    def transform_data(self, df: pd.DataFrame, feature_cols: List[str], target_cols: List[str]) -> pd.DataFrame:
        """
        Transform data using fitted scalers.
        
        Args:
            df: DataFrame to transform
            feature_cols: List of feature column names
            target_cols: List of target column names
            
        Returns:
            Transformed DataFrame
        """
        if not self.is_fitted:
            raise ValueError("Scalers must be fitted before transforming data")
        
        df_transformed = df.copy()
        
        # Transform features
        df_transformed[feature_cols] = self.feature_scaler.transform(df[feature_cols])
        
        # Transform targets with enhanced transformation
        df_processed = df[target_cols].copy()
        df_processed['target_volatility'] = df_processed['target_volatility']
        df_processed['target_skewness'] = df_processed['target_skewness']
        df_processed['target_kurtosis'] = np.sqrt(df_processed['target_kurtosis'] + 1)
        
        df_transformed[target_cols] = self.target_scaler.transform(df_processed)
        
        return df_transformed
    
    def inverse_transform_targets(self, targets: np.ndarray) -> np.ndarray:
        """
        Inverse transform targets to original scale.
        
        Args:
            targets: Transformed targets
            
        Returns:
            Targets in original scale
        """
        targets_original = self.target_scaler.inverse_transform(targets)
        
        # Apply inverse transformations
        targets_original[:, 2] = targets_original[:, 2]**2 - 1  # Kurtosis inverse
        
        return targets_original
    
    def prepare_sequences(self, df: pd.DataFrame, feature_cols: List[str], 
                         target_cols: List[str], sequence_length: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare sequences for LSTM training.
        
        Args:
            df: Transformed DataFrame
            feature_cols: List of feature column names
            target_cols: List of target column names
            sequence_length: Length of input sequences
            
        Returns:
            Tuple of (X, y) arrays
        """
        X, y = [], []
        
        for i in range(sequence_length, len(df)):
            # Input sequence
            X.append(df[feature_cols].iloc[i-sequence_length:i].values)
            
            # Target (next period)
            y.append(df[target_cols].iloc[i].values)
        
        return np.array(X), np.array(y)

class EnhancedCryptoDataset(Dataset):
    """
    Enhanced dataset for cryptocurrency volatility prediction.
    """
    
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def create_train_val_split(X: np.ndarray, y: np.ndarray, validation_split: float = 0.2) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Create train-validation split.
    
    Args:
        X: Input features
        y: Target values
        validation_split: Fraction of data for validation
        
    Returns:
        Tuple of (X_train, X_val, y_train, y_val)
    """
    split_idx = int(len(X) * (1 - validation_split))
    
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]
    
    return X_train, X_val, y_train, y_val 
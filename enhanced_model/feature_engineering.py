#!/usr/bin/env python3
"""
Realistic Feature Engineering for Cryptocurrency Prediction

This module provides professional feature engineering capabilities specifically
designed for realistic cryptocurrency price prediction with emphasis on:
- Time-of-day patterns (US/Asian trading hours, weekend effects)
- Market microstructure (bid-ask spreads, volume patterns, price efficiency)
- Regime detection (volatility clustering, trend vs mean-reversion)
- Multi-scale analysis (short-term vs long-term patterns)
- Realistic constraints and temporal consistency
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler, StandardScaler
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple, Dict, Optional
import torch
import warnings
warnings.filterwarnings('ignore')

class RealisticFeatureEngineer:
    """
    Realistic feature engineer for cryptocurrency prediction with market-aware patterns.
    """
    
    def __init__(self):
        self.feature_scaler = RobustScaler()
        self.target_scaler = RobustScaler()
        self.is_fitted = False
        
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Engineer realistic features for cryptocurrency prediction.
        
        Args:
            df: Preprocessed DataFrame with OHLCV data
            
        Returns:
            DataFrame with engineered features
        """
        print(f"🔄 Engineering realistic features for dataset with {len(df)} rows...")
        
        # Ensure timestamp is datetime
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.sort_values('timestamp').reset_index(drop=True)
        
        # Calculate log returns first
        df['log_return'] = np.log(df['close'] / df['close'].shift(1))
        
        # Time-based features (most important for realism)
        df = self.add_time_based_features(df)
        
        # Market microstructure features
        df = self.add_microstructure_features(df)
        
        # Volatility features with time-aware patterns
        df = self.add_volatility_features(df)
        
        # Technical indicators
        df = self.add_technical_indicators(df)
        
        # Regime detection features
        df = self.add_regime_features(df)
        
        # Multi-scale features
        df = self.add_multiscale_features(df)
        
        # Interaction features
        df = self.add_interaction_features(df)
        
        # Aggressive NaN handling
        df = self.handle_nan_values(df)
        
        print(f"✅ Realistic feature engineering complete. Final features: {len(df.columns)}")
        return df
    
    def add_time_based_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add time-based features that capture trading hour patterns.
        This is crucial for realistic predictions.
        """
        if 'timestamp' not in df.columns:
            print("⚠️ No timestamp column found, skipping time features")
            return df
        
        # Extract time components
        df['hour'] = df['timestamp'].dt.hour
        df['minute'] = df['timestamp'].dt.minute
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['day_of_month'] = df['timestamp'].dt.day
        df['month'] = df['timestamp'].dt.month
        
        # Cyclical encoding for time features
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['dow_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['dow_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        
        # Trading hour indicators (most important for realism)
        # US trading hours (9:30-16:00 EST = 14:30-21:00 UTC)
        df['us_trading_hours'] = ((df['hour'] >= 14) & (df['hour'] <= 21)).astype(int)
        
        # Asian trading hours (0:00-8:00 UTC)
        df['asian_trading_hours'] = ((df['hour'] >= 0) & (df['hour'] <= 8)).astype(int)
        
        # European trading hours (8:00-16:00 UTC)
        df['european_trading_hours'] = ((df['hour'] >= 8) & (df['hour'] <= 16)).astype(int)
        
        # Weekend indicator
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        
        # Time of day categories
        df['is_morning'] = ((df['hour'] >= 6) & (df['hour'] <= 12)).astype(int)
        df['is_afternoon'] = ((df['hour'] >= 12) & (df['hour'] <= 18)).astype(int)
        df['is_evening'] = ((df['hour'] >= 18) & (df['hour'] <= 24)).astype(int)
        df['is_night'] = ((df['hour'] >= 0) & (df['hour'] <= 6)).astype(int)
        
        # Market session overlap indicators
        df['us_european_overlap'] = ((df['hour'] >= 14) & (df['hour'] <= 16)).astype(int)
        df['asian_european_overlap'] = ((df['hour'] >= 8) & (df['hour'] <= 10)).astype(int)
        
        print(f"📅 Added {len([col for col in df.columns if 'hour' in col or 'trading' in col or 'weekend' in col])} time-based features")
        return df
    
    def add_microstructure_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add market microstructure features that capture realistic market behavior.
        """
        # Bid-ask spread proxy (high-low ratio)
        df['spread_proxy'] = (df['high'] - df['low']) / df['close']
        
        # Price efficiency (how much price moved relative to range)
        df['price_efficiency'] = abs(df['close'] - df['open']) / (df['high'] - df['low'])
        df['price_efficiency'] = df['price_efficiency'].fillna(0.5)  # Default to middle
        
        # Volume-based features
        if 'volume' in df.columns:
            # Volume relative to recent average
            df['volume_ma_24'] = df['volume'].rolling(window=24).mean()
            df['volume_ratio'] = df['volume'] / df['volume_ma_24']
            df['volume_ratio'] = df['volume_ratio'].fillna(1.0)
            
            # Volume-weighted average price
            df['vwap'] = (df['close'] * df['volume']).rolling(window=24).sum() / df['volume'].rolling(window=24).sum()
            df['vwap'] = df['vwap'].fillna(df['close'])
            
            # Price relative to VWAP
            df['price_vwap_ratio'] = df['close'] / df['vwap']
        else:
            # Fallback for data without volume
            df['volume_ratio'] = 1.0
            df['vwap'] = df['close']
            df['price_vwap_ratio'] = 1.0
        
        # Tick size effect (simplified)
        df['tick_effect'] = (df['close'] % 0.01) / 0.01
        
        # Price momentum indicators
        df['price_momentum_5'] = df['close'] / df['close'].shift(5) - 1
        df['price_momentum_12'] = df['close'] / df['close'].shift(12) - 1
        df['price_momentum_24'] = df['close'] / df['close'].shift(24) - 1
        
        # Gap indicators
        df['gap_up'] = (df['open'] > df['close'].shift(1)).astype(int)
        df['gap_down'] = (df['open'] < df['close'].shift(1)).astype(int)
        
        print(f"🏪 Added {len([col for col in df.columns if 'spread' in col or 'volume' in col or 'vwap' in col or 'momentum' in col or 'gap' in col])} microstructure features")
        return df
    
    def add_volatility_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add volatility features with time-aware patterns.
        """
        # Realized volatility at different windows
        windows = [6, 12, 24, 48, 96]  # 30min, 1hr, 2hr, 4hr, 8hr
        
        for window in windows:
            if len(df) >= window:
                df[f'realized_vol_{window}'] = df['log_return'].rolling(window=window).std()
            else:
                df[f'realized_vol_{window}'] = df['log_return'].std()
        
        # Parkinson volatility (uses high-low range)
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
        df['vol_of_vol'] = df['realized_vol_24'].rolling(window=24).std()
        
        # Time-aware volatility (different for trading hours)
        if 'us_trading_hours' in df.columns:
            df['vol_us_trading'] = df['realized_vol_24'] * df['us_trading_hours']
            df['vol_non_us_trading'] = df['realized_vol_24'] * (1 - df['us_trading_hours'])
        
        # Volatility clustering
        df['vol_clustering'] = df['realized_vol_24'].rolling(window=48).std()
        
        print(f"📊 Added {len([col for col in df.columns if 'vol' in col])} volatility features")
        return df
    
    def add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add technical indicators with adaptive windows.
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
        
        # Average True Range
        df['atr'] = self.calculate_atr(df['high'], df['low'], df['close'])
        
        # Moving averages
        df['sma_12'] = df['close'].rolling(window=12).mean()
        df['sma_24'] = df['close'].rolling(window=24).mean()
        df['ema_12'] = df['close'].ewm(span=12).mean()
        df['ema_24'] = df['close'].ewm(span=24).mean()
        
        # Price relative to moving averages
        df['price_sma_12_ratio'] = df['close'] / df['sma_12']
        df['price_sma_24_ratio'] = df['close'] / df['sma_24']
        
        print(f"📈 Added {len([col for col in df.columns if 'rsi' in col or 'macd' in col or 'bb_' in col or 'stoch' in col or 'williams' in col or 'atr' in col or 'sma' in col or 'ema' in col])} technical indicators")
        return df
    
    def add_regime_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add features for market regime detection.
        """
        # Trend vs mean-reversion indicators
        df['trend_strength'] = abs(df['price_momentum_24']) / df['realized_vol_24']
        
        # Volatility regime
        vol_median = df['realized_vol_24'].median()
        df['high_vol_regime'] = (df['realized_vol_24'] > vol_median * 1.5).astype(int)
        df['low_vol_regime'] = (df['realized_vol_24'] < vol_median * 0.5).astype(int)
        
        # Price regime
        price_ma = df['close'].rolling(window=48).mean()
        df['above_ma'] = (df['close'] > price_ma).astype(int)
        df['below_ma'] = (df['close'] < price_ma).astype(int)
        
        # Momentum regime
        df['strong_momentum'] = (abs(df['price_momentum_12']) > df['realized_vol_12'] * 2).astype(int)
        df['weak_momentum'] = (abs(df['price_momentum_12']) < df['realized_vol_12'] * 0.5).astype(int)
        
        # Range-bound vs trending
        df['range_bound'] = (df['high_vol_regime'] == 0) & (df['strong_momentum'] == 0)
        df['trending'] = (df['strong_momentum'] == 1)
        
        print(f"🔄 Added {len([col for col in df.columns if 'regime' in col or 'trend' in col or 'momentum' in col or 'above_' in col or 'below_' in col or 'range_' in col])} regime features")
        return df
    
    def add_multiscale_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add multi-scale features for different temporal patterns.
        """
        # Short-term patterns (5-15 minutes)
        df['short_term_vol'] = df['log_return'].rolling(window=6).std()
        df['short_term_momentum'] = df['close'] / df['close'].shift(6) - 1
        
        # Medium-term patterns (30-60 minutes)
        df['medium_term_vol'] = df['log_return'].rolling(window=24).std()
        df['medium_term_momentum'] = df['close'] / df['close'].shift(24) - 1
        
        # Long-term patterns (2-4 hours)
        df['long_term_vol'] = df['log_return'].rolling(window=96).std()
        df['long_term_momentum'] = df['close'] / df['close'].shift(96) - 1
        
        # Volatility ratios (capture regime changes)
        df['vol_ratio_short_medium'] = df['short_term_vol'] / df['medium_term_vol']
        df['vol_ratio_medium_long'] = df['medium_term_vol'] / df['long_term_vol']
        
        # Momentum ratios
        df['momentum_ratio_short_medium'] = df['short_term_momentum'] / (df['medium_term_momentum'] + 1e-8)
        df['momentum_ratio_medium_long'] = df['medium_term_momentum'] / (df['long_term_momentum'] + 1e-8)
        
        print(f"📏 Added {len([col for col in df.columns if 'term_' in col or 'ratio_' in col])} multi-scale features")
        return df
    
    def add_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add interaction features between different indicators.
        """
        # Volatility-regime interactions
        if 'realized_vol_24' in df.columns and 'rsi' in df.columns:
            df['vol_rsi_interaction'] = df['realized_vol_24'] * df['rsi']
        
        if 'realized_vol_24' in df.columns and 'macd' in df.columns:
            df['vol_macd_interaction'] = df['realized_vol_24'] * df['macd']
        
        # Time-volatility interactions (most important for realism)
        if 'us_trading_hours' in df.columns and 'realized_vol_24' in df.columns:
            df['hour_vol_interaction'] = df['hour'] * df['realized_vol_24']
            df['us_trading_vol_interaction'] = df['us_trading_hours'] * df['realized_vol_24']
            df['asian_trading_vol_interaction'] = df['asian_trading_hours'] * df['realized_vol_24']
        
        # Weekend-volatility interactions
        if 'is_weekend' in df.columns and 'realized_vol_24' in df.columns:
            df['weekend_vol_interaction'] = df['is_weekend'] * df['realized_vol_24']
        
        # Volume-volatility interactions
        if 'volume_ratio' in df.columns and 'realized_vol_24' in df.columns:
            df['volume_vol_interaction'] = df['volume_ratio'] * df['realized_vol_24']
        
        # Regime-interaction features
        if 'high_vol_regime' in df.columns and 'us_trading_hours' in df.columns:
            df['high_vol_us_trading'] = df['high_vol_regime'] * df['us_trading_hours']
        
        print(f"🔗 Added {len([col for col in df.columns if 'interaction' in col])} interaction features")
        return df
    
    def handle_nan_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handle NaN values aggressively for limited data.
        """
        initial_rows = len(df)
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        
        # Fill NaN values aggressively
        for col in numeric_columns:
            if df[col].isna().sum() > 0:
                # Forward fill, then backward fill, then fill with column mean
                df[col] = df[col].fillna(method='ffill').fillna(method='bfill')
                if df[col].isna().sum() > 0:
                    df[col] = df[col].fillna(df[col].mean())
                if df[col].isna().sum() > 0:
                    df[col] = df[col].fillna(0)  # Final fallback
        
        final_rows = len(df)
        print(f"🧹 NaN handling complete. Rows: {initial_rows} → {final_rows}")
        
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
        """
        # Fit feature scaler
        self.feature_scaler.fit(df[feature_cols])
        
        # Fit target scaler with enhanced transformation
        df_processed = df[target_cols].copy()
        
        # Apply enhanced transformations for targets
        df_processed['target_volatility'] = df_processed['target_volatility']
        df_processed['target_skewness'] = df_processed['target_skewness']
        df_processed['target_kurtosis'] = np.sqrt(df_processed['target_kurtosis'] + 1)
        
        self.target_scaler.fit(df_processed)
        self.is_fitted = True
    
    def transform_data(self, df: pd.DataFrame, feature_cols: List[str], target_cols: List[str]) -> pd.DataFrame:
        """
        Transform data using fitted scalers.
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
        """
        targets_original = self.target_scaler.inverse_transform(targets)
        
        # Apply inverse transformations
        targets_original[:, 2] = targets_original[:, 2]**2 - 1  # Kurtosis inverse
        
        return targets_original
    
    def prepare_sequences(self, df: pd.DataFrame, feature_cols: List[str], 
                         target_cols: List[str], sequence_length: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare sequences for LSTM training with time features.
        """
        print(f"🔄 Creating sequences with {len(df)} rows and sequence length {sequence_length}...")
        
        # Convert to numpy arrays for faster processing
        feature_data = df[feature_cols].values
        target_data = df[target_cols].values
        
        # Calculate number of sequences
        n_sequences = len(df) - sequence_length
        
        # Pre-allocate arrays
        X = np.zeros((n_sequences, sequence_length, len(feature_cols)))
        y = np.zeros((n_sequences, len(target_cols)))
        
        # Use vectorized operations to create sequences
        for i in range(sequence_length):
            X[:, i, :] = feature_data[i:n_sequences + i]
        
        # Set targets
        y = target_data[sequence_length:]
        
        print(f"✅ Sequences created: X shape {X.shape}, y shape {y.shape}")
        return X, y

class RealisticCryptoDataset(Dataset):
    """
    Realistic dataset for cryptocurrency volatility prediction with time features.
    """
    
    def __init__(self, X: np.ndarray, y: np.ndarray, time_features: Optional[np.ndarray] = None):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
        self.time_features = torch.FloatTensor(time_features) if time_features is not None else None
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        if self.time_features is not None:
            return self.X[idx], self.y[idx], self.time_features[idx]
        else:
            return self.X[idx], self.y[idx]

def create_train_val_split(X: np.ndarray, y: np.ndarray, time_features: Optional[np.ndarray] = None, 
                          validation_split: float = 0.2) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Create train-validation split with optional time features.
    """
    split_idx = int(len(X) * (1 - validation_split))
    
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]
    
    if time_features is not None:
        time_train, time_val = time_features[:split_idx], time_features[split_idx:]
        return X_train, X_val, y_train, y_val, time_train, time_val
    
    return X_train, X_val, y_train, y_val 
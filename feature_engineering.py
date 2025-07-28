import numpy as np
import pandas as pd
from typing import List, Tuple
from sklearn.preprocessing import StandardScaler, RobustScaler
import torch
from torch.utils.data import Dataset

class FeatureEngineer:
    """
    Advanced feature engineering for Bitcoin price data.
    """
    
    def __init__(self):
        self.scalers = {}
        self.feature_names = []
        
    def add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add technical analysis indicators with adaptive windows for limited data."""
        
        data_length = len(df)
        
        # Adaptive windows based on data availability
        if data_length < 100:
            # Very limited data - use only small windows
            sma_windows = [6, 12]
            ema_spans = [6, 12]
            bb_windows = [12]
            rsi_windows = [6, 12]
            print(f"⚠️ Limited data ({data_length} points). Using small technical indicator windows.")
        elif data_length < 300:
            # Limited data - use medium windows
            sma_windows = [12, 24, 48]
            ema_spans = [12, 24]
            bb_windows = [24]
            rsi_windows = [12, 24]
            print(f"⚠️ Moderate data ({data_length} points). Using medium technical indicator windows.")
        else:
            # Sufficient data - use full windows
            sma_windows = [12, 24, 48, 96]  # 1h, 2h, 4h, 8h
            ema_spans = [12, 24, 48]
            bb_windows = [24, 48]
            rsi_windows = [14, 24]
        
        # Simple Moving Averages
        for window in sma_windows:
            if window <= data_length // 2:  # Only add if window is reasonable
                df[f'sma_{window}'] = df['close'].rolling(window=window).mean()
                df[f'price_vs_sma_{window}'] = df['close'] / df[f'sma_{window}'] - 1
        
        # Exponential Moving Averages
        for span in ema_spans:
            if span <= data_length // 2:
                df[f'ema_{span}'] = df['close'].ewm(span=span).mean()
                df[f'price_vs_ema_{span}'] = df['close'] / df[f'ema_{span}'] - 1
        
        # Bollinger Bands
        for window in bb_windows:
            if window <= data_length // 2:
                rolling_mean = df['close'].rolling(window=window).mean()
                rolling_std = df['close'].rolling(window=window).std()
                df[f'bb_upper_{window}'] = rolling_mean + (rolling_std * 2)
                df[f'bb_lower_{window}'] = rolling_mean - (rolling_std * 2)
                df[f'bb_position_{window}'] = (df['close'] - df[f'bb_lower_{window}']) / \
                                            (df[f'bb_upper_{window}'] - df[f'bb_lower_{window}'])
        
        # RSI (Relative Strength Index)
        def calculate_rsi(prices, window=14):
            if window >= len(prices):
                return pd.Series([np.nan] * len(prices))
            delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
            rs = gain / loss
            return 100 - (100 / (1 + rs))
        
        for window in rsi_windows:
            if window <= data_length // 2:
                df[f'rsi_{window}'] = calculate_rsi(df['close'], window)
        
        # MACD (only if we have enough data)
        if data_length >= 50:
            ema_12 = df['close'].ewm(span=12).mean()
            ema_26 = df['close'].ewm(span=26).mean()
            df['macd'] = ema_12 - ema_26
            df['macd_signal'] = df['macd'].ewm(span=9).mean()
            df['macd_histogram'] = df['macd'] - df['macd_signal']
        
        return df
    
    def add_volatility_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add various volatility measures with adaptive windows."""
        
        data_length = len(df)
        
        # Adaptive windows based on data availability
        if data_length < 100:
            atr_windows = [6, 12]
            realized_vol_windows = [6, 12]
        elif data_length < 300:
            atr_windows = [12, 24]
            realized_vol_windows = [12, 24, 48]
        else:
            atr_windows = [12, 24, 48]
            realized_vol_windows = [12, 24, 48, 96]
        
        # Garman-Klass volatility estimator
        df['gk_volatility'] = np.sqrt(
            0.5 * (np.log(df['high'] / df['low'])) ** 2 - 
            (2 * np.log(2) - 1) * (np.log(df['close'] / df['open'])) ** 2
        )
        
        # Yang-Zhang volatility estimator
        df['yz_volatility'] = np.sqrt(
            np.log(df['open'] / df['close'].shift(1)) * np.log(df['close'] / df['open']) +
            np.log(df['high'] / df['close']) * np.log(df['high'] / df['open']) +
            np.log(df['low'] / df['close']) * np.log(df['low'] / df['open'])
        )
        
        # Average True Range (ATR)
        for window in atr_windows:
            if window <= data_length // 2:
                df[f'atr_{window}'] = df['true_range'].rolling(window=window).mean()
                df[f'atr_ratio_{window}'] = df['true_range'] / df[f'atr_{window}']
        
        # Realized volatility at different scales
        for window in realized_vol_windows:
            if window <= data_length // 2:
                df[f'realized_vol_{window}'] = df['return'].rolling(window=window).std() * np.sqrt(288)  # Annualized
        
        return df
    
    def add_market_microstructure_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add market microstructure features."""
        
        # Price impact measures
        df['price_impact'] = abs(df['close'] - df['open']) / df['true_range']
        df['gap'] = (df['open'] - df['close'].shift(1)) / df['close'].shift(1)
        
        # Intraday patterns
        df['open_close_ratio'] = df['open'] / df['close']
        df['high_low_ratio'] = df['high'] / df['low']
        df['close_position'] = (df['close'] - df['low']) / (df['high'] - df['low'])
        
        # Price acceleration
        df['price_acceleration'] = df['return'] - df['return'].shift(1)
        
        # Momentum indicators
        for window in [6, 12, 24]:
            df[f'momentum_{window}'] = df['close'] / df['close'].shift(window) - 1
            df[f'return_autocorr_{window}'] = df['return'].rolling(window=window*2).apply(
                lambda x: x.autocorr(lag=window) if len(x) > window else np.nan
            )
        
        return df
    
    def add_regime_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add features to capture different market regimes."""
        
        # Volatility regime indicators
        vol_short = df['realized_vol_24']
        vol_long = df['realized_vol_96']
        df['vol_regime'] = (vol_short > vol_long).astype(int)
        
        # Trend indicators
        for window in [24, 48]:
            df[f'trend_{window}'] = (df['close'] > df[f'sma_{window}']).astype(int)
        
        # Market stress indicators
        df['stress_indicator'] = (
            (df['realized_vol_24'] > df['realized_vol_24'].rolling(96).quantile(0.8)) &
            (abs(df['return']) > df['return'].rolling(96).std() * 2)
        ).astype(int)
        
        return df
    
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Complete feature engineering pipeline with enhanced robustness for limited data."""
        
        initial_rows = len(df)
        print(f"🔄 Starting feature engineering with {initial_rows} data points...")
        
        try:
            print("Adding technical indicators...")
            df = self.add_technical_indicators(df)
            
            print("Adding volatility features...")
            df = self.add_volatility_features(df)
            
            print("Adding microstructure features...")
            df = self.add_market_microstructure_features(df)
            
            print("Adding regime features...")
            df = self.add_regime_features(df)
            
            # Handle NaN values more aggressively for limited data
            if initial_rows < 500:
                print(f"⚠️ Limited data detected. Using aggressive NaN handling in feature engineering...")
                
                # Fill NaN values in technical indicators with forward/backward fill
                technical_cols = [col for col in df.columns if any(indicator in col for indicator in 
                    ['sma_', 'ema_', 'bb_', 'rsi_', 'macd_', 'atr_', 'realized_vol_'])]
                
                for col in technical_cols:
                    if col in df.columns:
                        df[col] = df[col].fillna(method='ffill').fillna(method='bfill').fillna(0)
                
                # Fill any remaining NaN values with 0
                df = df.fillna(0)
                
                print(f"📊 After feature engineering NaN handling: {len(df)} data points")
            else:
                # For larger datasets, remove rows with NaN values
                initial_feature_rows = len(df)
                df = df.dropna().reset_index(drop=True)
                final_feature_rows = len(df)
                
                if final_feature_rows < initial_feature_rows:
                    print(f"📊 Removed {initial_feature_rows - final_feature_rows} rows with NaN values during feature engineering")
            
            print(f"✅ Feature engineering completed. Final shape: {df.shape}")
            return df
            
        except Exception as e:
            print(f"❌ Error during feature engineering: {str(e)}")
            # Return original dataframe if feature engineering fails
            return df
    
    def prepare_sequences(self, df: pd.DataFrame, feature_cols: List[str], 
                         target_cols: List[str], sequence_length: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare sequences for LSTM training.
        """
        
        # Select and scale features
        features = df[feature_cols].values
        targets = df[target_cols].values
        
        # Create sequences
        X, y = [], []
        
        for i in range(sequence_length, len(features)):
            X.append(features[i-sequence_length:i])
            y.append(targets[i])
        
        return np.array(X), np.array(y)
    
    def fit_scalers(self, df: pd.DataFrame, feature_cols: List[str], target_cols: List[str]):
        """Fit scalers on training data."""
        
        # Separate scalers for features and targets
        self.scalers['features'] = RobustScaler()
        # Use RobustScaler for targets to handle outliers better, especially for kurtosis
        self.scalers['targets'] = RobustScaler()
        
        # Preprocess targets before scaling
        df_processed = df.copy()
        
        # Apply more reasonable transformation to kurtosis
        # Use a softer transformation that doesn't amplify values as much
        if 'target_kurtosis' in target_cols:
            # Use a softer transformation: sqrt(kurtosis + 1) instead of log
            # This keeps values in a more reasonable range
            df_processed['target_kurtosis'] = np.sqrt(df_processed['target_kurtosis'] + 1)
        
        # Fit scalers
        self.scalers['features'].fit(df[feature_cols])
        self.scalers['targets'].fit(df_processed[target_cols])
        
        self.feature_names = feature_cols
        
    def transform_data(self, df: pd.DataFrame, feature_cols: List[str], target_cols: List[str]) -> pd.DataFrame:
        """Transform data using fitted scalers."""
        
        if 'features' not in self.scalers:
            raise ValueError("Scalers not fitted. Call fit_scalers() first.")
        
        df_scaled = df.copy()
        
        # Apply softer transformation to kurtosis before scaling
        if 'target_kurtosis' in target_cols:
            df_scaled['target_kurtosis'] = np.sqrt(df_scaled['target_kurtosis'] + 1)
        
        # Scale features and targets
        df_scaled[feature_cols] = self.scalers['features'].transform(df[feature_cols])
        df_scaled[target_cols] = self.scalers['targets'].transform(df_scaled[target_cols])
        
        return df_scaled
    
    def inverse_transform_targets(self, targets: np.ndarray) -> np.ndarray:
        """Inverse transform predictions back to original scale."""
        # First inverse transform the scaled values
        targets_original = self.scalers['targets'].inverse_transform(targets)
        
        # Reverse the sqrt transformation for kurtosis (column 2)
        if targets_original.shape[1] > 2:  # Ensure we have kurtosis column
            targets_original[:, 2] = targets_original[:, 2]**2 - 1
        
        return targets_original


class CryptoDataset(Dataset):
    """
    PyTorch Dataset for cryptocurrency time series data.
    """
    
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def create_train_val_split(X: np.ndarray, y: np.ndarray, 
                          val_split: float = 0.2) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Create time-aware train/validation split.
    """
    split_idx = int(len(X) * (1 - val_split))
    
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]
    
    return X_train, X_val, y_train, y_val 
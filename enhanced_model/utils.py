#!/usr/bin/env python3
"""
Utility functions for enhanced model prediction and data validation.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any
from datetime import datetime

def format_prediction_output(prediction: Dict[str, Any]) -> str:
    """
    Format prediction output for display.
    
    Args:
        prediction: Prediction dictionary
        
    Returns:
        Formatted string
    """
    try:
        output = f"""
📊 Enhanced Prediction Results:
   Volatility: {prediction.get('predicted_volatility', 'N/A'):.6f}
   Skewness: {prediction.get('predicted_skewness', 'N/A'):.6f}
   Kurtosis: {prediction.get('predicted_kurtosis', 'N/A'):.6f}
   Risk Level: {prediction.get('risk_level', 'N/A')}
   Uncertainty: {prediction.get('uncertainty_volatility', 'N/A'):.6f}, {prediction.get('uncertainty_skewness', 'N/A'):.6f}, {prediction.get('uncertainty_kurtosis', 'N/A'):.6f}
   Current Price: ${prediction.get('current_price', 'N/A'):.2f}
   Prediction Time: {prediction.get('prediction_time', 'N/A')}
"""
        return output
    except Exception as e:
        return f"Error formatting prediction: {str(e)}"

def validate_crypto_data(df: pd.DataFrame) -> bool:
    """
    Validate cryptocurrency OHLC data.
    
    Args:
        df: DataFrame with OHLC data
        
    Returns:
        True if data is valid, False otherwise
    """
    try:
        # Check required columns
        required_columns = ['timestamp', 'open', 'high', 'low', 'close']
        if not all(col in df.columns for col in required_columns):
            print("❌ Missing required columns")
            return False
        
        # Check data types
        if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
            print("❌ Timestamp column is not datetime")
            return False
        
        # Check for negative prices
        price_columns = ['open', 'high', 'low', 'close']
        for col in price_columns:
            if (df[col] <= 0).any():
                print(f"❌ Negative or zero prices found in {col}")
                return False
        
        # Check OHLC consistency
        if not ((df['low'] <= df['open']) & (df['low'] <= df['close']) & 
                (df['high'] >= df['open']) & (df['high'] >= df['close'])).all():
            print("❌ OHLC data inconsistency detected")
            return False
        
        # Check for sufficient data
        if len(df) < 100:
            print(f"❌ Insufficient data: {len(df)} < 100")
            return False
        
        # Check for reasonable price ranges
        price_range = (df['high'].max() - df['low'].min()) / df['close'].mean()
        if price_range > 10:  # More than 1000% range is suspicious
            print(f"❌ Unreasonable price range: {price_range:.2f}")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Data validation error: {str(e)}")
        return False

def calculate_rolling_statistics(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """
    Calculate rolling statistics for enhanced features.
    
    Args:
        df: DataFrame with price data
        window: Rolling window size
        
    Returns:
        DataFrame with rolling statistics
    """
    try:
        df = df.copy()
        
        # Calculate returns
        df['return'] = df['close'].pct_change()
        
        # Rolling volatility
        df['rolling_volatility'] = df['return'].rolling(window=window).std()
        
        # Rolling skewness
        df['rolling_skewness'] = df['return'].rolling(window=window).skew()
        
        # Rolling kurtosis
        df['rolling_kurtosis'] = df['return'].rolling(window=window).kurt()
        
        # Rolling mean
        df['rolling_mean'] = df['return'].rolling(window=window).mean()
        
        return df
        
    except Exception as e:
        print(f"❌ Error calculating rolling statistics: {str(e)}")
        return df

def normalize_features(df: pd.DataFrame, feature_columns: List[str]) -> pd.DataFrame:
    """
    Normalize feature columns using robust scaling.
    
    Args:
        df: DataFrame with features
        feature_columns: List of feature column names
        
    Returns:
        DataFrame with normalized features
    """
    try:
        df = df.copy()
        
        for col in feature_columns:
            if col in df.columns:
                # Robust normalization (subtract median, divide by IQR)
                median_val = df[col].median()
                q75, q25 = df[col].quantile(0.75), df[col].quantile(0.25)
                iqr = q75 - q25
                
                if iqr > 0:
                    df[col] = (df[col] - median_val) / iqr
                else:
                    # Fallback to standard normalization
                    std_val = df[col].std()
                    if std_val > 0:
                        df[col] = (df[col] - df[col].mean()) / std_val
        
        return df
        
    except Exception as e:
        print(f"❌ Error normalizing features: {str(e)}")
        return df

def create_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create time-based features for enhanced prediction.
    
    Args:
        df: DataFrame with timestamp column
        
    Returns:
        DataFrame with time features
    """
    try:
        df = df.copy()
        
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
        
        # Trading hours indicators
        df['us_trading_hours'] = ((df['hour'] >= 9) & (df['hour'] <= 16)).astype(int)
        df['asian_trading_hours'] = ((df['hour'] >= 0) & (df['hour'] <= 8)).astype(int)
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        
        return df
        
    except Exception as e:
        print(f"❌ Error creating time features: {str(e)}")
        return df

def assess_market_regime(volatility: float, skewness: float, kurtosis: float) -> str:
    """
    Assess market regime based on statistical moments.
    
    Args:
        volatility: Volatility measure
        skewness: Skewness measure
        kurtosis: Kurtosis measure
        
    Returns:
        Market regime classification
    """
    try:
        # Volatility-based regime
        if volatility > 0.05:
            vol_regime = "HIGH_VOL"
        elif volatility > 0.02:
            vol_regime = "MED_VOL"
        else:
            vol_regime = "LOW_VOL"
        
        # Skewness-based regime
        if abs(skewness) > 1.0:
            skew_regime = "HEAVY_TAIL"
        elif abs(skewness) > 0.5:
            skew_regime = "MODERATE_TAIL"
        else:
            skew_regime = "NORMAL_TAIL"
        
        # Kurtosis-based regime
        if kurtosis > 5:
            kurt_regime = "FAT_TAIL"
        elif kurtosis > 2:
            kurt_regime = "MODERATE_TAIL"
        else:
            kurt_regime = "NORMAL_TAIL"
        
        # Combine regimes
        if vol_regime == "HIGH_VOL" and (skew_regime == "HEAVY_TAIL" or kurt_regime == "FAT_TAIL"):
            return "CRISIS"
        elif vol_regime == "HIGH_VOL":
            return "VOLATILE"
        elif skew_regime == "HEAVY_TAIL" or kurt_regime == "FAT_TAIL":
            return "TAIL_RISK"
        else:
            return "NORMAL"
            
    except Exception as e:
        print(f"❌ Error assessing market regime: {str(e)}")
        return "UNKNOWN"

def calculate_prediction_confidence(uncertainty: List[float]) -> float:
    """
    Calculate prediction confidence based on uncertainty measures.
    
    Args:
        uncertainty: List of uncertainty values [vol, skew, kurt]
        
    Returns:
        Confidence score (0-1)
    """
    try:
        # Normalize uncertainties (lower is better)
        max_uncertainty = max(uncertainty)
        if max_uncertainty > 0:
            normalized_uncertainties = [u / max_uncertainty for u in uncertainty]
            # Convert to confidence (1 - average uncertainty)
            confidence = 1.0 - np.mean(normalized_uncertainties)
            return max(0.0, min(1.0, confidence))
        else:
            return 1.0
            
    except Exception as e:
        print(f"❌ Error calculating confidence: {str(e)}")
        return 0.5
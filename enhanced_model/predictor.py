#!/usr/bin/env python3
"""
Enhanced Predictor for Monte Carlo Simulation

This module provides prediction capabilities for the enhanced model architecture
specifically designed for Monte Carlo simulation with better statistical moment prediction.
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import os
import pickle
import json
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

from config import EnhancedConfig
from enhanced_model import EnhancedVolatilityModel, create_enhanced_model
from feature_engineering import EnhancedFeatureEngineer
from data_processor import EnhancedCryptoDataProcessor

class EnhancedRealTimeVolatilityPredictor:
    """
    Enhanced real-time volatility predictor for Monte Carlo simulation.
    """
    
    def __init__(self, config: EnhancedConfig, crypto_symbol: str = 'BTC'):
        self.config = config
        self.crypto_symbol = crypto_symbol
        self.crypto_config = EnhancedConfig.SUPPORTED_CRYPTOS[crypto_symbol]
        self.device = config.DEVICE
        self.model = None
        self.feature_engineer = None
        self.feature_cols = None
        self.target_cols = None
        self.is_loaded = False
        
        print(f"Enhanced predictor initialized for {self.crypto_config['name']} ({crypto_symbol})")
    
    def load_latest_model(self) -> bool:
        """
        Load the latest enhanced model.
        
        Returns:
            True if model loaded successfully, False otherwise
        """
        try:
            # Look for enhanced model (with 'enhanced_' prefix)
            model_path = os.path.join(self.config.MODEL_SAVE_PATH, f"{self.crypto_symbol}_enhanced_model.pth")
            feature_engineer_path = os.path.join(self.config.MODEL_SAVE_PATH, f"{self.crypto_symbol}_enhanced_feature_engineer.pkl")
            
            if not os.path.exists(model_path):
                print(f"Enhanced model not found: {model_path}")
                return False
            
            if not os.path.exists(feature_engineer_path):
                print(f"Enhanced feature engineer not found: {feature_engineer_path}")
                return False
            
            # Load checkpoint first to get feature count
            checkpoint = torch.load(model_path, map_location=self.device)
            
            # Get feature and target columns from checkpoint
            self.feature_cols = checkpoint['feature_cols']
            self.target_cols = checkpoint['target_cols']
            
            # Create model with correct input size based on actual features
            actual_input_size = len(self.feature_cols)
            print(f"Creating model with input size: {actual_input_size} (from {len(self.feature_cols)} features)")
            
            # Create model with dynamic input size
            self.model = EnhancedVolatilityModel(
                input_size=actual_input_size,
                hidden_size=self.config.HIDDEN_SIZE,
                num_layers=self.config.NUM_LAYERS,
                num_heads=8,
                dropout=self.config.DROPOUT,
                num_quantiles=5
            )
            
            # Load model state
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.to(self.device)
            self.model.eval()
            
            # Load feature engineer
            with open(feature_engineer_path, 'rb') as f:
                self.feature_engineer = pickle.load(f)
            
            self.is_loaded = True
            print(f"✅ Enhanced model loaded successfully")
            print(f"   Model epoch: {checkpoint['epoch']}")
            print(f"   Features: {len(self.feature_cols)}")
            print(f"   Targets: {len(self.target_cols)}")
            print(f"   Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error loading enhanced model: {str(e)}")
            return False
    
    def load_model(self, model_config: Dict) -> bool:
        """
        Load model with specific configuration.
        
        Args:
            model_config: Model configuration dictionary
            
        Returns:
            True if model loaded successfully, False otherwise
        """
        try:
            # Create model with config
            self.model = create_enhanced_model(self.config)
            
            # Load state dict if provided
            if 'state_dict' in model_config:
                self.model.load_state_dict(model_config['state_dict'])
            
            self.model.to(self.device)
            self.model.eval()
            
            # Set feature columns
            self.feature_cols = model_config.get('feature_cols', [])
            self.target_cols = model_config.get('target_cols', ['target_volatility', 'target_skewness', 'target_kurtosis'])
            
            self.is_loaded = True
            return True
            
        except Exception as e:
            print(f"❌ Error loading model with config: {str(e)}")
            return False
    
    def predict_next_period(self, price_data: pd.DataFrame, current_price: float) -> Dict:
        """
        Predict volatility, skewness, and kurtosis for the next period.
        
        Args:
            price_data: Historical OHLC data
            current_price: Current price
            
        Returns:
            Dictionary with predictions and uncertainty
        """
        if not self.is_loaded:
            if not self.load_latest_model():
                raise ValueError("Model not loaded. Please load a model first.")
        
        try:
            # Preprocess data using the provided DataFrame directly
            # Create a temporary processor that doesn't try to load from file
            processor = EnhancedCryptoDataProcessor("", self.crypto_symbol)
            processor.df = price_data.copy()  # Use provided data directly
            
            # Skip the load_data step since we already have the data
            # Preprocess the data directly
            df = processor.calculate_returns(processor.df)
            df = processor.add_time_features(df)
            df = processor.calculate_rolling_statistics(df, self.config.RETURN_WINDOWS)
            df = processor.calculate_target_statistics(df, self.config.PREDICTION_HORIZON)
            
            # Add features
            df = self.feature_engineer.engineer_features(df)
            
            # Remove NaN values
            df = df.dropna().reset_index(drop=True)
            
            # Debug: Check feature mismatch
            available_features = set(df.columns)
            expected_features = set(self.feature_cols)
            missing_features = expected_features - available_features
            extra_features = available_features - expected_features
            
            if missing_features:
                print(f"Warning: Missing features: {missing_features}")
                # Try to create missing features with default values
                for feature in missing_features:
                    if feature.startswith('realized_vol_'):
                        df[feature] = df['log_return'].std()  # Use overall std
                    else:
                        df[feature] = 0.0  # Default value for other features
            
            if extra_features:
                print(f"Warning: Extra features: {extra_features}")
            
            # Handle limited data by adjusting sequence length
            available_data = len(df)
            required_sequence_length = self.config.SEQUENCE_LENGTH
            
            if available_data < required_sequence_length:
                # Use adaptive sequence length for limited data
                adaptive_sequence_length = max(24, min(available_data - 1, required_sequence_length))
                print(f"Limited data ({available_data} points). Using adaptive sequence length: {adaptive_sequence_length}")
                required_sequence_length = adaptive_sequence_length
            
            if len(df) < required_sequence_length:
                raise ValueError(f"Insufficient data: {len(df)} < {required_sequence_length}")
            
            # Ensure we have all required features
            df_features = df[self.feature_cols].copy()
            
            # Get the last sequence
            last_sequence = df_features.iloc[-required_sequence_length:].values
            
            # Transform features
            last_sequence_scaled = self.feature_engineer.feature_scaler.transform(last_sequence)
            
            # Prepare input tensor
            X = torch.FloatTensor(last_sequence_scaled).unsqueeze(0).to(self.device)
            
            # Make prediction
            with torch.no_grad():
                predictions = self.model(X)
            
            # Extract point predictions
            point_predictions = predictions['point_predictions'].cpu().numpy()[0]
            uncertainty = predictions['uncertainty'].cpu().numpy()[0]
            
            # Inverse transform targets
            targets_original = self.feature_engineer.inverse_transform_targets(point_predictions.reshape(1, -1))[0]
            
            # Apply bounds
            volatility = np.clip(targets_original[0], 0.001, 0.1)
            skewness = np.clip(targets_original[1], -2.0, 2.0)
            kurtosis = np.clip(targets_original[2], -1.0, 10.0)
            
            # Assess risk level
            risk_level = self._assess_risk_level(volatility, skewness, kurtosis)
            
            return {
                'predicted_volatility': float(volatility),
                'predicted_skewness': float(skewness),
                'predicted_kurtosis': float(kurtosis),
                'uncertainty_volatility': float(uncertainty[0]),
                'uncertainty_skewness': float(uncertainty[1]),
                'uncertainty_kurtosis': float(uncertainty[2]),
                'current_price': current_price,
                'prediction_time': datetime.now().isoformat(),
                'risk_level': risk_level,
                'model_type': 'enhanced',
                'confidence': 0.8 if available_data >= self.config.SEQUENCE_LENGTH else 0.6
            }
            
        except Exception as e:
            raise Exception(f"Prediction failed: {str(e)}")
    
    def predict_for_timestamp(self, price_data: pd.DataFrame, target_timestamp: datetime, 
                            current_price: float) -> Dict:
        """
        Predict for a specific timestamp.
        
        Args:
            price_data: Historical OHLC data
            target_timestamp: Target timestamp for prediction
            current_price: Current price
            
        Returns:
            Dictionary with predictions
        """
        # Use the same logic as predict_next_period for now
        # In a more sophisticated implementation, you could adjust for time-specific factors
        return self.predict_next_period(price_data, current_price)
    
    def _assess_risk_level(self, volatility: float, skewness: float, kurtosis: float) -> str:
        """
        Assess risk level based on predicted moments.
        
        Args:
            volatility: Predicted volatility
            skewness: Predicted skewness
            kurtosis: Predicted kurtosis
            
        Returns:
            Risk level string
        """
        risk_score = 0
        
        # Volatility risk
        if volatility > 0.05:
            risk_score += 3
        elif volatility > 0.03:
            risk_score += 2
        elif volatility > 0.02:
            risk_score += 1
        
        # Skewness risk (negative skewness indicates higher downside risk)
        if skewness < -1.0:
            risk_score += 2
        elif skewness < -0.5:
            risk_score += 1
        
        # Kurtosis risk (high kurtosis indicates fat tails)
        if kurtosis > 8:
            risk_score += 3
        elif kurtosis > 5:
            risk_score += 2
        elif kurtosis > 3:
            risk_score += 1
        
        # Determine risk level
        if risk_score >= 6:
            return 'HIGH'
        elif risk_score >= 3:
            return 'MEDIUM'
        else:
            return 'LOW'
    
    def get_model_info(self) -> Dict:
        """
        Get information about the loaded model.
        
        Returns:
            Dictionary with model information
        """
        if not self.is_loaded:
            return {'error': 'Model not loaded'}
        
        return {
            'crypto_symbol': self.crypto_symbol,
            'model_type': 'enhanced',
            'device': str(self.device),
            'feature_count': len(self.feature_cols) if self.feature_cols else 0,
            'target_count': len(self.target_cols) if self.target_cols else 0,
            'sequence_length': self.config.SEQUENCE_LENGTH,
            'prediction_horizon': self.config.PREDICTION_HORIZON
        }

def main():
    """Example usage of the enhanced predictor with live data."""
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python predictor.py <crypto_symbol>")
        print("Supported cryptos: BTC, ETH, XAU, SOL")
        print("\nThis demonstrates how to use the predictor with live data.")
        print("In production, you would pass live OHLC data from your data source.")
        sys.exit(1)
    
    crypto_symbol = sys.argv[1].upper()
    
    if crypto_symbol not in EnhancedConfig.SUPPORTED_CRYPTOS:
        print(f"Unsupported crypto: {crypto_symbol}")
        print(f"Supported: {list(EnhancedConfig.SUPPORTED_CRYPTOS.keys())}")
        sys.exit(1)
    
    # Initialize config and predictor
    config = EnhancedConfig()
    predictor = EnhancedRealTimeVolatilityPredictor(config, crypto_symbol)
    
    # Load model
    if not predictor.load_latest_model():
        print("❌ Failed to load model")
        sys.exit(1)
    
    # Get model info
    model_info = predictor.get_model_info()
    print(f"📊 Model Info: {model_info}")
    
    # Simulate live data (in production, this would come from your data source)
    print("\n🎯 Simulating live data for prediction...")
    print("Note: In production, you would pass real live OHLC data here.")
    
    try:
        # Simulate recent live OHLC data (this is what you'd get from Pyth Network or other APIs)
        # In production, you'd replace this with actual live data
        current_time = pd.Timestamp.now()
        live_data = []
        
        # Generate simulated live data for the last 200 periods (5-minute intervals)
        base_price = 45000 if crypto_symbol == 'BTC' else 3000 if crypto_symbol == 'ETH' else 2000 if crypto_symbol == 'XAU' else 100
        
        for i in range(200):
            timestamp = current_time - pd.Timedelta(minutes=5 * (200 - i))
            price_change = np.random.normal(0, 0.001) * base_price
            open_price = base_price + price_change
            high_price = open_price + abs(np.random.normal(0, 0.002)) * base_price
            low_price = open_price - abs(np.random.normal(0, 0.002)) * base_price
            close_price = open_price + np.random.normal(0, 0.001) * base_price
            volume = 1000 + np.random.normal(0, 200)
            
            live_data.append({
                'timestamp': timestamp,
                'open': open_price,
                'high': high_price,
                'low': low_price,
                'close': close_price,
                'volume': volume
            })
        
        # Convert to DataFrame (this is what you'd pass to the predictor)
        live_df = pd.DataFrame(live_data)
        current_price = live_df['close'].iloc[-1]
        
        print(f"📊 Simulated {len(live_df)} periods of live data")
        print(f"💰 Current price: ${current_price:.2f}")
        print(f"🕐 Latest timestamp: {live_df['timestamp'].iloc[-1]}")
        
        # Make prediction on live data
        prediction = predictor.predict_next_period(live_df, current_price)
        
        print(f"\n✅ Prediction successful!")
        print(f"📈 Volatility: {prediction['predicted_volatility']:.6f}")
        print(f"📊 Skewness: {prediction['predicted_skewness']:.6f}")
        print(f"📉 Kurtosis: {prediction['predicted_kurtosis']:.6f}")
        print(f"⚠️  Risk Level: {prediction['risk_level']}")
        print(f"❓ Uncertainty: {prediction['uncertainty_volatility']:.6f}, {prediction['uncertainty_skewness']:.6f}, {prediction['uncertainty_kurtosis']:.6f}")
        
        print(f"\n💡 Usage in production:")
        print(f"   # Get live data from your source (Pyth Network, etc.)")
        print(f"   live_data = get_live_ohlc_data('{crypto_symbol}')")
        print(f"   current_price = live_data['close'].iloc[-1]")
        print(f"   prediction = predictor.predict_next_period(live_data, current_price)")
        
    except Exception as e:
        print(f"❌ Prediction failed: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main() 
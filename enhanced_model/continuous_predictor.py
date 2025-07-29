#!/usr/bin/env python3
"""
Realistic Continuous Predictor for Cryptocurrency Price Prediction

This module provides continuous prediction capabilities for the realistic enhanced model
with time-aware features and market-aware constraints.
"""

import torch
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import os
import json
import time
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

# Import realistic components
from enhanced_model import create_realistic_enhanced_model, RealisticLoss
from feature_engineering import RealisticFeatureEngineer
from data_processor import EnhancedCryptoDataProcessor
from utils import format_prediction_output, validate_crypto_data
from kurtosis_smoothing import PredictionPostProcessor, analyze_prediction_quality

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

class RealisticRealTimeVolatilityPredictor:
    """
    Realistic real-time volatility predictor with time-aware features.
    """
    
    def __init__(self, config, crypto_symbol: str = 'BTC'):
        self.config = config
        self.crypto_symbol = crypto_symbol
        self.device = config.DEVICE
        
        # Initialize components
        self.feature_engineer = RealisticFeatureEngineer()
        self.model = None
        self.is_model_loaded = False
        
        # Initialize kurtosis post-processor
        self.post_processor = PredictionPostProcessor()
        
        print(f"🚀 Initialized Realistic Real-Time Predictor for {crypto_symbol}")
    
    def load_model(self):
        """
        Load the trained realistic model.
        """
        try:
            model_path = os.path.join(self.config.MODEL_SAVE_PATH, f'{self.crypto_symbol}_realistic_model.pth')
            
            if not os.path.exists(model_path):
                print(f"❌ Model file not found: {model_path}")
                print("Please train the model first using the trainer.")
                return False
            
            # Load checkpoint
            checkpoint = torch.load(model_path, map_location=self.device)
            
            # Get feature columns
            feature_cols = checkpoint.get('feature_cols', [])
            if not feature_cols:
                print("❌ No feature columns found in model checkpoint")
                return False
            
            # Create model with correct input size
            input_size = len(feature_cols)
            self.model = create_realistic_enhanced_model(self.config)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.to(self.device)
            self.model.eval()
            
            # Load feature engineer if available
            if 'feature_engineer' in checkpoint:
                self.feature_engineer = checkpoint['feature_engineer']
            
            self.feature_cols = feature_cols
            self.is_model_loaded = True
            
            print(f"✅ Realistic model loaded successfully")
            print(f"   Model path: {model_path}")
            print(f"   Input size: {input_size}")
            print(f"   Feature columns: {len(feature_cols)}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error loading model: {str(e)}")
            return False
    
    def preprocess_recent_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess recent data with realistic feature engineering.
        """
        print(f"🔄 Preprocessing recent data for {self.crypto_symbol}...")
        
        # Ensure timestamp column exists
        if 'timestamp' not in data.columns:
            print("⚠️ No timestamp column found, creating one...")
            data['timestamp'] = pd.date_range(start='2022-01-01', periods=len(data), freq='5T')
        
        # Engineer realistic features
        data = self.feature_engineer.engineer_features(data)
        
        # Calculate targets (for validation)
        data = self.calculate_targets(data)
        
        print(f"✅ Data preprocessing complete. Shape: {data.shape}")
        return data
    
    def calculate_targets(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate realistic targets for validation.
        """
        # Calculate rolling statistics for targets
        window = 24  # 2 hours for target calculation
        
        # Volatility target (realized volatility)
        df['target_volatility'] = df['log_return'].rolling(window=window).std()
        
        # Skewness target
        df['target_skewness'] = df['log_return'].rolling(window=window).skew()
        
        # Kurtosis target (excess kurtosis)
        df['target_kurtosis'] = df['log_return'].rolling(window=window).kurt()
        
        # Fill NaN values
        df['target_volatility'] = df['target_volatility'].fillna(method='ffill').fillna(0.01)
        df['target_skewness'] = df['target_skewness'].fillna(method='ffill').fillna(0.0)
        df['target_kurtosis'] = df['target_kurtosis'].fillna(method='ffill').fillna(0.0)
        
        # Apply realistic constraints to targets
        df['target_volatility'] = np.clip(df['target_volatility'], 0.001, 0.5)
        df['target_skewness'] = np.clip(df['target_skewness'], -0.8, 0.8)
        df['target_kurtosis'] = np.clip(df['target_kurtosis'], 0.1, 10.0)
        
        return df
    
    def prepare_prediction_input(self, df: pd.DataFrame) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Prepare input for prediction with time-aware features.
        """
        # Define feature columns (all numeric columns except targets and timestamp)
        exclude_cols = ['timestamp', 'target_volatility', 'target_skewness', 'target_kurtosis']
        feature_cols = [col for col in df.columns if col not in exclude_cols and df[col].dtype in ['float64', 'int64']]
        
        # Ensure we have the expected features
        if not all(col in df.columns for col in self.feature_cols):
            missing_cols = [col for col in self.feature_cols if col not in df.columns]
            print(f"⚠️ Missing features: {missing_cols}")
            # Add missing columns with default values
            for col in missing_cols:
                df[col] = 0.0
        
        # Use only the expected features in the correct order
        feature_data = df[self.feature_cols].values
        
        # Transform features
        feature_data = self.feature_engineer.feature_scaler.transform(feature_data)
        
        # Create sequence
        if len(feature_data) >= self.config.SEQUENCE_LENGTH:
            sequence = feature_data[-self.config.SEQUENCE_LENGTH:]
        else:
            # Pad with zeros if not enough data
            padding = np.zeros((self.config.SEQUENCE_LENGTH - len(feature_data), len(self.feature_cols)))
            sequence = np.vstack([padding, feature_data])
        
        # Convert to tensor
        X = torch.FloatTensor(sequence).unsqueeze(0).to(self.device)  # Add batch dimension
        
        # Extract time features
        time_features = self.extract_time_features(df)
        
        return X, time_features
    
    def extract_time_features(self, df: pd.DataFrame) -> Optional[torch.Tensor]:
        """
        Extract time features for prediction.
        """
        if 'hour_sin' not in df.columns or 'hour_cos' not in df.columns:
            print("⚠️ No time features found, skipping time-aware prediction")
            return None
        
        # Extract time features for the last sequence_length timesteps
        if len(df) >= self.config.SEQUENCE_LENGTH:
            time_data = df[['hour_sin', 'hour_cos']].iloc[-self.config.SEQUENCE_LENGTH:].values
        else:
            # Pad with zeros if not enough data
            padding = np.zeros((self.config.SEQUENCE_LENGTH - len(df), 2))
            time_data = np.vstack([padding, df[['hour_sin', 'hour_cos']].values])
        
        # Convert to tensor
        time_features = torch.FloatTensor(time_data).unsqueeze(0).to(self.device)  # Add batch dimension
        
        return time_features
    
    def predict_single_step(self, df: pd.DataFrame) -> Dict:
        """
        Make a single prediction with time-aware features.
        """
        if not self.is_model_loaded:
            print("❌ Model not loaded. Please load the model first.")
            return None
        
        try:
            # Prepare input
            X, time_features = self.prepare_prediction_input(df)
            
            # Make prediction
            with torch.no_grad():
                predictions = self.model(X, time_features)
            
            # Extract predictions
            point_predictions = predictions['point_predictions'].cpu().numpy()[0]
            uncertainty = predictions['uncertainty'].cpu().numpy()[0]
            
            # Inverse transform predictions
            predictions_original = self.feature_engineer.inverse_transform_targets(point_predictions.reshape(1, -1))[0]
            
            # Extract individual predictions
            volatility = predictions_original[0]
            skewness = predictions_original[1]
            kurtosis = predictions_original[2]
            
            # Apply realistic constraints
            volatility = np.clip(volatility, self.config.MIN_VOLATILITY, self.config.MAX_VOLATILITY)
            skewness = np.clip(skewness, -self.config.MAX_SKEWNESS, self.config.MAX_SKEWNESS)
            kurtosis = np.clip(kurtosis, self.config.MIN_KURTOSIS, self.config.MAX_KURTOSIS)
            
            # Calculate annualized volatility
            volatility_annualized = volatility * np.sqrt(288 * 365)  # 288 5-min intervals per day
            
            result = {
                'predicted_volatility': float(volatility),
                'predicted_skewness': float(skewness),
                'predicted_kurtosis': float(kurtosis),
                'volatility_annualized': float(volatility_annualized),
                'uncertainty': uncertainty.tolist(),
                'timestamp': datetime.now().isoformat(),
                'crypto_symbol': self.crypto_symbol
            }
            
            return result
            
        except Exception as e:
            print(f"❌ Error making prediction: {str(e)}")
            return None
    
    def generate_288_predictions(self, df: pd.DataFrame, current_price: float) -> Dict:
        """
        Generate 288 predictions (24 hours) with time-aware features.
        """
        if not self.is_model_loaded:
            print("❌ Model not loaded. Please load the model first.")
            return None
        
        try:
            print(f"🔄 Generating 288 predictions for {self.crypto_symbol}...")
            
            predictions = []
            
            # Generate predictions for each of the 288 5-minute intervals
            for i in range(288):
                # Make prediction
                prediction = self.predict_single_step(df)
                
                if prediction is None:
                    print(f"❌ Failed to generate prediction {i+1}")
                    continue
                
                # Add prediction index and current price
                prediction['prediction_index'] = i
                prediction['current_price'] = current_price
                
                predictions.append(prediction)
                
                # Update data for next prediction (simulate time progression)
                # This is a simplified approach - in practice, you might want to use actual time progression
                if i < 287:  # Don't update on the last iteration
                    # Add a small time increment to the last timestamp
                    last_timestamp = pd.to_datetime(df['timestamp'].iloc[-1])
                    new_timestamp = last_timestamp + timedelta(minutes=5)
                    
                    # Create a new row with updated timestamp
                    new_row = df.iloc[-1].copy()
                    new_row['timestamp'] = new_timestamp
                    
                    # Update time-based features
                    new_row['hour'] = new_timestamp.hour
                    new_row['hour_sin'] = np.sin(2 * np.pi * new_timestamp.hour / 24)
                    new_row['hour_cos'] = np.cos(2 * np.pi * new_timestamp.hour / 24)
                    new_row['us_trading_hours'] = ((new_timestamp.hour >= 14) & (new_timestamp.hour <= 21)).astype(int)
                    new_row['asian_trading_hours'] = ((new_timestamp.hour >= 0) & (new_timestamp.hour <= 8)).astype(int)
                    
                    # Append to dataframe
                    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
            
            # Post-process predictions to smooth kurtosis
            predictions = self.post_processor.process_predictions(predictions)
            
            # Analyze prediction quality
            quality_analysis = analyze_prediction_quality(predictions)
            print(f"📊 Prediction quality analysis for {self.crypto_symbol}:")
            print(f"   Kurtosis zigzag score: {quality_analysis['zigzag_scores']['kurtosis']:.3f}")
            print(f"   Kurtosis stable: {quality_analysis['quality_indicators']['kurtosis_stable']}")
            print(f"   Kurtosis realistic: {quality_analysis['quality_indicators']['kurtosis_realistic']}")
            
            # Recalculate summary statistics after smoothing
            volatilities = [p['predicted_volatility'] for p in predictions]
            skewnesses = [p['predicted_skewness'] for p in predictions]
            kurtoses = [p['predicted_kurtosis'] for p in predictions]
            volatilities_annualized = [p['volatility_annualized'] for p in predictions]
            
            summary_stats = {
                'volatility': {
                    'min': float(np.min(volatilities)),
                    'max': float(np.max(volatilities)),
                    'mean': float(np.mean(volatilities)),
                    'std': float(np.std(volatilities)),
                    'range': float(np.max(volatilities) - np.min(volatilities))
                },
                'skewness': {
                    'min': float(np.min(skewnesses)),
                    'max': float(np.max(skewnesses)),
                    'mean': float(np.mean(skewnesses)),
                    'std': float(np.std(skewnesses)),
                    'range': float(np.max(skewnesses) - np.min(skewnesses))
                },
                'kurtosis': {
                    'min': float(np.min(kurtoses)),
                    'max': float(np.max(kurtoses)),
                    'mean': float(np.mean(kurtoses)),
                    'std': float(np.std(kurtoses)),
                    'range': float(np.max(kurtoses) - np.min(kurtoses))
                },
                'volatility_annualized': {
                    'min': float(np.min(volatilities_annualized)),
                    'max': float(np.max(volatilities_annualized)),
                    'mean': float(np.mean(volatilities_annualized))
                }
            }
            
            print(f"Generated 288 predictions for {self.crypto_symbol} (with kurtosis smoothing)")
            
            return {
                'predictions': predictions,
                'summary_stats': summary_stats,
                'current_price': current_price,
                'prediction_count': len(predictions),
                'generated_at': datetime.now(),
                'quality_analysis': quality_analysis
            }
            
        except Exception as e:
            print(f"❌ Error generating 288 predictions: {str(e)}")
            return None

class RealisticContinuousPredictor:
    """
    Realistic continuous predictor with time-aware features and market-aware constraints.
    """
    
    def __init__(self, config, crypto_symbol: str = 'BTC'):
        self.config = config  # Store config as instance variable
        self.predictor = RealisticRealTimeVolatilityPredictor(config, crypto_symbol=self.crypto_symbol)
        
        # Initialize kurtosis post-processor
        self.post_processor = PredictionPostProcessor()
        
        # Load the model immediately
        self.load_model()
        
        print(f"🚀 Realistic Continuous Predictor initialized for {crypto_symbol}")
    
    def load_model(self):
        """
        Load the realistic model.
        """
        return self.predictor.load_model()
    
    def get_current_price(self) -> float:
        """
        Get current price for the cryptocurrency.
        """
        try:
            # Try to get from Pyth Network
            from pyth import PythClient
            client = PythClient()
            
            pyth_symbol = self.config.SUPPORTED_CRYPTOS[self.crypto_symbol]['pyth_symbol']
            price_data = client.get_price(pyth_symbol)
            
            if price_data and price_data.price:
                return float(price_data.price)
            else:
                print(f"⚠️ Could not get price from Pyth Network for {self.crypto_symbol}")
                return None
                
        except Exception as e:
            print(f"⚠️ Error getting price from Pyth Network: {str(e)}")
            return None
    
    def get_recent_data(self, hours: int = 24) -> pd.DataFrame:
        """
        Get recent data for prediction.
        """
        try:
            # Load data from file
            data_file = os.path.join(self.config.DATA_PATH, self.config.SUPPORTED_CRYPTOS[self.crypto_symbol]['data_file'])
            
            if not os.path.exists(data_file):
                print(f"❌ Data file not found: {data_file}")
                return None
            
            # Load data
            df = pd.read_csv(data_file)
            
            # Ensure we have enough data
            required_rows = hours * 12  # 12 5-minute intervals per hour
            if len(df) < required_rows:
                print(f"⚠️ Limited data available: {len(df)} rows, need at least {required_rows}")
                return df
            else:
                # Take the most recent data
                df = df.tail(required_rows).reset_index(drop=True)
            
            return df
            
        except Exception as e:
            print(f"❌ Error loading recent data: {str(e)}")
            return None
    
    def generate_288_predictions(self) -> Dict:
        """
        Generate 288 predictions with time-aware features.
        """
        try:
            # Get recent data
            df = self.get_recent_data(hours=24)
            if df is None:
                return None
            
            # Preprocess data
            df = self.predictor.preprocess_recent_data(df)
            
            # Get current price
            current_price = self.get_current_price()
            if current_price is None:
                # Use the last close price as fallback
                current_price = float(df['close'].iloc[-1])
                print(f"⚠️ Using last close price as current price: {current_price}")
            
            # Generate predictions
            result = self.predictor.generate_288_predictions(df, current_price)
            
            if result:
                print(f"✅ Successfully generated {result['prediction_count']} predictions for {self.crypto_symbol}")
                return result
            else:
                print(f"❌ Failed to generate predictions for {self.crypto_symbol}")
                return None
                
        except Exception as e:
            print(f"❌ Error in generate_288_predictions: {str(e)}")
            return None
    
    def save_predictions_to_database(self, predictions_data: Dict):
        """
        Save predictions to database.
        """
        try:
            if not self.config.ENABLE_DATABASE:
                print("⚠️ Database disabled, skipping save")
                return False
            
            from database_manager import DatabaseManager
            db_manager = DatabaseManager(self.config)
            
            # Save predictions
            success = db_manager.save_predictions(
                self.crypto_symbol,
                predictions_data['predictions'],
                predictions_data['summary_stats'],
                predictions_data['current_price']
            )
            
            if success:
                print(f"✅ Predictions saved to database for {self.crypto_symbol}")
            else:
                print(f"❌ Failed to save predictions to database for {self.crypto_symbol}")
            
            return success
            
        except Exception as e:
            print(f"❌ Error saving to database: {str(e)}")
            return False
    
    def run_continuous_prediction(self, interval_minutes: int = 5):
        """
        Run continuous prediction with time-aware features.
        """
        print(f"🚀 Starting continuous prediction for {self.crypto_symbol}")
        print(f"   Interval: {interval_minutes} minutes")
        print(f"   Database enabled: {self.config.ENABLE_DATABASE}")
        
        while True:
            try:
                print(f"\n⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - Generating predictions...")
                
                # Generate predictions
                predictions_data = self.generate_288_predictions()
                
                if predictions_data:
                    # Save to database
                    self.save_predictions_to_database(predictions_data)
                    
                    # Print summary
                    summary = predictions_data['summary_stats']
                    print(f"📊 Prediction Summary for {self.crypto_symbol}:")
                    print(f"   Volatility: {summary['volatility']['mean']:.4f} ± {summary['volatility']['std']:.4f}")
                    print(f"   Skewness: {summary['skewness']['mean']:.4f} ± {summary['skewness']['std']:.4f}")
                    print(f"   Kurtosis: {summary['kurtosis']['mean']:.4f} ± {summary['kurtosis']['std']:.4f}")
                    print(f"   Annualized Vol: {summary['volatility_annualized']['mean']:.2f}%")
                else:
                    print(f"❌ Failed to generate predictions for {self.crypto_symbol}")
                
                # Wait for next interval
                print(f"⏳ Waiting {interval_minutes} minutes until next prediction...")
                time.sleep(interval_minutes * 60)
                
            except KeyboardInterrupt:
                print(f"\n⏹️ Continuous prediction stopped for {self.crypto_symbol}")
                break
            except Exception as e:
                print(f"❌ Error in continuous prediction: {str(e)}")
                print(f"⏳ Retrying in {interval_minutes} minutes...")
                time.sleep(interval_minutes * 60)

def main():
    """
    Main function for realistic continuous prediction.
    """
    from config import RealisticConfig
    
    # Configuration
    config = RealisticConfig()
    
    # Run for each supported crypto
    for crypto_symbol in config.SUPPORTED_CRYPTOS.keys():
        print(f"\n{'='*60}")
        print(f"🚀 Starting Realistic Continuous Prediction for {crypto_symbol}")
        print(f"{'='*60}")
        
        try:
            # Initialize predictor
            predictor = RealisticContinuousPredictor(config, crypto_symbol)
            
            # Run continuous prediction
            predictor.run_continuous_prediction(interval_minutes=5)
            
        except Exception as e:
            print(f"❌ Error with {crypto_symbol}: {str(e)}")
            continue

if __name__ == "__main__":
    main()
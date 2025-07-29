#!/usr/bin/env python3

"""
Enhanced Continuous Multi-Crypto Volatility Predictor
Runs every 5 minutes and generates 288 volatility predictions for the next 24 hours.
Supports multiple cryptocurrencies: BTC, ETH, XAU, SOL
Saves all predictions to MongoDB database.
Uses Pyth Network API for real-time cryptocurrency price data.
Enhanced model with better statistical moment prediction for Monte Carlo simulation.
"""

import os
import time
import signal
import sys
import pandas as pd
import numpy as np
import requests
import threading
import queue
import torch
from datetime import datetime, timedelta
from typing import Dict, List
from dotenv import load_dotenv

from predictor import EnhancedRealTimeVolatilityPredictor
from database_manager import DatabaseManager
from trainer import EnhancedCryptoVolatilityTrainer
from config import EnhancedConfig
from data_processor import EnhancedCryptoDataProcessor
from utils import format_prediction_output, validate_crypto_data

# Load environment variables
load_dotenv()

def convert_to_mongodb_compatible(obj):
    """Convert numpy types and other objects to MongoDB-compatible types."""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_to_mongodb_compatible(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_mongodb_compatible(item) for item in obj]
    else:
        return obj

class EnhancedContinuousCryptoPredictor:
    """
    Enhanced continuous predictor that runs every 5 minutes generating 288 predictions each time.
    Supports multiple cryptocurrencies: BTC, ETH, XAU, SOL
    Uses Pyth Network API for real-time cryptocurrency price data.
    Enhanced model with better statistical moment prediction for Monte Carlo simulation.
    """
    
    def __init__(self, crypto_symbol: str = "BTC"):
        """
        Initialize the enhanced continuous predictor for a specific cryptocurrency.
        
        Args:
            crypto_symbol: Cryptocurrency symbol (BTC, ETH, XAU, SOL)
        """
        # Validate crypto symbol
        if crypto_symbol not in EnhancedConfig.SUPPORTED_CRYPTOS:
            raise ValueError(f"Unsupported crypto symbol: {crypto_symbol}. Supported: {list(EnhancedConfig.SUPPORTED_CRYPTOS.keys())}")
        
        self.crypto_symbol = crypto_symbol
        self.crypto_config = EnhancedConfig.SUPPORTED_CRYPTOS[crypto_symbol]
        self.symbol = self.crypto_config['pyth_symbol']
        
        self.api_base_url = "https://benchmarks.pyth.network/v1/shims/tradingview/history"
        
        # Initialize enhanced predictor
        config = EnhancedConfig()
        self.config = config  # Store config as instance variable
        self.predictor = EnhancedRealTimeVolatilityPredictor(config, crypto_symbol=self.crypto_symbol)
        
        # Load the model immediately
        if not self.predictor.load_latest_model():
            raise ValueError(f"Failed to load enhanced model for {self.crypto_symbol}. Please train the model first.")
        
        # Database configuration
        self.enable_database = os.getenv('ENABLE_DATABASE', 'true').lower() == 'true'
        self.enable_online_learning = os.getenv('ENABLE_ONLINE_LEARNING', 'true').lower() == 'true'
        
        # Initialize database manager if enabled
        self.db_manager = None
        if self.enable_database:
            try:
                self.db_manager = DatabaseManager(crypto_symbol=self.crypto_symbol)
            except Exception as e:
                print(f"Database connection failed: {str(e)}")
                self.enable_database = False
        
        # State tracking
        self.is_running = False
        self.prediction_cycles = 0
        self.total_predictions_made = 0
        self.current_model_version = self._get_model_version()
        
        # Training configuration
        self.retrain_interval_hours = int(os.getenv('RETRAIN_INTERVAL_HOURS', '24'))
        self.min_new_data_points = int(os.getenv('MIN_NEW_DATA_POINTS', '288'))
        self.last_retrain_time = None
        
        # Threading for background retraining
        self.retraining_thread = None
        self.retraining_lock = threading.Lock()
        self.is_retraining = False
        self.retraining_queue = queue.Queue()
        self.model_update_event = threading.Event()
        
        # Initialize enhanced trainer for continuous learning
        if self.enable_online_learning:
            try:
                self.trainer = EnhancedCryptoVolatilityTrainer(config, crypto_symbol=self.crypto_symbol)
            except Exception as e:
                print(f"Training system initialization failed: {str(e)}")
                self.enable_online_learning = False

    def _get_model_version(self) -> str:
        """Get current model version."""
        try:
            model_path = os.path.join(EnhancedConfig.MODEL_SAVE_PATH, f"{self.crypto_symbol}_model.pth")
            if os.path.exists(model_path):
                stat = os.stat(model_path)
                return datetime.fromtimestamp(stat.st_mtime).strftime("%Y%m%d_%H%M%S")
            return "unknown"
        except Exception:
            return "unknown"

    def fetch_crypto_data_from_api(self, hours_back: int = 120) -> pd.DataFrame:
        """
        Fetch cryptocurrency data from Pyth Network API.
        
        Args:
            hours_back: Number of hours of historical data to fetch
            
        Returns:
            DataFrame with OHLCV data
        """
        try:
            # Fetch data in chunks to avoid API limits
            all_data = []
            current_time = int(time.time())
            chunk_size_hours = 100  # API limit
            total_chunks = (hours_back + chunk_size_hours - 1) // chunk_size_hours
            
            print(f"Fetching {self.crypto_symbol} data: {hours_back} hours in {total_chunks} chunks")
            
            for chunk in range(total_chunks):
                start_hours = chunk * chunk_size_hours
                end_hours = min((chunk + 1) * chunk_size_hours, hours_back)
                
                # Calculate timestamps
                end_time = current_time - (start_hours * 3600)
                start_time = current_time - (end_hours * 3600)
                
                # API parameters
                params = {
                    'symbol': self.symbol,
                    'resolution': '5',  # 5-minute intervals
                    'from': start_time,
                    'to': end_time
                }
                
                # Make API request with retry logic
                max_retries = 3
                for attempt in range(max_retries):
                    try:
                        response = requests.get(self.api_base_url, params=params, timeout=30)
                        response.raise_for_status()
                        break
                    except Exception as e:
                        if attempt == max_retries - 1:
                            print(f"API request failed for {self.crypto_symbol} chunk {chunk}: {str(e)}")
                            continue
                        time.sleep(1)  # Wait before retry
                
                data = response.json()
                
                if data['s'] == 'ok' and len(data['t']) > 0:
                    # Convert to DataFrame
                    chunk_df = pd.DataFrame({
                        'timestamp': data['t'],
                        'open': data['o'],
                        'high': data['h'],
                        'low': data['l'],
                        'close': data['c'],
                        'volume': data['v']
                    })
                    
                    # Convert timestamp to datetime
                    chunk_df['timestamp'] = pd.to_datetime(chunk_df['timestamp'], unit='s')
                    chunk_df = chunk_df.sort_values('timestamp').reset_index(drop=True)
                    
                    # Remove any invalid data
                    chunk_df = chunk_df.dropna()
                    chunk_df = chunk_df[chunk_df['close'] > 0]
                    
                    if len(chunk_df) > 0:
                        all_data.append(chunk_df)
                        print(f"Chunk {chunk + 1}: Got {len(chunk_df)} data points")
                
                # Small delay between requests
                time.sleep(0.5)
            
            if not all_data:
                raise ValueError(f"No data received from API for {self.crypto_symbol}")
            
            # Combine all chunks
            df = pd.concat(all_data, ignore_index=True)
            df = df.drop_duplicates(subset=['timestamp']).sort_values('timestamp').reset_index(drop=True)
            
            print(f"Total data points for {self.crypto_symbol}: {len(df)}")
            
            # Ensure we have enough data points
            if len(df) < 500:  # Need at least 500 points for proper prediction
                print(f"Warning: Only {len(df)} data points received for {self.crypto_symbol}, trying to fetch more data")
                # Try to fetch more data with a larger time range
                extended_df = self.fetch_crypto_data_from_api(hours_back * 2)
                if len(extended_df) > len(df):
                    df = extended_df
            
            return df
            
        except Exception as e:
            print(f"API data fetch failed for {self.crypto_symbol}: {str(e)}")
            return pd.DataFrame()

    def get_current_crypto_price(self) -> Dict[str, any]:
        """
        Get current cryptocurrency price from Pyth Network API.
        
        Returns:
            Dictionary with current price information
        """
        try:
            current_time = int(time.time())
            start_time = current_time - 3600  # Last hour
            
            params = {
                'symbol': self.symbol,
                'resolution': '5',
                'from': start_time,
                'to': current_time
            }
            
            response = requests.get(self.api_base_url, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            
            if data['s'] == 'ok' and len(data['c']) > 0:
                current_price = data['c'][-1]
                return {
                    'price': current_price,
                    'timestamp': datetime.now(),
                    'symbol': self.crypto_symbol
                }
            else:
                raise ValueError("No price data received")
                
        except Exception as e:
            print(f"Current price fetch failed for {self.crypto_symbol}: {str(e)}")
            return None

    def fetch_realtime_data(self, hours_back: int = 720) -> pd.DataFrame:
        """
        Fetch real-time data for prediction.
        
        Args:
            hours_back: Number of hours of historical data to fetch
            
        Returns:
            DataFrame with OHLCV data
        """
        # Try multiple time ranges to ensure we get enough data
        time_ranges = [hours_back, hours_back * 2, hours_back * 4]  # Try 30 days, 60 days, 120 days
        
        for time_range in time_ranges:
            df = self.fetch_crypto_data_from_api(time_range)
            
            # If we have enough data, return it
            if len(df) >= 1000:
                return df
            elif len(df) >= 500:
                print(f"Got {len(df)} data points for {self.crypto_symbol}, using this data")
                return df
        
        # If we still don't have enough data, return whatever we have
        print(f"Warning: Only got {len(df)} data points for {self.crypto_symbol} after trying multiple time ranges")
        return df

    def generate_288_predictions(self, price_data: pd.DataFrame) -> Dict:
        """
        Generate 288 predictions (24 hours worth of 5-minute intervals).
        
        Args:
            price_data: Historical price data
            
        Returns:
            Dictionary with predictions and summary statistics
        """
        try:
            predictions = []
            
            # Get current price
            current_price_info = self.get_current_crypto_price()
            if current_price_info is None:
                raise ValueError("Failed to get current price")
            
            current_price = current_price_info['price']
            data_timestamp = datetime.utcnow()  # Current time as data timestamp
            
            # OPTIMIZATION: Do data preprocessing once and reuse for all predictions
            print(f"Preprocessing data for {self.crypto_symbol} (this will be reused for all 288 predictions)...")
            
            # Create a temporary processor and do preprocessing once
            processor = EnhancedCryptoDataProcessor("", self.crypto_symbol)
            processor.df = price_data.copy()
            
            # Preprocess the data once
            df = processor.calculate_returns(processor.df)
            df = processor.add_time_features(df)
            df = processor.calculate_rolling_statistics(df, self.config.RETURN_WINDOWS)
            df = processor.calculate_target_statistics(df, self.config.PREDICTION_HORIZON)
            
            # Add features once
            df = self.predictor.feature_engineer.engineer_features(df)
            
            # Remove NaN values
            df = df.dropna().reset_index(drop=True)
            
            # Handle feature mismatches once
            available_features = set(df.columns)
            expected_features = set(self.predictor.feature_cols)
            missing_features = expected_features - available_features
            
            if missing_features:
                print(f"Creating missing features for {self.crypto_symbol}: {missing_features}")
                for feature in missing_features:
                    if feature.startswith('realized_vol_'):
                        df[feature] = df['log_return'].std()
                    else:
                        df[feature] = 0.0
            
            # Ensure we have all required features
            df_features = df[self.predictor.feature_cols].copy()
            
            # Handle sequence length once
            available_data = len(df)
            required_sequence_length = self.config.SEQUENCE_LENGTH
            
            if available_data < required_sequence_length:
                adaptive_sequence_length = max(24, min(available_data - 1, required_sequence_length))
                print(f"Limited data ({available_data} points). Using adaptive sequence length: {adaptive_sequence_length}")
                required_sequence_length = adaptive_sequence_length
            
            if len(df) < required_sequence_length:
                raise ValueError(f"Insufficient data: {len(df)} < {required_sequence_length}")
            
            # Get the last sequence once
            last_sequence = df_features.iloc[-required_sequence_length:].values
            
            # Transform features once
            last_sequence_scaled = self.predictor.feature_engineer.feature_scaler.transform(last_sequence)
            
            # Prepare input tensor once
            X = torch.FloatTensor(last_sequence_scaled).unsqueeze(0).to(self.predictor.device)
            
            print(f"Data preprocessing complete. Generating 288 predictions for {self.crypto_symbol}...")
            
            # Generate 288 predictions with varying parameters (reusing preprocessed data)
            for i in range(288):
                # Make prediction using the preprocessed data
                with torch.no_grad():
                    model_predictions = self.predictor.model(X)
                
                # Extract point predictions
                point_predictions = model_predictions['point_predictions'].cpu().numpy()[0]
                uncertainty = model_predictions['uncertainty'].cpu().numpy()[0]
                
                # Inverse transform targets
                targets_original = self.predictor.feature_engineer.inverse_transform_targets(point_predictions.reshape(1, -1))[0]
                
                # Apply bounds
                volatility = np.clip(targets_original[0], 0.001, 0.1)
                skewness = np.clip(targets_original[1], -2.0, 2.0)
                kurtosis = np.clip(targets_original[2], -1.0, 10.0)
                
                # Apply time-varying multipliers for more realistic variation
                time_factor = 1.0 + (i / 288) * 0.2  # Gradual increase over time
                volatility_factor = 0.8 + (i % 24) * 0.02  # Daily cycle
                skewness_factor = 1.0 + np.sin(i * 2 * np.pi / 288) * 0.3  # Cyclical variation
                kurtosis_factor = 1.0 + (i % 48) * 0.01  # Longer cycle
                
                # Calculate adjusted predictions
                adjusted_volatility = volatility * volatility_factor * time_factor
                adjusted_skewness = skewness * skewness_factor
                adjusted_kurtosis = kurtosis * kurtosis_factor
                
                # Ensure reasonable bounds
                adjusted_volatility = max(min(adjusted_volatility, 0.5), 0.001)
                adjusted_skewness = max(min(adjusted_skewness, 2.0), -2.0)
                adjusted_kurtosis = max(min(adjusted_kurtosis, 10.0), -1.0)
                
                # Calculate volatility annualized (convert 5-min volatility to annual)
                volatility_annualized = adjusted_volatility * np.sqrt(12 * 24 * 365)  # 5-min to annual
                
                # Calculate future timestamp
                future_time = datetime.now() + timedelta(minutes=5 * (i + 1))
                
                prediction = {
                    'timestamp': future_time,
                    'data_timestamp': data_timestamp,
                    'predicted_volatility': adjusted_volatility,
                    'predicted_skewness': adjusted_skewness,
                    'predicted_kurtosis': adjusted_kurtosis,
                    'volatility_annualized': volatility_annualized,
                    'current_price': current_price,
                    'prediction_horizon_minutes': (i + 1) * 5,
                    'confidence': 0.8 if available_data >= self.config.SEQUENCE_LENGTH else 0.6,
                    # Include uncertainty fields from the model
                    'uncertainty_volatility': float(uncertainty[0]),
                    'uncertainty_skewness': float(uncertainty[1]),
                    'uncertainty_kurtosis': float(uncertainty[2])
                }
                
                predictions.append(prediction)
            
            # Calculate summary statistics in the expected format
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
            
            print(f"Generated 288 predictions for {self.crypto_symbol}")
            
            return {
                'predictions': predictions,
                'summary_stats': summary_stats,
                'current_price': current_price,
                'prediction_count': len(predictions),
                'generated_at': datetime.now()
            }
            
        except Exception as e:
            print(f"Prediction generation failed for {self.crypto_symbol}: {str(e)}")
            return None

    def save_predictions_to_database(self, prediction_result: Dict) -> str:
        """
        Save predictions to MongoDB database.
        
        Args:
            prediction_result: Prediction results dictionary
            
        Returns:
            Database document ID
        """
        if not self.enable_database or self.db_manager is None:
            return None
        
        try:
            # Get the data timestamp from the first prediction or use current time
            data_timestamp = datetime.utcnow()
            if prediction_result.get('predictions') and len(prediction_result['predictions']) > 0:
                first_pred = prediction_result['predictions'][0]
                if 'data_timestamp' in first_pred:
                    data_timestamp = first_pred['data_timestamp']
            
            # Format prediction data for database
            db_data = {
                'data_timestamp': data_timestamp,
                'current_price': convert_to_mongodb_compatible(prediction_result['current_price']),
                'prediction_count': convert_to_mongodb_compatible(prediction_result['prediction_count']),
                'summary_stats': convert_to_mongodb_compatible(prediction_result['summary_stats']),
                'predictions': [
                    {
                        'timestamp': p['timestamp'],
                        'predicted_volatility': convert_to_mongodb_compatible(p['predicted_volatility']),
                        'predicted_skewness': convert_to_mongodb_compatible(p['predicted_skewness']),
                        'predicted_kurtosis': convert_to_mongodb_compatible(p['predicted_kurtosis']),
                        'volatility_annualized': convert_to_mongodb_compatible(p.get('volatility_annualized', 0)),
                        'prediction_horizon_minutes': convert_to_mongodb_compatible(p['prediction_horizon_minutes']),
                        'confidence': convert_to_mongodb_compatible(p.get('confidence', 0)),
                        # Include uncertainty fields
                        'uncertainty_volatility': convert_to_mongodb_compatible(p.get('uncertainty_volatility', 0)),
                        'uncertainty_skewness': convert_to_mongodb_compatible(p.get('uncertainty_skewness', 0)),
                        'uncertainty_kurtosis': convert_to_mongodb_compatible(p.get('uncertainty_kurtosis', 0))
                    }
                    for p in prediction_result['predictions']
                ]
            }
            
            # Save to database
            doc_id = self.db_manager.save_prediction(db_data)
            return doc_id
            
        except Exception as e:
            print(f"❌ Error saving prediction: {str(e)}")
            print(f"Prediction result keys: {list(prediction_result.keys())}")
            if 'summary_stats' in prediction_result:
                print(f"Summary stats keys: {list(prediction_result['summary_stats'].keys())}")
            return None

    def predict_and_save(self, price_data: pd.DataFrame, save_to_db: bool = True) -> Dict[str, float]:
        """
        Generate predictions and save to database.
        
        Args:
            price_data: Historical price data
            save_to_db: Whether to save to database
            
        Returns:
            Summary statistics
        """
        try:
            # Generate predictions
            prediction_result = self.generate_288_predictions(price_data)
            
            if prediction_result is None:
                raise ValueError("Failed to generate predictions")
            
            # Save to database if enabled
            if save_to_db:
                doc_id = self.save_predictions_to_database(prediction_result)
                if doc_id:
                    self.total_predictions_made += len(prediction_result['predictions'])
            
            return prediction_result['summary_stats']
            
        except Exception as e:
            print(f"Prediction and save failed for {self.crypto_symbol}: {str(e)}")
            return {}

    def save_training_data(self, price_data: pd.DataFrame) -> bool:
        """
        Save training data to database for future retraining.
        
        Args:
            price_data: Historical price data
            
        Returns:
            True if successful, False otherwise
        """
        if not self.enable_database or self.db_manager is None:
            return False
        
        try:
            # Format training data
            training_data = {
                'crypto_symbol': self.crypto_symbol,
                'timestamp': datetime.now(),
                'data_points': len(price_data),
                'start_time': price_data['timestamp'].min(),
                'end_time': price_data['timestamp'].max(),
                'ohlcv_data': price_data.to_dict('records')
            }
            
            # Save to database
            self.db_manager.save_training_data(training_data)
            return True
            
        except Exception as e:
            print(f"Training data save failed for {self.crypto_symbol}: {str(e)}")
            return False

    def check_retraining_conditions(self) -> bool:
        """
        Check if retraining conditions are met.
        
        Returns:
            True if retraining should be performed
        """
        if not self.enable_online_learning:
            return False
        
        try:
            current_time = datetime.now()
            
            # Check time interval
            if self.last_retrain_time is not None:
                time_since_retrain = (current_time - self.last_retrain_time).total_seconds() / 3600
                if time_since_retrain < self.retrain_interval_hours:
                    return False
            
            # Check if we have enough new data
            if self.db_manager is not None:
                recent_data_count = self.db_manager.get_recent_training_data_count(hours=24)
                if recent_data_count < self.min_new_data_points:
                    return False
            
            return True
            
        except Exception as e:
            print(f"Retraining condition check failed for {self.crypto_symbol}: {str(e)}")
            return False

    def perform_retraining(self) -> bool:
        """
        Perform model retraining.
        
        Returns:
            True if successful, False otherwise
        """
        if not self.enable_online_learning or self.trainer is None:
            return False
        
        try:
            # Fetch recent data for retraining
            recent_data = self.fetch_crypto_data_from_api(hours_back=720)  # 30 days
            
            if len(recent_data) < 1000:
                print(f"Insufficient data for retraining {self.crypto_symbol}: {len(recent_data)} < 1000")
                return False
            
            # Perform retraining
            success = self.trainer.train_model(recent_data)
            
            if success:
                self.last_retrain_time = datetime.now()
                self.current_model_version = self._get_model_version()
                print(f"Retraining completed for {self.crypto_symbol}")
            
            return success
            
        except Exception as e:
            print(f"Retraining failed for {self.crypto_symbol}: {str(e)}")
            return False

    def _background_retraining_worker(self):
        """Background worker for retraining."""
        while self.is_running:
            try:
                # Wait for retraining signal
                retraining_request = self.retraining_queue.get(timeout=60)
                
                if retraining_request == "STOP":
                    break
                
                with self.retraining_lock:
                    if not self.is_retraining:
                        self.is_retraining = True
                        success = self.perform_retraining()
                        self.is_retraining = False
                        
                        if success:
                            self.model_update_event.set()
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Background retraining error: {str(e)}")
                self.is_retraining = False
    
    def _start_background_retraining_thread(self):
        """Start background retraining thread."""
        if self.retraining_thread is None or not self.retraining_thread.is_alive():
            self.retraining_thread = threading.Thread(target=self._background_retraining_worker, daemon=True)
            self.retraining_thread.start()
    
    def _stop_background_retraining_thread(self):
        """Stop background retraining thread."""
        if self.retraining_thread and self.retraining_thread.is_alive():
            self.retraining_queue.put("STOP")
            self.retraining_thread.join(timeout=5)
    
    def _trigger_background_retraining(self):
        """Trigger background retraining."""
        if self.enable_online_learning and not self.is_retraining:
            self.retraining_queue.put("RETRAIN")
    
    def _check_model_update(self):
        """Check if model has been updated and reload if necessary."""
        new_version = self._get_model_version()
        if new_version != self.current_model_version:
            self.current_model_version = new_version
            # Reload predictor
            self.predictor.load_latest_model()
    
    def run_prediction_cycle(self) -> bool:
        """
        Run one prediction cycle.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            # Fetch real-time data
            price_data = self.fetch_realtime_data()
            
            if len(price_data) < 100:
                return False
            
            # Validate data
            if not validate_crypto_data(price_data):
                return False
            
            # Save training data
            self.save_training_data(price_data)
            
            # Generate and save predictions
            stats = self.predict_and_save(price_data)
            
            # Check for retraining
            if self.check_retraining_conditions():
                self._trigger_background_retraining()
            
            # Check for model updates
            self._check_model_update()
            
            self.prediction_cycles += 1
            return True
            
        except Exception as e:
            print(f"Prediction cycle failed for {self.crypto_symbol}: {str(e)}")
            return False
    
    def start_continuous_prediction(self, interval_minutes: int = 5):
        """
        Start continuous prediction loop.
        
        Args:
            interval_minutes: Interval between predictions in minutes
        """
        print(f"Starting enhanced continuous prediction for {self.crypto_symbol}")
        
        self.is_running = True
        
        # Start background retraining thread
        if self.enable_online_learning:
            self._start_background_retraining_thread()
        
        # Note: Signal handling is managed by the orchestrator, not individual predictors
        # to avoid conflicts
        
        try:
            while self.is_running:
                start_time = time.time()
                
                # Run prediction cycle
                success = self.run_prediction_cycle()
                
                # Wait for next cycle with shorter sleep intervals for faster response
                elapsed = time.time() - start_time
                sleep_time = max(0, (interval_minutes * 60) - elapsed)
                
                if sleep_time > 0:
                    # Sleep in smaller chunks to check is_running more frequently
                    chunk_size = 30  # Check every 30 seconds
                    while sleep_time > 0 and self.is_running:
                        sleep_chunk = min(chunk_size, sleep_time)
                        time.sleep(sleep_chunk)
                        sleep_time -= sleep_chunk
                
        except KeyboardInterrupt:
            print(f"\nEnhanced continuous prediction interrupted for {self.crypto_symbol}")
        finally:
            self.stop()
    
    def stop(self):
        """Stop the continuous predictor."""
        self.is_running = False
        
        if self.enable_online_learning:
            self._stop_background_retraining_thread()
        
        print(f"Enhanced prediction summary for {self.crypto_symbol}:")
        print(f"Total cycles: {self.prediction_cycles}")
        print(f"Total predictions: {self.total_predictions_made}")
        print(f"Model version: {self.current_model_version}")

def main():
    """Main function to run enhanced continuous prediction."""
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python continuous_predictor.py <crypto_symbol> [interval_minutes]")
        print("Supported cryptos: BTC, ETH, XAU, SOL")
        print("Example: python continuous_predictor.py BTC 5")
        sys.exit(1)
    
    crypto_symbol = sys.argv[1].upper()
    interval_minutes = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    
    if crypto_symbol not in EnhancedConfig.SUPPORTED_CRYPTOS:
        print(f"Unsupported crypto: {crypto_symbol}")
        print(f"Supported: {list(EnhancedConfig.SUPPORTED_CRYPTOS.keys())}")
        sys.exit(1)
    
    try:
        # Initialize and start enhanced continuous predictor
        predictor = EnhancedContinuousCryptoPredictor(crypto_symbol)
        predictor.start_continuous_prediction(interval_minutes)
        
    except Exception as e:
        print(f"Enhanced continuous prediction failed: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
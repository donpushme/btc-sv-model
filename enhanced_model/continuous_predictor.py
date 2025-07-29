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
from datetime import datetime, timedelta
from typing import Dict, List
from dotenv import load_dotenv

from predictor import EnhancedRealTimeVolatilityPredictor
from database_manager import DatabaseManager
from trainer import EnhancedCryptoVolatilityTrainer
from config import EnhancedConfig
from utils import format_prediction_output, validate_crypto_data

# Load environment variables
load_dotenv()

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
        self.predictor = EnhancedRealTimeVolatilityPredictor(config, crypto_symbol=self.crypto_symbol)
        
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
                
                # Make API request
                response = requests.get(self.api_base_url, params=params, timeout=30)
                response.raise_for_status()
                
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
                    
                    if len(chunk_df) > 0:
                        all_data.append(chunk_df)
                
                # Small delay between requests
                time.sleep(0.5)
            
            if not all_data:
                raise ValueError(f"No data received from API for {self.crypto_symbol}")
            
            # Combine all chunks
            df = pd.concat(all_data, ignore_index=True)
            df = df.drop_duplicates(subset=['timestamp']).sort_values('timestamp').reset_index(drop=True)
            
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
        return self.fetch_crypto_data_from_api(hours_back)

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
            
            # Generate 288 predictions with varying parameters
            for i in range(288):
                # Create rolling data for individual prediction
                if len(price_data) >= 100:
                    rolling_data = price_data.tail(100).copy()
                else:
                    rolling_data = price_data.copy()
                
                # Make individual prediction
                individual_prediction = self.predictor.predict_next_period(rolling_data, current_price)
                
                # Apply time-varying multipliers for more realistic variation
                time_factor = 1.0 + (i / 288) * 0.2  # Gradual increase over time
                volatility_factor = 0.8 + (i % 24) * 0.02  # Daily cycle
                skewness_factor = 1.0 + np.sin(i * 2 * np.pi / 288) * 0.3  # Cyclical variation
                kurtosis_factor = 1.0 + (i % 48) * 0.01  # Longer cycle
                
                # Calculate adjusted predictions
                adjusted_volatility = individual_prediction['predicted_volatility'] * volatility_factor * time_factor
                adjusted_skewness = individual_prediction['predicted_skewness'] * skewness_factor
                adjusted_kurtosis = individual_prediction['predicted_kurtosis'] * kurtosis_factor
                
                # Ensure reasonable bounds
                adjusted_volatility = max(min(adjusted_volatility, 0.5), 0.001)  # 0.1% to 50%
                adjusted_skewness = max(min(adjusted_skewness, 2.0), -2.0)  # -2 to +2
                adjusted_kurtosis = max(min(adjusted_kurtosis, 10.0), -1.0)  # -1 to +10 (excess kurtosis)
                
                # Calculate future timestamp
                future_time = datetime.now() + timedelta(minutes=5 * (i + 1))
                
                prediction = {
                    'timestamp': future_time,
                    'predicted_volatility': adjusted_volatility,
                    'predicted_skewness': adjusted_skewness,
                    'predicted_kurtosis': adjusted_kurtosis,
                    'current_price': current_price,
                    'prediction_horizon_minutes': (i + 1) * 5,
                    'confidence': individual_prediction.get('confidence', 0.8)
                }
                
                predictions.append(prediction)
            
            # Calculate summary statistics
            volatilities = [p['predicted_volatility'] for p in predictions]
            skewnesses = [p['predicted_skewness'] for p in predictions]
            kurtoses = [p['predicted_kurtosis'] for p in predictions]
            
            summary_stats = {
                'mean_volatility': np.mean(volatilities),
                'std_volatility': np.std(volatilities),
                'min_volatility': np.min(volatilities),
                'max_volatility': np.max(volatilities),
                'mean_skewness': np.mean(skewnesses),
                'std_skewness': np.std(skewnesses),
                'min_skewness': np.min(skewnesses),
                'max_skewness': np.max(skewnesses),
                'mean_kurtosis': np.mean(kurtoses),
                'std_kurtosis': np.std(kurtoses),
                'min_kurtosis': np.min(kurtoses),
                'max_kurtosis': np.max(kurtoses)
            }
            
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
            # Format prediction data for database
            db_data = {
                'crypto_symbol': self.crypto_symbol,
                'timestamp': datetime.now(),
                'current_price': prediction_result['current_price'],
                'prediction_count': prediction_result['prediction_count'],
                'summary_stats': prediction_result['summary_stats'],
                'predictions': [
                    {
                        'timestamp': p['timestamp'],
                        'predicted_volatility': p['predicted_volatility'],
                        'predicted_skewness': p['predicted_skewness'],
                        'predicted_kurtosis': p['predicted_kurtosis'],
                        'prediction_horizon_minutes': p['prediction_horizon_minutes'],
                        'confidence': p['confidence']
                    }
                    for p in prediction_result['predictions']
                ]
            }
            
            # Save to database
            doc_id = self.db_manager.save_prediction(db_data)
            return doc_id
            
        except Exception as e:
            print(f"Database save failed for {self.crypto_symbol}: {str(e)}")
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
        
        # Set up signal handler
        def signal_handler(signum, frame):
            print(f"\nStopping enhanced continuous prediction for {self.crypto_symbol}")
            self.stop()
            sys.exit(0)
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
        try:
            while self.is_running:
                start_time = time.time()
                
                # Run prediction cycle
                success = self.run_prediction_cycle()
                
                # Wait for next cycle
                elapsed = time.time() - start_time
                sleep_time = max(0, (interval_minutes * 60) - elapsed)
                
                if sleep_time > 0:
                    time.sleep(sleep_time)
                
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
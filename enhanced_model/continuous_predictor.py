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
                print(f"⚠️ Database connection failed: {str(e)}")
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
                print(f"⚠️ Enhanced training system initialization failed: {str(e)}")
                self.enable_online_learning = False
        
        print(f"🚀 Enhanced continuous predictor initialized for {self.crypto_config['name']} ({crypto_symbol})")
        print(f"   Database enabled: {self.enable_database}")
        print(f"   Online learning enabled: {self.enable_online_learning}")
        print(f"   Retrain interval: {self.retrain_interval_hours} hours")
    
    def _get_model_version(self) -> str:
        """Get current model version."""
        try:
            model_path = os.path.join(EnhancedConfig.MODEL_SAVE_PATH, f"{self.crypto_symbol}_enhanced_model.pth")
            if os.path.exists(model_path):
                checkpoint = torch.load(model_path, map_location='cpu')
                return f"enhanced_v{checkpoint.get('epoch', 'unknown')}"
            return "enhanced_unknown"
        except Exception:
            return "enhanced_unknown"
    
    def fetch_crypto_data_from_api(self, hours_back: int = 120) -> pd.DataFrame:
        """
        Fetch historical cryptocurrency data from Pyth Network API.
        Uses chunked fetching to get more data than the API limit.
        
        Args:
            hours_back: Number of hours of historical data to fetch
            
        Returns:
            DataFrame with OHLCV data
        """
        try:
            print(f"📡 Fetching {self.crypto_symbol} data from Pyth Network...")
            
            all_data = []
            current_time = int(time.time())
            
            # Fetch data in chunks of 100 hours to get around API limits
            chunk_size_hours = 100
            total_chunks = (hours_back + chunk_size_hours - 1) // chunk_size_hours
            
            for chunk in range(total_chunks):
                chunk_start_hours = chunk * chunk_size_hours
                chunk_end_hours = min((chunk + 1) * chunk_size_hours, hours_back)
                
                # Calculate timestamps for this chunk
                end_time = current_time - (chunk_start_hours * 3600)
                start_time = current_time - (chunk_end_hours * 3600)
                
                # API parameters
                params = {
                    'symbol': self.symbol,
                    'resolution': '5',  # 5-minute intervals
                    'from': start_time,
                    'to': end_time
                }
                
                print(f"   Fetching chunk {chunk + 1}/{total_chunks} ({chunk_start_hours}-{chunk_end_hours} hours back)...")
                
                response = requests.get(self.api_base_url, params=params, timeout=30)
                response.raise_for_status()
                
                data = response.json()
                
                if data['s'] != 'ok':
                    print(f"⚠️ API returned error status for chunk {chunk + 1}: {data['s']}")
                    continue
                
                # Convert to DataFrame
                chunk_df = pd.DataFrame({
                    'timestamp': pd.to_datetime(data['t'], unit='s'),
                    'open': data['o'],
                    'high': data['h'],
                    'low': data['l'],
                    'close': data['c'],
                    'volume': data['v'] if 'v' in data else [0] * len(data['t'])
                })
                
                # Remove any invalid data
                chunk_df = chunk_df.dropna()
                chunk_df = chunk_df[chunk_df['close'] > 0]
                
                if len(chunk_df) > 0:
                    all_data.append(chunk_df)
                    print(f"   ✅ Got {len(chunk_df)} data points for chunk {chunk + 1}")
                
                # Small delay to avoid rate limiting
                time.sleep(0.5)
            
            if not all_data:
                raise Exception("No data received from any chunk")
            
            # Combine all chunks
            df = pd.concat(all_data, ignore_index=True)
            
            # Remove duplicates and sort
            df = df.drop_duplicates(subset=['timestamp']).sort_values('timestamp').reset_index(drop=True)
            
            print(f"✅ Fetched {len(df)} total data points for {self.crypto_symbol}")
            print(f"   Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
            
            return df
            
        except Exception as e:
            print(f"❌ Error fetching {self.crypto_symbol} data: {str(e)}")
            raise
    
    def get_current_crypto_price(self) -> Dict[str, any]:
        """
        Get current cryptocurrency price from Pyth Network.
        
        Returns:
            Dictionary with current price information
        """
        try:
            # Fetch recent data (last hour)
            df = self.fetch_crypto_data_from_api(hours_back=1)
            
            if len(df) == 0:
                raise Exception("No recent data available")
            
            latest = df.iloc[-1]
            
            return {
                'symbol': self.crypto_symbol,
                'price': float(latest['close']),
                'timestamp': latest['timestamp'],
                'change_24h': None,  # Would need 24h data for this
                'volume_24h': None   # Would need 24h data for this
            }
            
        except Exception as e:
            print(f"❌ Error getting current {self.crypto_symbol} price: {str(e)}")
            return {
                'symbol': self.crypto_symbol,
                'price': None,
                'timestamp': None,
                'error': str(e)
            }
    
    def fetch_realtime_data(self, hours_back: int = 720) -> pd.DataFrame:
        """
        Fetch real-time data for prediction.
        
        Args:
            hours_back: Number of hours of historical data to fetch (default: 720 = 30 days)
            
        Returns:
            DataFrame with OHLCV data
        """
        return self.fetch_crypto_data_from_api(hours_back)
    
    def generate_288_predictions(self, price_data: pd.DataFrame) -> Dict:
        """
        Generate 288 predictions (24 hours of 5-minute intervals) using enhanced model.
        
        Args:
            price_data: Historical OHLC data
            
        Returns:
            Dictionary with predictions and summary statistics
        """
        try:
            if not self.predictor.is_loaded:
                if not self.predictor.load_latest_model():
                    raise Exception("Failed to load enhanced model")
            
            current_price = price_data['close'].iloc[-1]
            predictions = []
            
            print(f"🎯 Generating 288 enhanced predictions for {self.crypto_symbol}...")
            
            # Generate predictions for each of the 288 time points
            for i in range(288):
                # Create rolling window data for individual prediction
                # Use more recent data for predictions further in the future
                lookback_start = max(0, len(price_data) - 500 - i)
                rolling_data = price_data.iloc[lookback_start:].copy()
                
                # Make individual prediction
                individual_prediction = self.predictor.predict_next_period(rolling_data, current_price)
                
                # Apply time-varying adjustments
                time_factor = 1.0 + (i / 288) * 0.2  # Gradual increase over time
                
                adjusted_volatility = individual_prediction['predicted_volatility'] * time_factor
                adjusted_skewness = individual_prediction['predicted_skewness'] * (1.0 + (i / 288) * 0.1)
                adjusted_kurtosis = individual_prediction['predicted_kurtosis'] * (1.0 + (i / 288) * 0.15)
                
                prediction = {
                    'timestamp': datetime.now() + timedelta(minutes=5 * (i + 1)),
                    'predicted_volatility': adjusted_volatility,
                    'predicted_skewness': adjusted_skewness,
                    'predicted_kurtosis': adjusted_kurtosis,
                    'uncertainty_volatility': individual_prediction['uncertainty_volatility'],
                    'uncertainty_skewness': individual_prediction['uncertainty_skewness'],
                    'uncertainty_kurtosis': individual_prediction['uncertainty_kurtosis'],
                    'risk_level': individual_prediction['risk_level'],
                    'time_horizon_minutes': (i + 1) * 5
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
                'max_kurtosis': np.max(kurtoses),
                'total_predictions': len(predictions)
            }
            
            return {
                'predictions': predictions,
                'summary_stats': summary_stats,
                'current_price': current_price,
                'prediction_time': datetime.now(),
                'model_version': self.current_model_version,
                'crypto_symbol': self.crypto_symbol
            }
            
        except Exception as e:
            print(f"❌ Error generating enhanced predictions: {str(e)}")
            raise
    
    def save_predictions_to_database(self, prediction_result: Dict) -> str:
        """
        Save predictions to MongoDB database.
        
        Args:
            prediction_result: Prediction results dictionary
            
        Returns:
            Database ID of saved prediction
        """
        if not self.enable_database or not self.db_manager:
            return "database_disabled"
        
        try:
            # Prepare data for database
            db_data = {
                'crypto_symbol': self.crypto_symbol,
                'prediction_time': prediction_result['prediction_time'],
                'current_price': prediction_result['current_price'],
                'model_version': prediction_result['model_version'],
                'predictions': prediction_result['predictions'],
                'summary_stats': prediction_result['summary_stats'],
                'model_type': 'enhanced'
            }
            
            # Save to database
            prediction_id = self.db_manager.save_prediction(db_data)
            print(f"💾 Enhanced predictions saved to database: {prediction_id}")
            return prediction_id
            
        except Exception as e:
            print(f"❌ Error saving enhanced predictions to database: {str(e)}")
            return f"error_{str(e)}"
    
    def predict_and_save(self, price_data: pd.DataFrame, save_to_db: bool = True) -> Dict[str, float]:
        """
        Generate predictions and save to database.
        
        Args:
            price_data: Historical OHLC data
            save_to_db: Whether to save predictions to database
            
        Returns:
            Dictionary with summary statistics
        """
        try:
            # Generate predictions
            prediction_result = self.generate_288_predictions(price_data)
            
            # Save to database if enabled
            if save_to_db:
                self.save_predictions_to_database(prediction_result)
            
            # Update counters
            self.total_predictions_made += len(prediction_result['predictions'])
            
            # Print summary
            stats = prediction_result['summary_stats']
            print(f"📊 Enhanced prediction summary for {self.crypto_symbol}:")
            print(f"   Volatility: {stats['mean_volatility']:.6f} ± {stats['std_volatility']:.6f}")
            print(f"   Skewness: {stats['mean_skewness']:.6f} ± {stats['std_skewness']:.6f}")
            print(f"   Kurtosis: {stats['mean_kurtosis']:.6f} ± {stats['std_kurtosis']:.6f}")
            print(f"   Total predictions: {stats['total_predictions']}")
            
            return stats
            
        except Exception as e:
            print(f"❌ Error in predict_and_save: {str(e)}")
            raise
    
    def save_training_data(self, price_data: pd.DataFrame) -> bool:
        """
        Save training data to database for future retraining.
        
        Args:
            price_data: Historical OHLC data
            
        Returns:
            True if successful, False otherwise
        """
        if not self.enable_database or not self.db_manager:
            return False
        
        try:
            # Prepare training data
            training_data = {
                'crypto_symbol': self.crypto_symbol,
                'timestamp': datetime.now(),
                'data_points': len(price_data),
                'start_time': price_data['timestamp'].iloc[0],
                'end_time': price_data['timestamp'].iloc[-1],
                'ohlcv_data': price_data.to_dict('records')
            }
            
            # Save to database
            self.db_manager.save_training_data(training_data)
            return True
            
        except Exception as e:
            print(f"❌ Error saving training data: {str(e)}")
            return False
    
    def check_retraining_conditions(self) -> bool:
        """
        Check if retraining conditions are met.
        
        Returns:
            True if retraining should be performed
        """
        if not self.enable_online_learning:
            return False
        
        current_time = datetime.now()
        
        # Check time interval
        if self.last_retrain_time:
            hours_since_retrain = (current_time - self.last_retrain_time).total_seconds() / 3600
            if hours_since_retrain < self.retrain_interval_hours:
                return False
        
        # Check if we have enough new data
        if self.db_manager:
            try:
                recent_data_count = self.db_manager.get_recent_training_data_count(hours=24)
                if recent_data_count < self.min_new_data_points:
                    print(f"⚠️ Insufficient new data for retraining: {recent_data_count} < {self.min_new_data_points}")
                    return False
            except Exception as e:
                print(f"⚠️ Error checking training data: {str(e)}")
                return False
        
        return True
    
    def perform_retraining(self) -> bool:
        """
        Perform model retraining with recent data.
        
        Returns:
            True if retraining was successful
        """
        if not self.enable_online_learning:
            return False
        
        try:
            print(f"🔄 Starting enhanced model retraining for {self.crypto_symbol}...")
            
            # Fetch recent data for retraining
            recent_data = self.fetch_crypto_data_from_api(hours_back=168)  # 1 week
            
            if len(recent_data) < 1000:
                print(f"⚠️ Insufficient data for retraining: {len(recent_data)} < 1000")
                return False
            
            # Save data to CSV for training
            temp_csv_path = f"temp_{self.crypto_symbol}_retrain.csv"
            recent_data.to_csv(temp_csv_path, index=False)
            
            # Perform retraining
            training_result = self.trainer.train(temp_csv_path)
            
            # Clean up temp file
            if os.path.exists(temp_csv_path):
                os.remove(temp_csv_path)
            
            if training_result:
                self.last_retrain_time = datetime.now()
                self.current_model_version = self._get_model_version()
                print(f"✅ Enhanced model retraining completed for {self.crypto_symbol}")
                return True
            else:
                print(f"❌ Enhanced model retraining failed for {self.crypto_symbol}")
                return False
                
        except Exception as e:
            print(f"❌ Error during enhanced retraining: {str(e)}")
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
                print(f"❌ Background retraining error: {str(e)}")
                self.is_retraining = False
    
    def _start_background_retraining_thread(self):
        """Start background retraining thread."""
        if self.retraining_thread is None or not self.retraining_thread.is_alive():
            self.retraining_thread = threading.Thread(target=self._background_retraining_worker, daemon=True)
            self.retraining_thread.start()
            print(f"🔄 Background retraining thread started for {self.crypto_symbol}")
    
    def _stop_background_retraining_thread(self):
        """Stop background retraining thread."""
        if self.retraining_thread and self.retraining_thread.is_alive():
            self.retraining_queue.put("STOP")
            self.retraining_thread.join(timeout=5)
            print(f"🔄 Background retraining thread stopped for {self.crypto_symbol}")
    
    def _trigger_background_retraining(self):
        """Trigger background retraining."""
        if self.enable_online_learning and not self.is_retraining:
            self.retraining_queue.put("RETRAIN")
            print(f"🔄 Background retraining triggered for {self.crypto_symbol}")
    
    def _check_model_update(self):
        """Check if model has been updated and reload if necessary."""
        new_version = self._get_model_version()
        if new_version != self.current_model_version:
            print(f"🔄 Model updated: {self.current_model_version} → {new_version}")
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
            print(f"\n🔄 Enhanced prediction cycle #{self.prediction_cycles + 1} for {self.crypto_symbol}")
            
            # Fetch real-time data
            price_data = self.fetch_realtime_data()
            
            if len(price_data) < 100:
                print(f"❌ Insufficient data for prediction: {len(price_data)} < 100")
                return False
            
            # Validate data
            if not validate_crypto_data(price_data):
                print(f"❌ Invalid data for {self.crypto_symbol}")
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
            print(f"✅ Enhanced prediction cycle #{self.prediction_cycles} completed for {self.crypto_symbol}")
            
            return True
            
        except Exception as e:
            print(f"❌ Enhanced prediction cycle failed for {self.crypto_symbol}: {str(e)}")
            return False
    
    def start_continuous_prediction(self, interval_minutes: int = 5):
        """
        Start continuous prediction loop.
        
        Args:
            interval_minutes: Interval between predictions in minutes
        """
        print(f"🚀 Starting enhanced continuous prediction for {self.crypto_symbol}")
        print(f"   Interval: {interval_minutes} minutes")
        print(f"   Press Ctrl+C to stop")
        
        self.is_running = True
        
        # Start background retraining thread
        if self.enable_online_learning:
            self._start_background_retraining_thread()
        
        # Set up signal handler
        def signal_handler(signum, frame):
            print(f"\n🛑 Stopping enhanced continuous prediction for {self.crypto_symbol}")
            self.stop()
            sys.exit(0)
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
        try:
            while self.is_running:
                start_time = time.time()
                
                # Run prediction cycle
                success = self.run_prediction_cycle()
                
                if not success:
                    print(f"⚠️ Enhanced prediction cycle failed for {self.crypto_symbol}, retrying in {interval_minutes} minutes")
                
                # Wait for next cycle
                elapsed = time.time() - start_time
                sleep_time = max(0, (interval_minutes * 60) - elapsed)
                
                if sleep_time > 0:
                    print(f"⏰ Next enhanced prediction cycle in {sleep_time/60:.1f} minutes")
                    time.sleep(sleep_time)
                
        except KeyboardInterrupt:
            print(f"\n🛑 Enhanced continuous prediction interrupted for {self.crypto_symbol}")
        finally:
            self.stop()
    
    def stop(self):
        """Stop the continuous predictor."""
        print(f"🛑 Stopping enhanced continuous predictor for {self.crypto_symbol}")
        self.is_running = False
        
        if self.enable_online_learning:
            self._stop_background_retraining_thread()
        
        print(f"📊 Enhanced prediction summary for {self.crypto_symbol}:")
        print(f"   Total cycles: {self.prediction_cycles}")
        print(f"   Total predictions: {self.total_predictions_made}")
        print(f"   Model version: {self.current_model_version}")

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
        print(f"❌ Enhanced continuous prediction failed: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
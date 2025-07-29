#!/usr/bin/env python3

"""
Continuous Multi-Crypto Volatility Predictor
Runs every 5 minutes and generates 288 volatility predictions for the next 24 hours.
Supports multiple cryptocurrencies: BTC, ETH, XAU, SOL
Saves all predictions to MongoDB database.
Uses Pyth Network API for real-time cryptocurrency price data.
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

from predictor import RealTimeVolatilityPredictor
from database_manager import DatabaseManager
from trainer import CryptoVolatilityTrainer
from config import Config
from utils import format_prediction_output, validate_crypto_data

# Load environment variables
load_dotenv()

class ContinuousCryptoPredictor:
    """
    Continuous predictor that runs every 5 minutes generating 288 predictions each time.
    Supports multiple cryptocurrencies: BTC, ETH, XAU, SOL
    Uses Pyth Network API for real-time cryptocurrency price data.
    Self-contained with integrated prediction and database functionality.
    """
    
    def __init__(self, crypto_symbol: str = "BTC"):
        """
        Initialize the continuous predictor for a specific cryptocurrency.
        
        Args:
            crypto_symbol: Cryptocurrency symbol (BTC, ETH, XAU, SOL)
        """
        # Validate crypto symbol
        if crypto_symbol not in Config.SUPPORTED_CRYPTOS:
            raise ValueError(f"Unsupported crypto symbol: {crypto_symbol}. Supported: {list(Config.SUPPORTED_CRYPTOS.keys())}")
        
        self.crypto_symbol = crypto_symbol
        self.crypto_config = Config.SUPPORTED_CRYPTOS[crypto_symbol]
        self.symbol = self.crypto_config['pyth_symbol']
        
        self.api_base_url = "https://benchmarks.pyth.network/v1/shims/tradingview/history"
        
        # Initialize base predictor for volatility predictions
        self.predictor = RealTimeVolatilityPredictor(crypto_symbol=self.crypto_symbol)
        
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
        
        # ✅ THREADING FOR BACKGROUND RETRAINING
        self.retraining_thread = None
        self.retraining_lock = threading.Lock()
        self.is_retraining = False
        self.retraining_queue = queue.Queue()
        self.model_update_event = threading.Event()
        
        # Initialize trainer for continuous learning
        if self.enable_online_learning:
            try:
                config = Config()
                self.trainer = CryptoVolatilityTrainer(config, crypto_symbol=self.crypto_symbol)
            except Exception as e:
                print(f"⚠️ Training system initialization failed: {str(e)}")
                self.enable_online_learning = False
        
        # Note: Signal handlers are managed by the orchestrator in multi-threaded mode
        # Individual predictors don't set up signal handlers to avoid thread conflicts
    
    def _get_model_version(self) -> str:
        """Get current model version."""
        try:
            if hasattr(self.predictor, 'model') and self.predictor.model is not None:
                return f"{self.crypto_symbol}_model"
            else:
                return "unknown"
        except:
            return "unknown"
    
    def fetch_crypto_data_from_api(self, hours_back: int = 120) -> pd.DataFrame:
        """
        Fetch cryptocurrency price data from Pyth Network API.
        
        Args:
            hours_back: How many hours of historical data to fetch
            
        Returns:
            DataFrame with cryptocurrency price data
        """
        try:
            # Calculate time range
            current_time = int(time.time())
            start_time = current_time - (hours_back * 3600)  # hours_back hours ago
            
            # Construct API URL
            url = f"{self.api_base_url}?symbol={self.symbol}&resolution=5&from={start_time}&to={current_time}"
            
            # Make API request
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            
            # Check API response status
            if data.get('s') != 'ok':
                raise ValueError(f"API returned error status: {data.get('s')}")
            
            # Extract data arrays
            timestamps = data.get('t', [])
            opens = data.get('o', [])
            highs = data.get('h', [])
            lows = data.get('l', [])
            closes = data.get('c', [])
            
            if not timestamps or len(timestamps) == 0:
                raise ValueError("No data returned from API")
            
            # Create DataFrame
            df = pd.DataFrame({
                'timestamp': [datetime.fromtimestamp(ts) for ts in timestamps],
                'open': opens,
                'high': highs,
                'low': lows,
                'close': closes
            })
            
            # Sort by timestamp
            df = df.sort_values('timestamp').reset_index(drop=True)
            
            # Remove any NaN values
            df.dropna(inplace=True)
            
            if len(df) == 0:
                raise ValueError("No valid data after processing")
            
            return df
            
        except requests.exceptions.RequestException as e:
            print(f"❌ Network error fetching from Pyth API: {str(e)}")
            raise
        except ValueError as e:
            print(f"❌ Data error from Pyth API: {str(e)}")
            raise
        except Exception as e:
            print(f"❌ Unexpected error fetching {self.crypto_config['name']} data: {str(e)}")
            raise
    
    def get_current_crypto_price(self) -> Dict[str, any]:
        """
        Get the current cryptocurrency price from Pyth Network API.
        
        Returns:
            Dict with current price information
        """
        try:
            current_time = int(time.time())
            # Get just the latest data point
            url = f"{self.api_base_url}?symbol={self.symbol}&resolution=1&from={current_time-300}&to={current_time}"
            
            response = requests.get(url, timeout=5)
            response.raise_for_status()
            data = response.json()
            
            if data.get('s') != 'ok' or not data.get('c'):
                raise ValueError("No current price data available")
            
            # Get the latest price data
            latest_close = data['c'][-1]
            latest_timestamp = data['t'][-1]
            latest_open = data['o'][-1] if data.get('o') else latest_close
            latest_high = data['h'][-1] if data.get('h') else latest_close
            latest_low = data['l'][-1] if data.get('l') else latest_close
            
            result = {
                'price': float(latest_close),
                'timestamp': datetime.fromtimestamp(latest_timestamp),
                'open': float(latest_open),
                'high': float(latest_high),
                'low': float(latest_low),
                'source': 'Pyth Network'
            }
            
            return result
            
        except Exception as e:
            print(f"❌ Failed to get current {self.crypto_config['name']} price: {str(e)}")
            raise

    def fetch_realtime_data(self, hours_back: int = 120) -> pd.DataFrame:
        """
        Fetch real-time cryptocurrency data using Pyth Network API.
        
        Args:
            hours_back: Hours of historical data to fetch
            
        Returns:
            DataFrame with cryptocurrency price data
        """
        return self.fetch_crypto_data_from_api(hours_back)
    
    def generate_288_predictions(self, price_data: pd.DataFrame) -> Dict:
        """
        Generate 288 volatility predictions for the next 24 hours with varying kurtosis and skewness.
        
        Args:
            price_data: Historical price data
            
        Returns:
            Dict with all 288 predictions
        """
        try:
            # Get base prediction for reference
            base_prediction = self.predict_and_save(price_data, save_to_db=False)
            
            # Generate time points for next 24 hours (288 × 5-minute intervals)
            start_time = pd.to_datetime(price_data['timestamp'].iloc[-1])
            predictions = []
            
            # Get base values for time-varying patterns
            base_volatility = base_prediction['predicted_volatility']
            base_skewness = base_prediction['predicted_skewness']
            base_kurtosis = base_prediction['predicted_kurtosis']
            
            for i in range(288):  # 288 = 24 hours × 12 intervals per hour
                future_time = start_time + timedelta(minutes=5 * (i + 1))
                hour_utc = future_time.hour
                
                # Create rolling window for this time point prediction
                # Use different window sizes to simulate forward-looking predictions
                window_offset = min(i, 72)  # Max 6 hours offset for rolling window
                rolling_data = price_data.iloc[-(144 + window_offset):-window_offset] if window_offset > 0 else price_data
                
                # Make individual prediction for this time point
                try:
                    individual_prediction = self.predictor.predict_next_period(rolling_data)
                    predicted_volatility = individual_prediction['predicted_volatility']
                    predicted_skewness = individual_prediction['predicted_skewness']
                    predicted_kurtosis = individual_prediction['predicted_kurtosis']
                except Exception as e:
                    # Fallback to base prediction if individual prediction fails
                    predicted_volatility = base_volatility
                    predicted_skewness = base_skewness
                    predicted_kurtosis = base_kurtosis
                
                # Calculate time-varying multipliers based on market patterns
                # US trading hours (14:30-21:00 UTC = 9:30 AM-4:00 PM EST)
                if 14 <= hour_utc <= 21:
                    vol_multiplier = 1.3  # Higher volatility during US market hours
                    skew_multiplier = 1.2  # More pronounced skewness during active trading
                    kurt_multiplier = 1.4  # Higher kurtosis (fat tails) during active hours
                elif 22 <= hour_utc <= 2:  # Late US/early Asian
                    vol_multiplier = 1.1
                    skew_multiplier = 1.1
                    kurt_multiplier = 1.2
                elif 3 <= hour_utc <= 9:  # Asian trading hours
                    vol_multiplier = 0.9
                    skew_multiplier = 0.9
                    kurt_multiplier = 0.8
                else:  # Low activity hours
                    vol_multiplier = 0.7
                    skew_multiplier = 0.8
                    kurt_multiplier = 0.6
                
                # Weekend effect
                if future_time.weekday() >= 5:  # Saturday, Sunday
                    vol_multiplier *= 0.6
                    skew_multiplier *= 0.7
                    kurt_multiplier *= 0.5
                
                # Add realistic variation with different patterns for each metric
                hourly_variation_vol = 1.0 + 0.15 * np.sin(2 * np.pi * hour_utc / 24)
                hourly_variation_skew = 1.0 + 0.1 * np.sin(2 * np.pi * (hour_utc + 6) / 24)  # Phase shift
                hourly_variation_kurt = 1.0 + 0.2 * np.sin(2 * np.pi * (hour_utc + 12) / 24)  # Different phase
                
                # Add noise with different characteristics for each metric
                vol_noise = np.random.normal(1.0, 0.05)  # 5% random variation
                skew_noise = np.random.normal(1.0, 0.08)  # 8% random variation
                kurt_noise = np.random.normal(1.0, 0.12)  # 12% random variation
                
                # Calculate final multipliers
                final_vol_multiplier = vol_multiplier * hourly_variation_vol * vol_noise
                final_skew_multiplier = skew_multiplier * hourly_variation_skew * skew_noise
                final_kurt_multiplier = kurt_multiplier * hourly_variation_kurt * kurt_noise
                
                # Apply multipliers to predictions
                adjusted_volatility = predicted_volatility * final_vol_multiplier
                adjusted_skewness = predicted_skewness * final_skew_multiplier
                adjusted_kurtosis = predicted_kurtosis * final_kurt_multiplier
                
                # Create prediction for this time point with all required database fields
                prediction = {
                    'sequence_number': i + 1,
                    'timestamp': future_time.isoformat(),
                    'minutes_ahead': (i + 1) * 5,
                    'predicted_volatility': adjusted_volatility,
                    'predicted_skewness': adjusted_skewness,
                    'predicted_kurtosis': adjusted_kurtosis,
                    'volatility_annualized': adjusted_volatility * np.sqrt(365 * 24 * 12),  # Annualized volatility
                    'volatility_multiplier': final_vol_multiplier,
                    'skewness_multiplier': final_skew_multiplier,
                    'kurtosis_multiplier': final_kurt_multiplier,
                    'hour_utc': hour_utc,
                    'is_us_trading_hours': 14 <= hour_utc <= 21,
                    'is_weekend': future_time.weekday() >= 5,
                    'current_price': base_prediction['current_price'],
                    'confidence_interval_lower': base_prediction['current_price'] * (1 - 2 * adjusted_volatility),
                    'confidence_interval_upper': base_prediction['current_price'] * (1 + 2 * adjusted_volatility),
                    'market_regime': base_prediction.get('market_regime', 'unknown'),
                    'risk_assessment': base_prediction.get('risk_assessment', 'unknown'),
                    'prediction_period': '5_minutes',
                    'data_timestamp': start_time.isoformat(),
                    'model_version': self.current_model_version,
                    'prediction_type': 'continuous_5min_varying'
                }
                
                # Extract predictions
                volatility = float(prediction['predicted_volatility'])
                skewness = float(prediction['predicted_skewness'])
                kurtosis = float(prediction['predicted_kurtosis'])
                
                # Apply validation bounds to prevent extreme predictions
                volatility = max(min(volatility, 0.1), 0.001)  # 0.1% to 10%
                skewness = max(min(skewness, 2.0), -2.0)       # -2 to +2
                kurtosis = max(min(kurtosis, 10.0), -1.0)      # -1 to +10 (excess kurtosis) - more reasonable bounds
                
                # Update prediction with validated values
                prediction['predicted_volatility'] = volatility
                prediction['predicted_skewness'] = skewness
                prediction['predicted_kurtosis'] = kurtosis
                
                predictions.append(prediction)
            
            # Calculate summary statistics for all metrics
            volatilities = [p['predicted_volatility'] for p in predictions]
            skewnesses = [p['predicted_skewness'] for p in predictions]
            kurtoses = [p['predicted_kurtosis'] for p in predictions]
            annualized_volatilities = [p['volatility_annualized'] for p in predictions]
            
            result = {
                'prediction_timestamp': datetime.utcnow().isoformat(),
                'data_timestamp': start_time.isoformat(),
                'current_price': base_prediction['current_price'],
                'base_prediction': base_prediction,
                'predictions_count': len(predictions),
                'predictions': predictions,
                'summary_stats': {
                    'volatility': {
                        'min': min(volatilities),
                        'max': max(volatilities),
                        'mean': np.mean(volatilities),
                        'std': np.std(volatilities),
                        'range': max(volatilities) - min(volatilities)
                    },
                    'skewness': {
                        'min': min(skewnesses),
                        'max': max(skewnesses),
                        'mean': np.mean(skewnesses),
                        'std': np.std(skewnesses),
                        'range': max(skewnesses) - min(skewnesses)
                    },
                    'kurtosis': {
                        'min': min(kurtoses),
                        'max': max(kurtoses),
                        'mean': np.mean(kurtoses),
                        'std': np.std(kurtoses),
                        'range': max(kurtoses) - min(kurtoses)
                    },
                    'volatility_annualized': {
                        'min': min(annualized_volatilities),
                        'max': max(annualized_volatilities),
                        'mean': np.mean(annualized_volatilities)
                    }
                }
            }
            
            return result
            
        except Exception as e:
            print(f"❌ Failed to generate 288 predictions: {str(e)}")
            raise
    
    def save_predictions_to_database(self, prediction_result: Dict) -> str:
        """
        Save ONE record containing all 288 predictions to database.
        
        Args:
            prediction_result: Result from generate_288_predictions
            
        Returns:
            Batch ID for the saved prediction record
        """
        if not self.enable_database:
            return None
        
        try:
            batch_id = f"continuous_{int(time.time())}"
            
            # Create single record containing all 288 predictions
            prediction_batch_record = {
                'batch_id': batch_id,
                'prediction_type': 'continuous_batch',
                'prediction_timestamp': prediction_result['prediction_timestamp'],
                'data_timestamp': prediction_result['data_timestamp'],
                'current_price': prediction_result['current_price'],
                'predictions_count': prediction_result['predictions_count'],
                'summary_stats': prediction_result['summary_stats'],
                'model_version': prediction_result['predictions'][0].get('model_version', 'unknown'),
                'predictions': prediction_result['predictions'],  # All 288 predictions as array
                'source': 'Pyth Network',
                'interval_minutes': 5,
                'prediction_horizon_hours': 24
            }
            
            # Save single record to database
            self.db_manager.save_prediction(
                prediction_batch_record, 
                prediction_batch_record.get('model_version', 'unknown')
            )
            
            return batch_id
            
        except Exception as e:
            print(f"❌ Failed to save to database: {str(e)}")
            return None
    
    def predict_and_save(self, price_data: pd.DataFrame, save_to_db: bool = True) -> Dict[str, float]:
        """
        Make prediction and optionally save to database.
        
        Args:
            price_data: Historical price data
            save_to_db: Whether to save prediction to database
            
        Returns:
            Dict: Prediction results
        """
        try:
            # Make prediction using base predictor
            prediction = self.predictor.predict_next_period(price_data)
            
            # Add model version and metadata
            prediction['model_version'] = self.current_model_version
            prediction['prediction_id'] = f"{self.current_model_version}_{int(time.time())}"
            prediction['source'] = 'Pyth Network'
            
            # Save to database if enabled
            if save_to_db and self.enable_database and self.db_manager:
                try:
                    prediction_id = self.db_manager.save_prediction(
                        prediction, self.current_model_version
                    )
                    prediction['database_id'] = prediction_id
                except Exception as e:
                    print(f"⚠️ Failed to save prediction to database: {str(e)}")
            
            return prediction
            
        except Exception as e:
            print(f"❌ Prediction failed: {str(e)}")
            raise
    
    def save_training_data(self, price_data: pd.DataFrame) -> bool:
        """
        Save real-time price data to database for future training.
        
        Args:
            price_data: Real-time cryptocurrency price data
            
        Returns:
            bool: Success status
        """
        if not self.enable_database or not self.db_manager:
            return False
            
        try:
            # Save more data points for better retraining
            # Save the last 200 data points instead of just 50
            recent_data = price_data.tail(200)  # Save last 200 data points
            self.db_manager.save_training_data(
                recent_data, 
                data_source="realtime_continuous"
            )
            return True
        except Exception as e:
            print(f"⚠️ Failed to save training data: {str(e)}")
            return False
    
    def check_retraining_conditions(self) -> bool:
        """
        Check if model should be retrained based on time and data conditions.
        
        Returns:
            bool: True if retraining is needed
        """
        if not self.enable_online_learning:
            return False
            
        current_time = datetime.utcnow()
        
        # Check time-based condition
        if self.last_retrain_time is None:
            time_based = True
        else:
            hours_since_retrain = (current_time - self.last_retrain_time).total_seconds() / 3600
            time_based = hours_since_retrain >= self.retrain_interval_hours
        
        # Check data-based condition
        data_based = False
        if self.enable_database and self.db_manager:
            try:
                # Get training data added since last retrain
                cutoff_time = self.last_retrain_time or (current_time - timedelta(hours=self.retrain_interval_hours))
                recent_data = self.db_manager.get_training_data_for_update(
                    hours=int((current_time - cutoff_time).total_seconds() / 3600)
                )
                data_based = len(recent_data) >= self.min_new_data_points
            except Exception as e:
                print(f"⚠️ Error checking training data: {str(e)}")
        
        should_retrain = time_based or data_based
        
        return should_retrain
    
    def perform_retraining(self) -> bool:
        """
        Trigger background retraining (non-blocking).
        
        Returns:
            bool: True if retraining was triggered, False otherwise
        """
        if not self.enable_online_learning:
            return False
        
        # Check if already retraining
        with self.retraining_lock:
            if self.is_retraining:
                return False
        
        # Trigger background retraining
        self._trigger_background_retraining()
        return True
    
    def _background_retraining_worker(self):
        """
        Background worker thread for model retraining.
        Runs independently without blocking prediction cycles.
        """
        while self.is_running:
            try:
                # Wait for retraining signal from queue
                retrain_request = self.retraining_queue.get(timeout=60)  # 1 minute timeout
                
                if retrain_request == "STOP":
                    break
                
                # Set retraining flag
                with self.retraining_lock:
                    self.is_retraining = True
                
                # Perform retraining
                retrain_success = self._perform_retraining_internal()
                
                if retrain_success:
                    # Signal model update
                    self.model_update_event.set()
                
                # Clear retraining flag
                with self.retraining_lock:
                    self.is_retraining = False
                
                # Mark task as done
                self.retraining_queue.task_done()
                
            except queue.Empty:
                # No retraining request, continue waiting
                continue
            except Exception as e:
                print(f"❌ Background retraining worker error: {str(e)}")
                with self.retraining_lock:
                    self.is_retraining = False
                self.retraining_queue.task_done()
    
    def _perform_retraining_internal(self) -> bool:
        """
        Internal retraining method used by background worker.
        
        Returns:
            bool: Success status
        """
        if not self.enable_online_learning or not hasattr(self, 'trainer'):
            return False
            
        try:
            start_time = datetime.utcnow()
            
            # Get recent training data with fallback to all data
            training_data = self.db_manager.get_training_data_for_update(
                hours=self.retrain_interval_hours * 2,  # Get extra data for better training
                fallback_to_all=True  # Fallback to all available data if recent data is insufficient
            )
            
            if len(training_data) == 0:
                print("❌ No training data available for retraining")
                return False
            
            # More flexible data requirements for retraining
            min_data_points = max(20, self.min_new_data_points // 4)  # Much lower threshold for retraining
            
            if len(training_data) < min_data_points:
                print(f"⚠️ Limited training data: {len(training_data)} < {min_data_points}")
            
            # Additional check for minimum data requirements - reduced from 50 to 5
            if len(training_data) < 5:  # Reduced from 20 to 5 for retraining with limited data
                print(f"⚠️ Very limited training data: {len(training_data)} < 5 minimum for retraining")
                return False
            
            # Save training data to temporary CSV for trainer
            import tempfile
            import os as temp_os
            temp_csv = None
            try:
                # Create temporary CSV file
                with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
                    temp_csv = f.name
                    training_data.to_csv(temp_csv, index=False)
                
                # Use more flexible retraining parameters
                # If we have limited data, use all available data instead of just 30 days
                if len(training_data) < 1000:
                    days_back = 60  # Use more historical data for small datasets
                else:
                    days_back = 30  # Use recent data for larger datasets
                
                # Perform retraining with recent data
                training_results = self.trainer.retrain_with_current_data(temp_csv, days_back=days_back)
                success = training_results is not None and training_results.get('success', False)
                
            finally:
                # Clean up temporary file
                if temp_csv and temp_os.path.exists(temp_csv):
                    temp_os.unlink(temp_csv)
            
            if success:
                # Update model version and reload predictor
                old_version = self.current_model_version
                self.current_model_version = f"{self.crypto_symbol}_model"
                self.last_retrain_time = datetime.utcnow()
                
                # Reload the predictor with new model
                try:
                    self.predictor = RealTimeVolatilityPredictor(crypto_symbol=self.crypto_symbol)
                    print(f"✅ Retraining completed! Model updated.")
                    return True
                except Exception as e:
                    print(f"⚠️ Failed to reload predictor: {str(e)}")
                    return False
            else:
                if training_results and 'error' in training_results:
                    print(f"❌ Retraining failed: {training_results['error']}")
                else:
                    print(f"❌ Retraining failed")
                return False
                
        except Exception as e:
            print(f"❌ Error during retraining: {str(e)}")
            return False
    
    def _start_background_retraining_thread(self):
        """Start the background retraining thread."""
        if not self.enable_online_learning:
            return
        
        try:
            self.retraining_thread = threading.Thread(
                target=self._background_retraining_worker,
                name="BackgroundRetraining",
                daemon=True
            )
            self.retraining_thread.start()
        except Exception as e:
            print(f"❌ Failed to start background retraining thread: {str(e)}")
    
    def _stop_background_retraining_thread(self):
        """Stop the background retraining thread."""
        if self.retraining_thread and self.retraining_thread.is_alive():
            try:
                # Send stop signal
                self.retraining_queue.put("STOP")
                # Wait for thread to finish (with timeout)
                self.retraining_thread.join(timeout=30)
            except Exception as e:
                print(f"⚠️ Error stopping background retraining thread: {str(e)}")
    
    def _trigger_background_retraining(self):
        """
        Trigger background retraining without blocking.
        """
        if not self.enable_online_learning:
            return
        
        # Check if already retraining
        with self.retraining_lock:
            if self.is_retraining:
                return
        
        # Send retraining request to background thread
        try:
            self.retraining_queue.put("RETRAIN", timeout=1)
        except queue.Full:
            pass
    
    def _check_model_update(self):
        """
        Check if model has been updated by background retraining.
        """
        if self.model_update_event.is_set():
            self.model_update_event.clear()
            return True
        return False
    
    def run_prediction_cycle(self) -> bool:
        """
        Run one complete prediction cycle.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            cycle_start = time.time()
            self.prediction_cycles += 1
            
            # Check if model has been updated by background retraining
            if self._check_model_update():
                print(f"🔄 Model updated - Cycle #{self.prediction_cycles}")
            
            # Fetch real-time data
            price_data = self.fetch_realtime_data()
            
            # Save real-time data for training (continuous learning)
            if self.enable_online_learning:
                self.save_training_data(price_data)
            
            # Check if model retraining is needed (non-blocking)
            if self.enable_online_learning and self.check_retraining_conditions():
                retrain_triggered = self.perform_retraining()
                if retrain_triggered:
                    print(f"🧠 Background retraining initiated")
            
            # Generate 288 predictions
            prediction_result = self.generate_288_predictions(price_data)
            
            # Save to database
            batch_id = None
            if self.enable_database:
                batch_id = self.save_predictions_to_database(prediction_result)
            
            # Update counters
            self.total_predictions_made += prediction_result['predictions_count']
            
            # Display essential summary
            stats = prediction_result['summary_stats']
            print(f"⏰ {datetime.utcnow().strftime('%H:%M:%S')} | "
                  f"${prediction_result['current_price']:,.0f} | "
                  f"Vol: {stats['volatility']['mean']:.4f} | "
                  f"Range: {stats['volatility']['min']:.4f}-{stats['volatility']['max']:.4f}")
            
            # Show retraining status only if relevant
            if self.enable_online_learning:
                with self.retraining_lock:
                    if self.is_retraining:
                        print(f"   🔄 Retraining in progress...")
            
            return True
            
        except Exception as e:
            print(f"❌ Cycle #{self.prediction_cycles} failed: {str(e)}")
            return False
    
    def start_continuous_prediction(self, interval_minutes: int = 5):
        """
        Start continuous prediction loop.
        
        Args:
            interval_minutes: How often to make predictions (should be 5 for your use case)
        """
        print(f"🚀 Starting {self.crypto_config['name']} prediction | "
              f"Interval: {interval_minutes}min | "
              f"DB: {'On' if self.enable_database else 'Off'} | "
              f"Learning: {'On' if self.enable_online_learning else 'Off'}")
        
        self.is_running = True
        
        # Start background retraining thread
        if self.enable_online_learning:
            self._start_background_retraining_thread()
        
        try:
            while self.is_running:
                cycle_success = self.run_prediction_cycle()
                
                if not cycle_success:
                    print(f"⚠️ Cycle failed, retrying in {interval_minutes} minutes...")
                
                # Sleep until next cycle
                # Sleep in small intervals to allow for graceful shutdown
                sleep_seconds = interval_minutes * 60
                for _ in range(sleep_seconds):
                    if not self.is_running:
                        break
                    time.sleep(1)
                    
        except KeyboardInterrupt:
            print(f"\n🛑 Interrupted by user")
        finally:
            self.stop()
    
    def stop(self):
        """Stop the continuous prediction."""
        if not self.is_running:
            return
        
        print(f"\n🛑 Stopping continuous prediction...")
        self.is_running = False
        
        # Stop background retraining thread
        if self.enable_online_learning:
            self._stop_background_retraining_thread()
        
        # Close database connection if enabled
        if self.db_manager:
            try:
                self.db_manager.close()
            except:
                pass
        
        # Final statistics
        print(f"\n📊 Final Stats: {self.prediction_cycles} cycles, {self.total_predictions_made:,} predictions")


def main():
    """Main function to run continuous prediction supporting multiple cryptocurrencies."""
    import sys
    
    # Get crypto symbol from command line argument, default to BTC
    crypto_symbol = sys.argv[1].upper() if len(sys.argv) > 1 else 'BTC'
    
    # Validate crypto symbol
    if crypto_symbol not in Config.SUPPORTED_CRYPTOS:
        print(f"❌ Unsupported cryptocurrency: {crypto_symbol}")
        print(f"Supported cryptocurrencies: {list(Config.SUPPORTED_CRYPTOS.keys())}")
        print("Usage: python continuous_predictor.py [crypto_symbol]")
        print("Example: python continuous_predictor.py BTC")
        print("Example: python continuous_predictor.py ETH")
        return
    
    crypto_config = Config.SUPPORTED_CRYPTOS[crypto_symbol]
    
    print("🚀 Multi-Crypto Volatility - Continuous Prediction Mode")
    print(f"🎯 Target: {crypto_config['name']} ({crypto_symbol})")
    
    try:
        # Initialize continuous predictor
        predictor = ContinuousCryptoPredictor(crypto_symbol=crypto_symbol)
        
        # Start continuous prediction (every 5 minutes)
        predictor.start_continuous_prediction(interval_minutes=5)
        
    except KeyboardInterrupt:
        print("\n🛑 Interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
    finally:
        print("\n🔚 Continuous predictor terminated")


if __name__ == "__main__":
    main() 
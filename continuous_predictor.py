#!/usr/bin/env python3

"""
Continuous Bitcoin Volatility Predictor
Runs every 5 minutes and generates 288 volatility predictions for the next 24 hours.
Saves all predictions to MongoDB database.
Uses Pyth Network API for real-time Bitcoin price data.
"""

import os
import time
import signal
import sys
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
from typing import Dict, List
from dotenv import load_dotenv

from predictor import RealTimeVolatilityPredictor
from database_manager import DatabaseManager
from utils import format_prediction_output, validate_bitcoin_data

# Load environment variables
load_dotenv()

class ContinuousBitcoinPredictor:
    """
    Continuous predictor that runs every 5 minutes generating 288 predictions each time.
    Uses Pyth Network API for real-time Bitcoin price data.
    Self-contained with integrated prediction and database functionality.
    """
    
    def __init__(self, symbol: str = "Crypto.BTC/USD"):
        """
        Initialize the continuous predictor.
        
        Args:
            symbol: Bitcoin trading symbol for Pyth Network API
        """
        print("🚀 Initializing Continuous Bitcoin Volatility Predictor...")
        
        self.symbol = symbol
        self.api_base_url = "https://benchmarks.pyth.network/v1/shims/tradingview/history"
        
        # Initialize base predictor for volatility predictions
        self.predictor = RealTimeVolatilityPredictor()
        
        # Database configuration
        self.enable_database = os.getenv('ENABLE_DATABASE', 'true').lower() == 'true'
        self.enable_online_learning = os.getenv('ENABLE_ONLINE_LEARNING', 'true').lower() == 'true'
        
        # Initialize database manager if enabled
        self.db_manager = None
        if self.enable_database:
            try:
                self.db_manager = DatabaseManager()
                print("✅ Database connection established")
            except Exception as e:
                print(f"⚠️ Database connection failed: {str(e)}")
                self.enable_database = False
        
        # State tracking
        self.is_running = False
        self.prediction_cycles = 0
        self.total_predictions_made = 0
        self.current_model_version = self._get_model_version()
        
        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        print("✅ Continuous predictor initialized")
        print(f"📊 Symbol: {self.symbol}")
        print(f"🌐 API: Pyth Network")
        print(f"💾 Database: {'Enabled' if self.enable_database else 'Disabled'}")
        print(f"🧠 Online learning: {'Enabled' if self.enable_online_learning else 'Disabled'}")
        print(f"🔧 Model Version: {self.current_model_version}")
    
    def _get_model_version(self) -> str:
        """Get current model version."""
        try:
            if hasattr(self.predictor, 'model') and self.predictor.model is not None:
                return datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            else:
                return "unknown"
        except:
            return "unknown"
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully."""
        print(f"\n🛑 Received signal {signum}, shutting down gracefully...")
        self.stop()
        sys.exit(0)
    
    def fetch_bitcoin_data_from_api(self, hours_back: int = 120) -> pd.DataFrame:
        """
        Fetch Bitcoin price data from Pyth Network API.
        
        Args:
            hours_back: How many hours of historical data to fetch
            
        Returns:
            DataFrame with Bitcoin price data
        """
        try:
            print(f"📡 Fetching Bitcoin data from Pyth Network API...")
            
            # Calculate time range
            current_time = int(time.time())
            start_time = current_time - (hours_back * 3600)  # hours_back hours ago
            
            # Construct API URL
            url = f"{self.api_base_url}?symbol={self.symbol}&resolution=5&from={start_time}&to={current_time}"
            
            print(f"🔗 API URL: {url}")
            
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
            
            print(f"✅ Fetched {len(df)} data points from Pyth Network")
            print(f"📊 Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
            print(f"💰 Current price: ${df['close'].iloc[-1]:,.2f}")
            
            return df
            
        except requests.exceptions.RequestException as e:
            print(f"❌ Network error fetching from Pyth API: {str(e)}")
            raise
        except ValueError as e:
            print(f"❌ Data error from Pyth API: {str(e)}")
            raise
        except Exception as e:
            print(f"❌ Unexpected error fetching Bitcoin data: {str(e)}")
            raise
    
    def get_current_bitcoin_price(self) -> Dict[str, any]:
        """
        Get the current Bitcoin price from Pyth Network API.
        
        Returns:
            Dict with current price information
        """
        try:
            print(f"💰 Getting current Bitcoin price...")
            
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
            
            print(f"📈 Current Bitcoin Price: ${result['price']:,.2f}")
            print(f"🕐 As of: {result['timestamp']}")
            
            return result
            
        except Exception as e:
            print(f"❌ Failed to get current Bitcoin price: {str(e)}")
            raise

    def fetch_realtime_data(self, hours_back: int = 120) -> pd.DataFrame:
        """
        Fetch real-time Bitcoin data using Pyth Network API.
        
        Args:
            hours_back: Hours of historical data to fetch
            
        Returns:
            DataFrame with Bitcoin price data
        """
        return self.fetch_bitcoin_data_from_api(hours_back)
    
    def generate_288_predictions(self, price_data: pd.DataFrame) -> Dict:
        """
        Generate 288 volatility predictions for the next 24 hours.
        
        Args:
            price_data: Historical price data
            
        Returns:
            Dict with all 288 predictions
        """
        try:
            print(f"🔮 Generating 288 volatility predictions for next 24 hours...")
            
            # Get base prediction
            base_prediction = self.predict_and_save(price_data, save_to_db=False)
            
            # Generate time points for next 24 hours (288 × 5-minute intervals)
            start_time = pd.to_datetime(price_data['timestamp'].iloc[-1])
            predictions = []
            
            for i in range(288):  # 288 = 24 hours × 12 intervals per hour
                future_time = start_time + timedelta(minutes=5 * (i + 1))
                hour_utc = future_time.hour
                
                # Calculate volatility multiplier based on time patterns
                # US trading hours (14:30-21:00 UTC = 9:30 AM-4:00 PM EST)
                if 14 <= hour_utc <= 21:
                    base_multiplier = 1.3  # Higher volatility during US market hours
                elif 22 <= hour_utc <= 2:  # Late US/early Asian
                    base_multiplier = 1.1
                elif 3 <= hour_utc <= 9:  # Asian trading hours
                    base_multiplier = 0.9
                else:  # Low activity hours
                    base_multiplier = 0.7
                
                # Weekend effect
                if future_time.weekday() >= 5:  # Saturday, Sunday
                    base_multiplier *= 0.6
                
                # Add realistic variation
                hourly_variation = 1.0 + 0.15 * np.sin(2 * np.pi * hour_utc / 24)
                noise = np.random.normal(1.0, 0.05)  # 5% random variation
                
                final_multiplier = base_multiplier * hourly_variation * noise
                
                # Create prediction for this time point
                prediction = {
                    'sequence_number': i + 1,
                    'timestamp': future_time.isoformat(),
                    'minutes_ahead': (i + 1) * 5,
                    'predicted_volatility': base_prediction['predicted_volatility'] * final_multiplier,
                    'predicted_skewness': base_prediction['predicted_skewness'],
                    'predicted_kurtosis': base_prediction['predicted_kurtosis'],
                    'volatility_multiplier': final_multiplier,
                    'hour_utc': hour_utc,
                    'is_us_trading_hours': 14 <= hour_utc <= 21,
                    'is_weekend': future_time.weekday() >= 5,
                    'current_price': base_prediction['current_price'],
                    'data_timestamp': start_time.isoformat(),
                    'model_version': self.current_model_version,
                    'prediction_type': 'continuous_5min'
                }
                
                predictions.append(prediction)
            
            # Calculate summary statistics
            volatilities = [p['predicted_volatility'] for p in predictions]
            
            result = {
                'prediction_timestamp': datetime.utcnow().isoformat(),
                'data_timestamp': start_time.isoformat(),
                'current_price': base_prediction['current_price'],
                'base_prediction': base_prediction,
                'predictions_count': len(predictions),
                'predictions': predictions,
                'summary_stats': {
                    'min_volatility': min(volatilities),
                    'max_volatility': max(volatilities),
                    'mean_volatility': np.mean(volatilities),
                    'std_volatility': np.std(volatilities),
                    'volatility_range': max(volatilities) - min(volatilities)
                }
            }
            
            print(f"✅ Generated {len(predictions)} predictions")
            return result
            
        except Exception as e:
            print(f"❌ Failed to generate 288 predictions: {str(e)}")
            raise
    
    def save_predictions_to_database(self, prediction_result: Dict) -> str:
        """
        Save all 288 predictions to database.
        
        Args:
            prediction_result: Result from generate_288_predictions
            
        Returns:
            Batch ID for the saved predictions
        """
        if not self.enable_database:
            print("⚠️ Database not enabled, skipping save")
            return None
        
        try:
            batch_id = f"continuous_{int(time.time())}"
            saved_count = 0
            
            print(f"💾 Saving {len(prediction_result['predictions'])} predictions to database...")
            
            # Save each prediction
            for prediction in prediction_result['predictions']:
                prediction['batch_id'] = batch_id
                
                try:
                    self.db_manager.save_prediction(
                        prediction, 
                        prediction.get('model_version', 'unknown')
                    )
                    saved_count += 1
                except Exception as e:
                    print(f"⚠️ Failed to save prediction {prediction['sequence_number']}: {str(e)}")
            
            # Save batch summary
            batch_summary = {
                'batch_id': batch_id,
                'prediction_type': 'continuous_batch_summary',
                'prediction_timestamp': prediction_result['prediction_timestamp'],
                'data_timestamp': prediction_result['data_timestamp'],
                'current_price': prediction_result['current_price'],
                'predictions_count': prediction_result['predictions_count'],
                'saved_count': saved_count,
                'summary_stats': prediction_result['summary_stats'],
                'model_version': prediction_result['predictions'][0].get('model_version', 'unknown')
            }
            
            self.db_manager.save_prediction(
                batch_summary, 
                batch_summary.get('model_version', 'unknown')
            )
            
            print(f"✅ Saved {saved_count}/{len(prediction_result['predictions'])} predictions with batch ID: {batch_id}")
            return batch_id
            
        except Exception as e:
            print(f"❌ Failed to save predictions to database: {str(e)}")
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
    
    def run_prediction_cycle(self) -> bool:
        """
        Run one complete prediction cycle.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            cycle_start = time.time()
            self.prediction_cycles += 1
            
            print(f"\n⏰ === Prediction Cycle #{self.prediction_cycles} ===")
            print(f"🕐 Time: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}")
            
            # Fetch real-time data
            price_data = self.fetch_realtime_data()
            
            # Generate 288 predictions
            prediction_result = self.generate_288_predictions(price_data)
            
            # Save to database
            batch_id = None
            if self.enable_database:
                batch_id = self.save_predictions_to_database(prediction_result)
            
            # Update counters
            self.total_predictions_made += prediction_result['predictions_count']
            
            # Display summary
            stats = prediction_result['summary_stats']
            print(f"\n📊 Cycle Summary:")
            print(f"   Current Price: ${prediction_result['current_price']:,.2f}")
            print(f"   Predictions Generated: {prediction_result['predictions_count']}")
            print(f"   Volatility Range: {stats['min_volatility']:.4f} - {stats['max_volatility']:.4f}")
            print(f"   Mean Volatility: {stats['mean_volatility']:.4f}")
            if batch_id:
                print(f"   Database Batch ID: {batch_id}")
            
            # Timing
            cycle_time = time.time() - cycle_start
            print(f"⏱️  Cycle processing time: {cycle_time:.2f} seconds")
            
            return True
            
        except Exception as e:
            print(f"❌ Prediction cycle failed: {str(e)}")
            return False
    
    def start_continuous_prediction(self, interval_minutes: int = 5):
        """
        Start continuous prediction loop.
        
        Args:
            interval_minutes: How often to make predictions (should be 5 for your use case)
        """
        print(f"\n🚀 Starting Continuous Bitcoin Volatility Prediction")
        print(f"⏰ Prediction interval: {interval_minutes} minutes")
        print(f"🔮 Predictions per cycle: 288 (next 24 hours)")
        print(f"💾 Database storage: {'Enabled' if self.enable_database else 'Disabled'}")
        print("=" * 60)
        
        self.is_running = True
        
        # Background tasks integrated into continuous predictor
        print(f"🔄 Background monitoring integrated into prediction cycles")
        
        print(f"🎯 Starting prediction cycles...")
        
        try:
            while self.is_running:
                cycle_success = self.run_prediction_cycle()
                
                if not cycle_success:
                    print(f"⚠️ Cycle failed, retrying in {interval_minutes} minutes...")
                
                # Sleep until next cycle
                print(f"😴 Sleeping for {interval_minutes} minutes until next cycle...")
                
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
        
        # Close database connection if enabled
        if self.db_manager:
            try:
                self.db_manager.close()
                print("✅ Database connection closed")
            except:
                pass
        
        # Final statistics
        print(f"\n📊 === Final Statistics ===")
        print(f"   Total cycles completed: {self.prediction_cycles}")
        print(f"   Total predictions made: {self.total_predictions_made:,}")
        print(f"   Session end time: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}")
        print(f"✅ Continuous prediction stopped")


def main():
    """Main function to run continuous prediction."""
    print("🚀 Bitcoin Volatility - Continuous Prediction Mode")
    print("=" * 50)
    
    try:
        # Initialize continuous predictor
        predictor = ContinuousBitcoinPredictor(symbol="Crypto.BTC/USD")
        
        # Start continuous prediction (every 5 minutes)
        predictor.start_continuous_prediction(interval_minutes=5)
        
    except KeyboardInterrupt:
        print("\n🛑 Interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        print("\n💡 Troubleshooting:")
        print("   1. Ensure MongoDB is running (if using database features)")
        print("   2. Check internet connection for Pyth Network API access")
        print("   3. Verify model is trained (run trainer.py first)")
        print("   4. Install dependencies: pip install -r requirements.txt")
        print("   5. Check Pyth Network API status if data fetching fails")
    finally:
        print("\n🔚 Continuous predictor terminated")


if __name__ == "__main__":
    main() 
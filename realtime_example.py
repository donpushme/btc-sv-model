#!/usr/bin/env python3

"""
Real-world Example: Enhanced Bitcoin Volatility Predictor
Demonstrates production-ready usage with database integration and continuous learning.
"""

import os
import time
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import schedule
import signal
import sys

from realtime_predictor import EnhancedRealTimePredictor
from utils import format_prediction_output, validate_bitcoin_data
from config import Config

class ProductionBitcoinPredictor:
    """
    Production-ready Bitcoin volatility predictor with real-time data fetching.
    """
    
    def __init__(self):
        """Initialize the production predictor."""
        print("🚀 Initializing Production Bitcoin Predictor...")
        
        # Initialize enhanced predictor
        self.predictor = EnhancedRealTimePredictor()
        self.config = Config()
        
        # State tracking
        self.is_running = False
        self.prediction_count = 0
        self.last_data_fetch = None
        
        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        print("✅ Production predictor initialized")
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully."""
        print(f"\n🛑 Received signal {signum}, shutting down gracefully...")
        self.stop()
        sys.exit(0)
    
    def fetch_realtime_data(self, symbol: str = "BTC-USD", 
                          period: str = "5d", interval: str = "5m") -> pd.DataFrame:
        """
        Fetch real-time Bitcoin data from Yahoo Finance.
        
        Args:
            symbol: Trading symbol
            period: Data period
            interval: Data interval
            
        Returns:
            pd.DataFrame: Bitcoin price data
        """
        try:
            print(f"📡 Fetching real-time data for {symbol}...")
            
            # Download data
            btc = yf.download(symbol, period=period, interval=interval, progress=False)
            
            if btc.empty:
                raise ValueError("No data received from yfinance")
            
            # Process data
            btc.reset_index(inplace=True)
            
            # Handle different column formats
            if len(btc.columns) == 7:
                btc.columns = ['timestamp', 'open', 'high', 'low', 'close', 'adj_close', 'volume']
            elif len(btc.columns) == 6:
                btc.columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
            else:
                # Standardize column names
                column_mapping = {
                    'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close',
                    'Adj Close': 'adj_close', 'Volume': 'volume', 'Datetime': 'timestamp'
                }
                btc = btc.rename(columns=column_mapping)
                
                # Ensure timestamp column
                if 'timestamp' not in btc.columns:
                    btc.columns = ['timestamp'] + list(btc.columns[1:])
            
            # Select required columns
            required_cols = ['timestamp', 'open', 'close', 'high', 'low']
            btc = btc[required_cols].copy()
            
            # Validate data
            validation = validate_bitcoin_data(btc)
            if validation['errors']:
                print(f"⚠️ Data validation warnings: {validation['errors']}")
            
            self.last_data_fetch = datetime.utcnow()
            print(f"✅ Fetched {len(btc)} data points (latest: {btc['timestamp'].iloc[-1]})")
            
            return btc
            
        except Exception as e:
            print(f"❌ Failed to fetch real-time data: {str(e)}")
            raise
    
    def make_scheduled_prediction(self):
        """Make a scheduled prediction with real-time data."""
        try:
            print(f"\n⏰ Making scheduled prediction #{self.prediction_count + 1}")
            print(f"🕐 Time: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}")
            
            # Fetch latest data
            data = self.fetch_realtime_data()
            
            # Make prediction
            prediction = self.predictor.predict_and_save(data)
            
            # Log results
            self.prediction_count += 1
            print(f"\n🔮 Prediction Results:")
            print(format_prediction_output(prediction))
            
            # Additional logging
            if prediction.get('database_id'):
                print(f"💾 Saved to database with ID: {prediction['database_id']}")
            
            print(f"📊 Total predictions made: {self.prediction_count}")
            
            return prediction
            
        except Exception as e:
            print(f"❌ Scheduled prediction failed: {str(e)}")
            return None
    
    def start_realtime_monitoring(self, prediction_interval_minutes: int = 30):
        """
        Start real-time monitoring and prediction.
        
        Args:
            prediction_interval_minutes: How often to make predictions
        """
        if self.is_running:
            print("⚠️ Already running")
            return
        
        print(f"🚀 Starting real-time monitoring...")
        print(f"📊 Prediction interval: {prediction_interval_minutes} minutes")
        print(f"🗄️ Database enabled: {self.predictor.enable_database}")
        print(f"🧠 Online learning enabled: {self.predictor.enable_online_learning}")
        
        self.is_running = True
        
        # Schedule predictions
        schedule.every(prediction_interval_minutes).minutes.do(self.make_scheduled_prediction)
        
        # Start background tasks
        self.predictor.start_scheduled_tasks()
        
        # Make initial prediction
        print("\n🎯 Making initial prediction...")
        self.make_scheduled_prediction()
        
        # Main monitoring loop
        print(f"\n🔄 Starting monitoring loop...")
        print(f"💡 Press Ctrl+C to stop gracefully")
        
        try:
            while self.is_running:
                schedule.run_pending()
                time.sleep(30)  # Check every 30 seconds
                
                # Log status every hour
                if self.prediction_count > 0 and self.prediction_count % (60 // prediction_interval_minutes) == 0:
                    self._log_status()
                    
        except KeyboardInterrupt:
            print(f"\n🛑 Stopped by user")
        finally:
            self.stop()
    
    def _log_status(self):
        """Log system status."""
        try:
            print(f"\n📊 === System Status ===")
            status = self.predictor.get_system_status()
            
            print(f"🕐 Current time: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}")
            print(f"🔧 Model version: {status['model_version']}")
            print(f"📈 Predictions made: {self.prediction_count}")
            print(f"🧠 Online learning: {status['online_learning_enabled']}")
            print(f"📊 New data points: {status['new_data_count']}")
            print(f"🔄 Last retrain: {status['last_retrain']}")
            
            if status.get('database_stats'):
                db_stats = status['database_stats']
                print(f"💾 DB predictions: {db_stats.get('predictions_count', 0):,}")
                print(f"💾 DB size: {db_stats.get('database_size_mb', 0):.1f} MB")
            
            print(f"========================")
            
        except Exception as e:
            print(f"❌ Status logging failed: {str(e)}")
    
    def get_prediction_analytics(self, hours: int = 24) -> dict:
        """
        Get analytics on recent predictions.
        
        Args:
            hours: Hours to analyze
            
        Returns:
            dict: Analytics summary
        """
        try:
            if not self.predictor.enable_database:
                return {"error": "Database not enabled"}
            
            # Get recent predictions
            predictions_df = self.predictor.get_prediction_history(hours=hours)
            
            if predictions_df.empty:
                return {"message": f"No predictions in last {hours} hours"}
            
            # Calculate analytics
            analytics = {
                "period_hours": hours,
                "total_predictions": len(predictions_df),
                "average_volatility": float(predictions_df['predicted_volatility'].mean()),
                "volatility_std": float(predictions_df['predicted_volatility'].std()),
                "average_price": float(predictions_df['current_price'].mean()),
                "price_range": {
                    "min": float(predictions_df['current_price'].min()),
                    "max": float(predictions_df['current_price'].max())
                },
                "risk_distribution": predictions_df['risk_assessment'].value_counts().to_dict(),
                "regime_distribution": predictions_df['market_regime'].value_counts().to_dict(),
                "time_range": {
                    "start": predictions_df['data_timestamp'].min().isoformat(),
                    "end": predictions_df['data_timestamp'].max().isoformat()
                }
            }
            
            return analytics
            
        except Exception as e:
            return {"error": str(e)}
    
    def force_retrain(self):
        """Force model retraining."""
        print("🔄 Forcing model retraining...")
        if self.predictor.force_retrain():
            print("✅ Model retrained successfully")
        else:
            print("❌ Model retraining failed")
    
    def stop(self):
        """Stop the predictor gracefully."""
        if not self.is_running:
            return
        
        print("\n🛑 Stopping production predictor...")
        self.is_running = False
        
        # Clear scheduled tasks
        schedule.clear()
        
        # Stop predictor
        self.predictor.close()
        
        # Final status
        print(f"📊 Session summary:")
        print(f"   Total predictions: {self.prediction_count}")
        print(f"   Last data fetch: {self.last_data_fetch}")
        print(f"   Runtime: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}")
        
        print("✅ Shutdown complete")


def main():
    """Main function to run the production predictor."""
    print("🚀 Bitcoin Volatility Prediction - Production Mode")
    print("=" * 50)
    
    try:
        # Initialize production predictor
        production_predictor = ProductionBitcoinPredictor()
        
        # Configuration
        prediction_interval = 30  # minutes
        
        print(f"\n⚙️ Configuration:")
        print(f"   Prediction interval: {prediction_interval} minutes")
        print(f"   Database: {'Enabled' if production_predictor.predictor.enable_database else 'Disabled'}")
        print(f"   Online learning: {'Enabled' if production_predictor.predictor.enable_online_learning else 'Disabled'}")
        
        # Start real-time monitoring
        production_predictor.start_realtime_monitoring(
            prediction_interval_minutes=prediction_interval
        )
        
    except KeyboardInterrupt:
        print("\n🛑 Interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        print("\n💡 Troubleshooting:")
        print("   1. Ensure MongoDB is running (if using database features)")
        print("   2. Check internet connection for data fetching")
        print("   3. Verify model is trained (run trainer.py first)")
        print("   4. Check requirements.txt dependencies are installed")
    finally:
        print("\n🔚 Production predictor terminated")


def demo_with_analytics():
    """Demo with analytics dashboard."""
    print("📊 Analytics Demo")
    print("=" * 30)
    
    try:
        # Initialize predictor
        predictor = ProductionBitcoinPredictor()
        
        # Make a few predictions
        print("🔮 Making sample predictions...")
        for i in range(3):
            prediction = predictor.make_scheduled_prediction()
            if prediction:
                print(f"✅ Prediction {i+1} completed")
            time.sleep(2)  # Wait between predictions
        
        # Get analytics
        print("\n📊 Getting analytics...")
        analytics = predictor.get_prediction_analytics(hours=1)
        
        print("\n📈 Analytics Summary:")
        for key, value in analytics.items():
            if isinstance(value, dict):
                print(f"  {key}:")
                for sub_key, sub_value in value.items():
                    print(f"    {sub_key}: {sub_value}")
            else:
                print(f"  {key}: {value}")
        
        # Shutdown
        predictor.stop()
        
    except Exception as e:
        print(f"❌ Demo failed: {str(e)}")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "demo":
        demo_with_analytics()
    else:
        main() 
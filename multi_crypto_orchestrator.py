#!/usr/bin/env python3

"""
Multi-Crypto Orchestrator
Manages continuous prediction for multiple cryptocurrencies simultaneously.
Supports BTC, ETH, XAU, SOL with separate models and database tables.
"""

import os
import time
import signal
import sys
import threading
import queue
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from dotenv import load_dotenv

from continuous_predictor import ContinuousCryptoPredictor
from config import Config

# Load environment variables
load_dotenv()

class MultiCryptoOrchestrator:
    """
    Orchestrates continuous prediction for multiple cryptocurrencies.
    Each cryptocurrency runs in its own thread with separate models and database tables.
    """
    
    def __init__(self, crypto_symbols: List[str] = None):
        """
        Initialize the multi-crypto orchestrator.
        
        Args:
            crypto_symbols: List of cryptocurrency symbols to run (default: all supported)
        """
        if crypto_symbols is None:
            crypto_symbols = list(Config.SUPPORTED_CRYPTOS.keys())
        
        # Validate crypto symbols
        for symbol in crypto_symbols:
            if symbol not in Config.SUPPORTED_CRYPTOS:
                raise ValueError(f"Unsupported crypto symbol: {symbol}. Supported: {list(Config.SUPPORTED_CRYPTOS.keys())}")
        
        self.crypto_symbols = crypto_symbols
        self.predictors = {}
        self.threads = {}
        self.is_running = False
        
        # Threading
        self.thread_lock = threading.Lock()
        self.stop_event = threading.Event()
        
        # Statistics
        self.stats = {
            'start_time': None,
            'total_cycles': 0,
            'total_predictions': 0,
            'errors': []
        }
        
        print(f"🚀 Initializing Multi-Crypto Orchestrator...")
        print(f"📊 Supported cryptocurrencies: {', '.join(self.crypto_symbols)}")
        
        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        print("✅ Multi-Crypto Orchestrator initialized")
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully."""
        print(f"\n🛑 Received signal {signum}, shutting down gracefully...")
        self.stop()
        sys.exit(0)
    
    def _run_crypto_predictor(self, crypto_symbol: str):
        """
        Run continuous prediction for a specific cryptocurrency in a separate thread.
        
        Args:
            crypto_symbol: Cryptocurrency symbol to run
        """
        try:
            print(f"🔄 Starting {Config.SUPPORTED_CRYPTOS[crypto_symbol]['name']} ({crypto_symbol}) predictor thread...")
            
            # Check if model exists before starting
            import glob
            model_pattern = f"models/{crypto_symbol}_model_*.pth"
            model_files = glob.glob(model_pattern)
            
            if not model_files:
                raise Exception(f"No trained model found for {crypto_symbol}. Please train a model first using: python trainer.py {crypto_symbol}")
            
            print(f"✅ Found {len(model_files)} model(s) for {crypto_symbol}")
            
            # Initialize predictor for this crypto
            predictor = ContinuousCryptoPredictor(crypto_symbol=crypto_symbol)
            self.predictors[crypto_symbol] = predictor
            
            # Get prediction interval from environment
            interval_minutes = int(os.getenv('PREDICTION_INTERVAL_MINUTES', '5'))
            
            # Start continuous prediction (this will run until stopped)
            predictor.start_continuous_prediction(interval_minutes=interval_minutes)
            
        except Exception as e:
            error_msg = f"Error in {crypto_symbol} predictor: {str(e)}"
            print(f"❌ {error_msg}")
            with self.thread_lock:
                self.stats['errors'].append({
                    'crypto_symbol': crypto_symbol,
                    'error': str(e),
                    'timestamp': datetime.utcnow()
                })
            
            # Remove the failed predictor from the dictionary
            if crypto_symbol in self.predictors:
                del self.predictors[crypto_symbol]
    
    def start(self):
        """Start continuous prediction for all cryptocurrencies."""
        if self.is_running:
            print("⚠️ Orchestrator is already running")
            return
        
        print(f"\n🚀 Starting Multi-Crypto Continuous Prediction")
        print(f"⏰ Prediction interval: {os.getenv('PREDICTION_INTERVAL_MINUTES', '5')} minutes")
        print(f"🔮 Predictions per cycle: 288 (next 24 hours) per crypto")
        print(f"📊 Cryptocurrencies: {', '.join(self.crypto_symbols)}")
        print("=" * 60)
        
        self.is_running = True
        self.stats['start_time'] = datetime.utcnow()
        
        # Start a thread for each cryptocurrency
        for crypto_symbol in self.crypto_symbols:
            thread = threading.Thread(
                target=self._run_crypto_predictor,
                args=(crypto_symbol,),
                name=f"predictor_{crypto_symbol}",
                daemon=True
            )
            self.threads[crypto_symbol] = thread
            thread.start()
            
            # Small delay between thread starts to avoid overwhelming the system
            time.sleep(2)
        
        print(f"✅ All {len(self.crypto_symbols)} cryptocurrency predictors started")
        
        # Monitor threads
        try:
            while self.is_running and not self.stop_event.is_set():
                # Check if any threads have died
                dead_threads = []
                for crypto_symbol, thread in self.threads.items():
                    if not thread.is_alive():
                        dead_threads.append(crypto_symbol)
                        print(f"⚠️ Thread for {crypto_symbol} has died, restarting...")
                
                # Restart dead threads
                for crypto_symbol in dead_threads:
                    if self.is_running:
                        # Clean up any existing predictor
                        if crypto_symbol in self.predictors:
                            try:
                                self.predictors[crypto_symbol].stop()
                            except:
                                pass
                            del self.predictors[crypto_symbol]
                        
                        # Start new thread
                        thread = threading.Thread(
                            target=self._run_crypto_predictor,
                            args=(crypto_symbol,),
                            name=f"predictor_{crypto_symbol}",
                            daemon=True
                        )
                        self.threads[crypto_symbol] = thread
                        thread.start()
                        print(f"✅ Restarted thread for {crypto_symbol}")
                
                # Update statistics
                self._update_stats()
                
                # Sleep for a bit before checking again
                time.sleep(30)
                
        except KeyboardInterrupt:
            print(f"\n🛑 Interrupted by user")
        finally:
            self.stop()
    
    def _update_stats(self):
        """Update running statistics."""
        with self.thread_lock:
            total_cycles = 0
            total_predictions = 0
            
            for crypto_symbol, predictor in self.predictors.items():
                if predictor and hasattr(predictor, 'prediction_cycles'):
                    total_cycles += predictor.prediction_cycles
                    total_predictions += predictor.total_predictions_made
            
            self.stats['total_cycles'] = total_cycles
            self.stats['total_predictions'] = total_predictions
            
            # Print status every 5 minutes
            if self.stats['total_cycles'] % 10 == 0 and self.stats['total_cycles'] > 0:
                print(f"\n📊 Multi-Crypto Status Update:")
                print(f"   Active predictors: {len(self.predictors)}/{len(self.crypto_symbols)}")
                print(f"   Total cycles: {total_cycles}")
                print(f"   Total predictions: {total_predictions:,}")
                for crypto_symbol, predictor in self.predictors.items():
                    if predictor and hasattr(predictor, 'prediction_cycles'):
                        print(f"   {crypto_symbol}: {predictor.prediction_cycles} cycles, {predictor.total_predictions_made:,} predictions")
    
    def stop(self):
        """Stop all cryptocurrency predictors."""
        if not self.is_running:
            return
        
        print(f"\n🛑 Stopping Multi-Crypto Orchestrator...")
        self.is_running = False
        self.stop_event.set()
        
        # Stop all predictors
        for crypto_symbol, predictor in self.predictors.items():
            if predictor:
                try:
                    predictor.stop()
                    print(f"✅ Stopped {crypto_symbol} predictor")
                except Exception as e:
                    print(f"⚠️ Error stopping {crypto_symbol} predictor: {str(e)}")
        
        # Wait for threads to finish
        for crypto_symbol, thread in self.threads.items():
            if thread.is_alive():
                thread.join(timeout=10)
                if thread.is_alive():
                    print(f"⚠️ Thread for {crypto_symbol} did not stop gracefully")
        
        # Final statistics
        self._update_stats()
        print(f"\n📊 === Final Multi-Crypto Statistics ===")
        print(f"   Session duration: {datetime.utcnow() - self.stats['start_time']}")
        print(f"   Total cycles completed: {self.stats['total_cycles']}")
        print(f"   Total predictions made: {self.stats['total_predictions']:,}")
        print(f"   Errors encountered: {len(self.stats['errors'])}")
        
        if self.stats['errors']:
            print(f"   Recent errors:")
            for error in self.stats['errors'][-5:]:  # Show last 5 errors
                print(f"     {error['crypto_symbol']}: {error['error']}")
        
        print(f"✅ Multi-Crypto Orchestrator stopped")
    
    def get_status(self) -> Dict:
        """Get current status of all predictors."""
        status = {
            'is_running': self.is_running,
            'crypto_symbols': self.crypto_symbols,
            'predictors': {},
            'statistics': self.stats.copy()
        }
        
        for crypto_symbol, predictor in self.predictors.items():
            if predictor:
                status['predictors'][crypto_symbol] = {
                    'is_running': predictor.is_running if hasattr(predictor, 'is_running') else False,
                    'prediction_cycles': predictor.prediction_cycles if hasattr(predictor, 'prediction_cycles') else 0,
                    'total_predictions': predictor.total_predictions_made if hasattr(predictor, 'total_predictions_made') else 0,
                    'model_version': predictor.current_model_version if hasattr(predictor, 'current_model_version') else 'unknown'
                }
        
        return status


def main():
    """Main function to run multi-crypto prediction."""
    print("🚀 Multi-Crypto Volatility - Orchestrated Continuous Prediction")
    print("=" * 60)
    
    try:
        # Get crypto symbols from command line arguments or use all
        crypto_symbols = None
        if len(sys.argv) > 1:
            crypto_symbols = sys.argv[1:]
            print(f"📊 Running for specified cryptocurrencies: {', '.join(crypto_symbols)}")
        else:
            crypto_symbols = list(Config.SUPPORTED_CRYPTOS.keys())
            print(f"📊 Running for all supported cryptocurrencies: {', '.join(crypto_symbols)}")
        
        # Pre-flight check: Verify models exist
        print("\n🔍 Pre-flight check: Verifying trained models...")
        import glob
        missing_models = []
        available_models = []
        
        for crypto_symbol in crypto_symbols:
            model_pattern = f"models/{crypto_symbol}_model_*.pth"
            model_files = glob.glob(model_pattern)
            
            if model_files:
                available_models.append(crypto_symbol)
                print(f"  ✅ {crypto_symbol}: {len(model_files)} model(s) found")
            else:
                missing_models.append(crypto_symbol)
                print(f"  ❌ {crypto_symbol}: No models found")
        
        if missing_models:
            print(f"\n⚠️ Missing models for: {', '.join(missing_models)}")
            print("💡 To train models, run:")
            for crypto_symbol in missing_models:
                print(f"   python trainer.py {crypto_symbol}")
            print(f"\n🚀 Proceeding with available models: {', '.join(available_models)}")
            crypto_symbols = available_models
        
        if not crypto_symbols:
            print("❌ No trained models found. Please train at least one model first.")
            return
        
        # Initialize orchestrator
        orchestrator = MultiCryptoOrchestrator(crypto_symbols=crypto_symbols)
        
        # Start multi-crypto prediction
        orchestrator.start()
        
    except KeyboardInterrupt:
        print("\n🛑 Interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        print("\n💡 Troubleshooting:")
        print("   1. Ensure MongoDB is running (if using database features)")
        print("   2. Check internet connection for Pyth Network API access")
        print("   3. Verify models are trained for all cryptocurrencies")
        print("   4. Install dependencies: pip install -r requirements.txt")
        print("   5. Check Pyth Network API status if data fetching fails")
        print("   6. Ensure sufficient system resources for multiple predictors")
    finally:
        print("\n🔚 Multi-Crypto Orchestrator terminated")


if __name__ == "__main__":
    main() 
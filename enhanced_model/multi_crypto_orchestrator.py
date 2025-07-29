#!/usr/bin/env python3

"""
Enhanced Multi-Crypto Orchestrator
Runs enhanced continuous prediction for multiple cryptocurrencies simultaneously.
Supports: BTC, ETH, XAU, SOL
Each crypto runs in its own thread with independent prediction cycles and retraining.
"""

import os
import sys
import time
import signal
import threading
import queue
from datetime import datetime
from typing import Dict, List, Optional
from dotenv import load_dotenv

from continuous_predictor import EnhancedContinuousCryptoPredictor
from config import EnhancedConfig

# Load environment variables
load_dotenv()

class EnhancedMultiCryptoOrchestrator:
    """
    Enhanced multi-crypto orchestrator that manages prediction for multiple cryptocurrencies.
    Each cryptocurrency runs in its own thread with independent cycles.
    """
    
    def __init__(self, crypto_symbols: Optional[List[str]] = None):
        """
        Initialize the enhanced multi-crypto orchestrator.
        
        Args:
            crypto_symbols: List of cryptocurrency symbols to run. If None, runs all supported cryptos.
        """
        if crypto_symbols is None:
            crypto_symbols = list(EnhancedConfig.SUPPORTED_CRYPTOS.keys())
        
        # Validate crypto symbols
        for symbol in crypto_symbols:
            if symbol not in EnhancedConfig.SUPPORTED_CRYPTOS:
                raise ValueError(f"Unsupported crypto symbol: {symbol}. Supported: {list(EnhancedConfig.SUPPORTED_CRYPTOS.keys())}")
        
        self.crypto_symbols = crypto_symbols
        self.predictors = {}
        self.threads = {}
        self.is_running = False
        self.thread_queues = {}
        self.thread_events = {}
        
        # Statistics tracking
        self.stats = {
            'start_time': None,
            'total_cycles': 0,
            'total_predictions': 0,
            'errors': 0,
            'crypto_stats': {}
        }
        
        print(f"Enhanced multi-crypto orchestrator: {len(self.crypto_symbols)} cryptos")
    
    def _create_predictor(self, crypto_symbol: str) -> EnhancedContinuousCryptoPredictor:
        """
        Create a predictor for a specific cryptocurrency.
        
        Args:
            crypto_symbol: Cryptocurrency symbol
            
        Returns:
            EnhancedContinuousCryptoPredictor instance
        """
        try:
            predictor = EnhancedContinuousCryptoPredictor(crypto_symbol)
            return predictor
        except Exception as e:
            print(f"Failed to create predictor for {crypto_symbol}: {str(e)}")
            raise
    
    def _crypto_prediction_worker(self, crypto_symbol: str):
        """
        Worker thread for a specific cryptocurrency.
        
        Args:
            crypto_symbol: Cryptocurrency symbol
        """
        try:
            # Create predictor
            predictor = self._create_predictor(crypto_symbol)
            self.predictors[crypto_symbol] = predictor
            
            # Initialize stats for this crypto
            self.stats['crypto_stats'][crypto_symbol] = {
                'cycles': 0,
                'predictions': 0,
                'errors': 0,
                'last_cycle': None,
                'last_error': None
            }
            
            # Run prediction cycles
            while self.is_running:
                try:
                    start_time = time.time()
                    
                    # Run one prediction cycle
                    success = predictor.run_prediction_cycle()
                    
                    if success:
                        # Update statistics
                        self.stats['crypto_stats'][crypto_symbol]['cycles'] += 1
                        self.stats['crypto_stats'][crypto_symbol]['predictions'] += 288  # 288 predictions per cycle
                        self.stats['crypto_stats'][crypto_symbol]['last_cycle'] = datetime.now()
                        
                        self.stats['total_cycles'] += 1
                        self.stats['total_predictions'] += 288
                    else:
                        self.stats['crypto_stats'][crypto_symbol]['errors'] += 1
                        self.stats['crypto_stats'][crypto_symbol]['last_error'] = datetime.now()
                        self.stats['errors'] += 1
                    
                    # Wait for next cycle (5 minutes)
                    elapsed = time.time() - start_time
                    sleep_time = max(0, (5 * 60) - elapsed)  # 5 minutes between cycles
                    
                    if sleep_time > 0:
                        time.sleep(sleep_time)
                    
                except Exception as e:
                    self.stats['crypto_stats'][crypto_symbol]['errors'] += 1
                    self.stats['crypto_stats'][crypto_symbol]['last_error'] = datetime.now()
                    self.stats['errors'] += 1
                    
                    # Wait before retrying
                    time.sleep(60)  # Wait 1 minute before retrying
                    
        except Exception as e:
            print(f"Fatal error in {crypto_symbol} worker: {str(e)}")
    
    def _monitor_threads(self):
        """Monitor thread health and restart failed threads."""
        while self.is_running:
            try:
                for crypto_symbol in self.crypto_symbols:
                    thread = self.threads.get(crypto_symbol)
                    
                    if thread and not thread.is_alive():
                        self._restart_crypto_thread(crypto_symbol)
                
                # Print status every 30 minutes
                self._print_status()
                time.sleep(1800)  # 30 minutes
                
            except Exception as e:
                time.sleep(60)
    
    def _restart_crypto_thread(self, crypto_symbol: str):
        """Restart a specific crypto thread."""
        try:
            # Stop existing thread if running
            if crypto_symbol in self.threads and self.threads[crypto_symbol].is_alive():
                self.thread_queues[crypto_symbol].put("STOP")
                self.threads[crypto_symbol].join(timeout=30)
            
            # Create new thread
            thread = threading.Thread(
                target=self._crypto_prediction_worker,
                args=(crypto_symbol,),
                daemon=True,
                name=f"Enhanced_{crypto_symbol}"
            )
            
            self.threads[crypto_symbol] = thread
            thread.start()
            
        except Exception as e:
            print(f"Failed to restart {crypto_symbol} thread: {str(e)}")
    
    def _print_status(self):
        """Print current status of all cryptocurrencies."""
        print(f"\nMulti-Crypto Status: {self.stats['total_cycles']} cycles, {self.stats['total_predictions']} predictions, {self.stats['errors']} errors")
        
        for crypto_symbol in self.crypto_symbols:
            stats = self.stats['crypto_stats'].get(crypto_symbol, {})
            thread_alive = self.threads.get(crypto_symbol, None) and self.threads[crypto_symbol].is_alive()
            
            status = "RUNNING" if thread_alive else "STOPPED"
            cycles = stats.get('cycles', 0)
            errors = stats.get('errors', 0)
            
            print(f"  {crypto_symbol}: {status} | Cycles: {cycles} | Errors: {errors}")
    
    def _get_runtime(self) -> str:
        """Get formatted runtime string."""
        if not self.stats['start_time']:
            return "0:00:00"
        
        runtime = datetime.now() - self.stats['start_time']
        hours = runtime.seconds // 3600
        minutes = (runtime.seconds % 3600) // 60
        seconds = runtime.seconds % 60
        
        return f"{hours}:{minutes:02d}:{seconds:02d}"
    
    def start(self):
        """Start the enhanced multi-crypto orchestrator."""
        print(f"Starting enhanced multi-crypto orchestrator...")
        print(f"Press Ctrl+C to stop")
        
        self.is_running = True
        self.stats['start_time'] = datetime.now()
        
        # Set up signal handlers
        def signal_handler(signum, frame):
            print(f"\nStopping enhanced orchestrator...")
            self.stop()
            sys.exit(0)
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
        try:
            # Start worker threads for each cryptocurrency
            for crypto_symbol in self.crypto_symbols:
                # Create thread queue and event
                self.thread_queues[crypto_symbol] = queue.Queue()
                self.thread_events[crypto_symbol] = threading.Event()
                
                # Create and start thread
                thread = threading.Thread(
                    target=self._crypto_prediction_worker,
                    args=(crypto_symbol,),
                    daemon=True,
                    name=f"Enhanced_{crypto_symbol}"
                )
                
                self.threads[crypto_symbol] = thread
                thread.start()
                
                # Small delay between thread starts
                time.sleep(1)
            
            # Start monitor thread
            monitor_thread = threading.Thread(target=self._monitor_threads, daemon=True)
            monitor_thread.start()
            
            print(f"Enhanced multi-crypto orchestrator started")
            
            # Keep main thread alive
            while self.is_running:
                time.sleep(60)  # Check every minute
                
        except KeyboardInterrupt:
            print(f"\nEnhanced orchestrator interrupted")
        except Exception as e:
            print(f"Enhanced orchestrator error: {str(e)}")
        finally:
            self.stop()
    
    def stop(self):
        """Stop the enhanced multi-crypto orchestrator."""
        print(f"Stopping enhanced multi-crypto orchestrator...")
        
        self.is_running = False
        
        # Stop all threads
        for crypto_symbol, thread in self.threads.items():
            if thread.is_alive():
                thread.join(timeout=30)
        
        # Print final statistics
        self._print_final_stats()
        
        print(f"Enhanced multi-crypto orchestrator stopped")
    
    def _print_final_stats(self):
        """Print final statistics."""
        print(f"\nFinal Statistics - Runtime: {self._get_runtime()}")
        print(f"Total cycles: {self.stats['total_cycles']}")
        print(f"Total predictions: {self.stats['total_predictions']}")
        print(f"Total errors: {self.stats['errors']}")
        
        for crypto_symbol in self.crypto_symbols:
            stats = self.stats['crypto_stats'].get(crypto_symbol, {})
            cycles = stats.get('cycles', 0)
            predictions = stats.get('predictions', 0)
            errors = stats.get('errors', 0)
            
            print(f"{crypto_symbol}: Cycles: {cycles} | Predictions: {predictions} | Errors: {errors}")

def main():
    """Main function to run enhanced multi-crypto orchestrator."""
    import sys
    
    # Parse command line arguments
    if len(sys.argv) > 1:
        crypto_symbols = [symbol.upper() for symbol in sys.argv[1:]]
    else:
        # Use all supported cryptos if none specified
        crypto_symbols = None
    
    try:
        # Initialize and start orchestrator
        orchestrator = EnhancedMultiCryptoOrchestrator(crypto_symbols)
        orchestrator.start()
        
    except Exception as e:
        print(f"Enhanced multi-crypto orchestrator failed: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
#!/usr/bin/env python3

"""
Train All Cryptocurrencies Script
Trains models for all supported cryptocurrencies (BTC, ETH, XAU, SOL)
using their existing data files in the training_data directory.
"""

import os
import sys
from datetime import datetime
from config import Config
from trainer import CryptoVolatilityTrainer

def train_crypto_model(crypto_symbol: str) -> bool:
    """
    Train a model for a specific cryptocurrency.
    
    Args:
        crypto_symbol: Cryptocurrency symbol (BTC, ETH, XAU, SOL)
        
    Returns:
        bool: True if training successful, False otherwise
    """
    crypto_config = Config.SUPPORTED_CRYPTOS[crypto_symbol]
    crypto_name = crypto_config['name']
    data_file = crypto_config['data_file']
    
    # Check if data file exists
    data_path = os.path.join('training_data', data_file)
    if not os.path.exists(data_path):
        print(f"❌ Data file not found for {crypto_name}: {data_path}")
        return False
    
    print(f"\n🚀 Training {crypto_name} ({crypto_symbol}) model...")
    print(f"📁 Using data file: {data_path}")
    
    try:
        # Initialize trainer
        config = Config()
        trainer = CryptoVolatilityTrainer(config, crypto_symbol)
        
        # Train the model
        training_history = trainer.train(data_path)
        
        if training_history:
            print(f"✅ {crypto_name} model training completed successfully!")
            print(f"📊 Final validation loss: {training_history['final_val_loss']:.6f}")
            return True
        else:
            print(f"❌ {crypto_name} model training failed")
            return False
            
    except Exception as e:
        print(f"❌ Error training {crypto_name} model: {str(e)}")
        return False

def main():
    """Main function to train all cryptocurrency models."""
    print("🚀 Multi-Crypto Model Training")
    print("=" * 50)
    
    # Check if training_data directory exists
    if not os.path.exists('training_data'):
        print("❌ training_data directory not found!")
        print("Please ensure you have the following data files:")
        for symbol, config in Config.SUPPORTED_CRYPTOS.items():
            print(f"  - training_data/{config['data_file']} ({config['name']})")
        return
    
    # List available data files
    print("📁 Available data files:")
    available_cryptos = []
    for symbol, config in Config.SUPPORTED_CRYPTOS.items():
        data_path = os.path.join('training_data', config['data_file'])
        if os.path.exists(data_path):
            file_size = os.path.getsize(data_path) / (1024 * 1024)  # MB
            print(f"  ✅ {config['data_file']} ({config['name']}) - {file_size:.1f} MB")
            available_cryptos.append(symbol)
        else:
            print(f"  ❌ {config['data_file']} ({config['name']}) - Missing")
    
    if not available_cryptos:
        print("❌ No data files found. Please add data files to training_data/ directory.")
        return
    
    # Get crypto symbols to train from command line or use all available
    crypto_symbols = available_cryptos
    if len(sys.argv) > 1:
        requested_symbols = [s.upper() for s in sys.argv[1:]]
        crypto_symbols = [s for s in requested_symbols if s in available_cryptos]
        
        if not crypto_symbols:
            print(f"❌ None of the requested symbols {requested_symbols} have available data.")
            return
        
        print(f"\n📊 Training models for: {', '.join(crypto_symbols)}")
    else:
        print(f"\n📊 Training models for all available cryptocurrencies: {', '.join(crypto_symbols)}")
    
    # Train models
    start_time = datetime.now()
    successful_trains = 0
    
    for crypto_symbol in crypto_symbols:
        if train_crypto_model(crypto_symbol):
            successful_trains += 1
    
    # Summary
    end_time = datetime.now()
    duration = end_time - start_time
    
    print(f"\n" + "=" * 50)
    print("📊 Training Summary")
    print(f"   Total models: {len(crypto_symbols)}")
    print(f"   Successful: {successful_trains}")
    print(f"   Failed: {len(crypto_symbols) - successful_trains}")
    print(f"   Duration: {duration}")
    
    if successful_trains > 0:
        print(f"\n✅ Training completed! Models saved to models/ directory.")
        print(f"💡 Next steps:")
        print(f"   1. Use multi_crypto_orchestrator.py for continuous prediction")
        print(f"   2. Use example_usage.py for individual crypto testing")
        print(f"   3. Check models/ directory for trained model files")
    else:
        print(f"\n❌ No models were trained successfully.")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⏹️ Training interrupted by user")
    except Exception as e:
        print(f"\n❌ Unexpected error: {str(e)}")
        import traceback
        traceback.print_exc() 
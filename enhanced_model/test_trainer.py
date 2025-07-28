#!/usr/bin/env python3
"""
Test script for enhanced model training
"""

import os
import sys
from config import EnhancedConfig
from trainer import EnhancedCryptoVolatilityTrainer

def main():
    print("🧪 Testing enhanced model training...")
    
    # Initialize config
    config = EnhancedConfig()
    
    # Test BTC training
    crypto_symbol = 'BTC'
    csv_path = f'../training_data/{crypto_symbol.lower()}_5min.csv'
    
    if not os.path.exists(csv_path):
        print(f"❌ CSV file not found: {csv_path}")
        return
    
    print(f"📁 Using CSV file: {csv_path}")
    
    try:
        # Initialize trainer
        trainer = EnhancedCryptoVolatilityTrainer(config, crypto_symbol)
        
        # Start training
        print("🚀 Starting training...")
        results = trainer.train(csv_path)
        
        print("✅ Training completed successfully!")
        print(f"Results: {results}")
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
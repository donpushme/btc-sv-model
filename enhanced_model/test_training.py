#!/usr/bin/env python3
"""
Simple Testing Script for Realistic Model Training

This script tests the training pipeline with:
- Smaller dataset (first 1000 rows)
- Fewer epochs (5 instead of 150)
- Smaller batch size (8 instead of 32)
- Quick validation to ensure everything works
"""

import os
import sys
import pandas as pd
import numpy as np
from config import RealisticConfig
from trainer import RealisticModelTrainer

def create_test_data(crypto_symbol: str, num_rows: int = 1000) -> str:
    """
    Create a small test dataset for quick testing.
    """
    print(f"📊 Creating test data for {crypto_symbol}...")
    
    # Create test data directory if it doesn't exist
    test_data_dir = "test_data"
    os.makedirs(test_data_dir, exist_ok=True)
    
    # Generate synthetic test data
    np.random.seed(42)  # For reproducible results
    
    # Create timestamp range
    timestamps = pd.date_range(start='2023-01-01', periods=num_rows, freq='5T')
    
    # Generate realistic price data
    base_price = 50000 if crypto_symbol == 'BTC' else 3000 if crypto_symbol == 'ETH' else 100 if crypto_symbol == 'SOL' else 2000
    
    # Random walk for prices
    returns = np.random.normal(0, 0.001, num_rows)  # Small random returns
    prices = [base_price]
    for ret in returns[1:]:
        prices.append(prices[-1] * (1 + ret))
    
    # Create OHLCV data
    data = {
        'timestamp': timestamps,
        'open': prices,
        'high': [p * (1 + abs(np.random.normal(0, 0.002))) for p in prices],
        'low': [p * (1 - abs(np.random.normal(0, 0.002))) for p in prices],
        'close': prices,
        'volume': np.random.uniform(1000, 10000, num_rows)
    }
    
    # Ensure high >= close >= low
    for i in range(num_rows):
        data['high'][i] = max(data['high'][i], data['close'][i])
        data['low'][i] = min(data['low'][i], data['close'][i])
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Save to file
    filename = f"{crypto_symbol.lower()}_test.csv"
    filepath = os.path.join(test_data_dir, filename)
    df.to_csv(filepath, index=False)
    
    print(f"✅ Test data saved: {filepath} ({len(df)} rows)")
    return filepath

def test_single_crypto(crypto_symbol: str) -> bool:
    """
    Test training for a single cryptocurrency with minimal settings.
    """
    print(f"\n🧪 Testing {crypto_symbol} training...")
    print("-" * 50)
    
    try:
        # Create test data
        test_file = create_test_data(crypto_symbol, num_rows=1000)
        
        # Create minimal config for testing
        test_config = RealisticConfig()
        test_config.NUM_EPOCHS = 3  # Very few epochs for quick testing
        test_config.BATCH_SIZE = 8  # Smaller batch size
        test_config.EARLY_STOPPING_PATIENCE = 2  # Quick early stopping
        test_config.DATA_PATH = "test_data"  # Use test data directory
        test_config.SEQUENCE_LENGTH = 48  # Shorter sequence for testing
        
        # Update the crypto config to use test data
        test_config.SUPPORTED_CRYPTOS[crypto_symbol]['data_file'] = f"{crypto_symbol.lower()}_test.csv"
        
        # Initialize trainer
        trainer = RealisticModelTrainer(test_config, crypto_symbol)
        
        # Load and preprocess data
        print("📊 Loading and preprocessing test data...")
        df = trainer.load_and_preprocess_data()
        
        # Prepare training data
        print("🔄 Preparing training data...")
        X_train, X_val, y_train, y_val, time_train, time_val, feature_cols = trainer.prepare_training_data(df)
        
        print(f"✅ Data prepared:")
        print(f"   X_train: {X_train.shape}")
        print(f"   X_val: {X_val.shape}")
        print(f"   y_train: {y_train.shape}")
        print(f"   y_val: {y_val.shape}")
        print(f"   Features: {len(feature_cols)}")
        
        # Train model (quick test)
        print("🚀 Starting quick training test...")
        trainer.train(X_train, X_val, y_train, y_val, time_train, time_val, feature_cols)
        
        print(f"✅ {crypto_symbol} training test completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Error testing {crypto_symbol}: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """
    Main testing function.
    """
    print("🧪 Realistic Model Training Test")
    print("=" * 60)
    print("This script tests the training pipeline with minimal settings:")
    print("- Small test datasets (1000 rows each)")
    print("- Few epochs (3 instead of 150)")
    print("- Small batch size (8 instead of 32)")
    print("- Quick early stopping")
    print("=" * 60)
    
    # Test with one crypto first
    test_cryptos = ['BTC']  # Start with just BTC for quick testing
    
    results = {}
    
    for crypto_symbol in test_cryptos:
        success = test_single_crypto(crypto_symbol)
        results[crypto_symbol] = success
        
        if success:
            print(f"🎉 {crypto_symbol} test PASSED!")
        else:
            print(f"💥 {crypto_symbol} test FAILED!")
    
    # Summary
    print("\n" + "=" * 60)
    print("📋 Test Summary")
    print("=" * 60)
    
    passed = sum(results.values())
    total = len(results)
    
    for crypto_symbol, success in results.items():
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"{crypto_symbol}: {status}")
    
    print(f"\n🎯 Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The training pipeline is working correctly.")
        print("💡 You can now run the full training with:")
        print("   cd enhanced_model")
        print("   python train_all_cryptos.py")
    else:
        print("💥 Some tests failed. Please check the error messages above.")
    
    # Cleanup test data
    print("\n🧹 Cleaning up test data...")
    test_data_dir = "test_data"
    if os.path.exists(test_data_dir):
        import shutil
        shutil.rmtree(test_data_dir)
        print("✅ Test data cleaned up")

if __name__ == "__main__":
    main() 
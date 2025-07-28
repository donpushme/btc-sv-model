#!/usr/bin/env python3
"""
Enhanced Training Script for All Cryptocurrencies

This script trains enhanced models for all supported cryptocurrencies
using the enhanced model architecture optimized for Monte Carlo simulation.
"""

import os
import sys
from config import EnhancedConfig
from trainer import EnhancedCryptoVolatilityTrainer

def train_all_enhanced_models():
    """
    Train enhanced models for all supported cryptocurrencies.
    """
    print("🚀 Enhanced Training for All Cryptocurrencies")
    print("=" * 60)
    
    config = EnhancedConfig()
    results = {}
    
    for crypto_symbol in EnhancedConfig.SUPPORTED_CRYPTOS.keys():
        print(f"\n📊 Training enhanced model for {crypto_symbol}...")
        print("-" * 40)
        
        try:
            # Initialize trainer
            trainer = EnhancedCryptoVolatilityTrainer(config, crypto_symbol)
            
            # Get data path
            data_file = EnhancedConfig.SUPPORTED_CRYPTOS[crypto_symbol]['data_file']
            csv_path = os.path.join(config.DATA_PATH, data_file)
            
            # Check if data file exists
            if not os.path.exists(csv_path):
                print(f"❌ Data file not found: {csv_path}")
                results[crypto_symbol] = {'error': 'Data file not found'}
                continue
            
            # Train model
            training_results = trainer.train(csv_path)
            
            print(f"✅ Enhanced training completed for {crypto_symbol}")
            print(f"   Final validation loss: {training_results['final_val_loss']:.6f}")
            print(f"   Best validation loss: {training_results['best_val_loss']:.6f}")
            print(f"   Epochs trained: {training_results['epochs_trained']}")
            
            results[crypto_symbol] = training_results
            
        except Exception as e:
            print(f"❌ Error training enhanced {crypto_symbol} model: {str(e)}")
            results[crypto_symbol] = {'error': str(e)}
    
    # Print summary
    print("\n" + "=" * 60)
    print("📋 Enhanced Training Summary")
    print("=" * 60)
    
    successful_training = 0
    total_cryptos = len(EnhancedConfig.SUPPORTED_CRYPTOS)
    
    for crypto_symbol, result in results.items():
        if 'error' in result:
            print(f"❌ {crypto_symbol}: {result['error']}")
        else:
            print(f"✅ {crypto_symbol}: Val Loss = {result['final_val_loss']:.6f}")
            successful_training += 1
    
    print(f"\n🎯 Successfully trained {successful_training}/{total_cryptos} enhanced models")
    
    if successful_training == total_cryptos:
        print("🎉 All enhanced models trained successfully!")
    else:
        print(f"⚠️  {total_cryptos - successful_training} models failed to train")
    
    return results

def main():
    """Main function."""
    print("Enhanced Monte Carlo Model Training")
    print("This will train enhanced models for all supported cryptocurrencies.")
    print("Models will be saved in the root models directory.")
    print("Training data will be read from the root training_data directory.")
    
    # Check if training data exists
    config = EnhancedConfig()
    missing_data = []
    
    for crypto_symbol, crypto_config in EnhancedConfig.SUPPORTED_CRYPTOS.items():
        data_file = crypto_config['data_file']
        csv_path = os.path.join(config.DATA_PATH, data_file)
        
        if not os.path.exists(csv_path):
            missing_data.append(f"{crypto_symbol}: {csv_path}")
    
    if missing_data:
        print("\n❌ Missing training data files:")
        for missing in missing_data:
            print(f"   {missing}")
        print("\nPlease ensure all training data files are present in the training_data directory.")
        sys.exit(1)
    
    # Start training
    results = train_all_enhanced_models()
    
    # Save results summary
    import json
    summary_path = os.path.join(config.RESULTS_PATH, "enhanced_training_summary.json")
    with open(summary_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n📄 Training summary saved to: {summary_path}")

if __name__ == "__main__":
    main() 
#!/usr/bin/env python3
"""
Enhanced Training Script for All Cryptocurrencies

This script trains enhanced models for all supported cryptocurrencies
using the enhanced model architecture optimized for Monte Carlo simulation.
"""

import os
import sys
from config import RealisticConfig
from trainer import RealisticModelTrainer

def train_all_enhanced_models():
    """
    Train enhanced models for all supported cryptocurrencies.
    """
    print("🚀 Realistic Enhanced Training for All Cryptocurrencies")
    print("=" * 60)
    
    config = RealisticConfig()
    results = {}
    
    for crypto_symbol in RealisticConfig.SUPPORTED_CRYPTOS.keys():
        print(f"\n📊 Training enhanced model for {crypto_symbol}...")
        print("-" * 40)
        
        try:
            # Initialize trainer
            trainer = RealisticModelTrainer(config, crypto_symbol)
            

            
            # Load and preprocess data
            df = trainer.load_and_preprocess_data()
            
            # Prepare training data
            X_train, X_val, y_train, y_val, time_train, time_val, feature_cols = trainer.prepare_training_data(df)
            
            # Train model
            trainer.train(X_train, X_val, y_train, y_val, time_train, time_val, feature_cols)
            
            print(f"✅ Enhanced training completed for {crypto_symbol}")
            
            # Get training results from trainer history
            final_val_loss = trainer.val_losses[-1] if trainer.val_losses else 0.0
            best_val_loss = min(trainer.val_losses) if trainer.val_losses else 0.0
            epochs_trained = len(trainer.train_losses)
            
            print(f"   Final validation loss: {final_val_loss:.6f}")
            print(f"   Best validation loss: {best_val_loss:.6f}")
            print(f"   Epochs trained: {epochs_trained}")
            
            results[crypto_symbol] = {
                'final_val_loss': final_val_loss,
                'best_val_loss': best_val_loss,
                'epochs_trained': epochs_trained
            }
            
        except Exception as e:
            print(f"❌ Error training enhanced {crypto_symbol} model: {str(e)}")
            results[crypto_symbol] = {'error': str(e)}
    
    # Print summary
    print("\n" + "=" * 60)
    print("📋 Enhanced Training Summary")
    print("=" * 60)
    
    successful_training = 0
    total_cryptos = len(RealisticConfig.SUPPORTED_CRYPTOS)
    
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
    print("Realistic Enhanced Model Training")
    print("This will train realistic enhanced models for all supported cryptocurrencies.")
    print("Models will be saved in the root models directory.")
    print("Training data will be read from the root training_data directory.")
    
    # Check if training data exists
    config = RealisticConfig()
    missing_data = []
    
    for crypto_symbol, crypto_config in RealisticConfig.SUPPORTED_CRYPTOS.items():
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
    summary_path = os.path.join(config.RESULTS_PATH, "realistic_training_summary.json")
    with open(summary_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n📄 Training summary saved to: {summary_path}")

if __name__ == "__main__":
    main() 
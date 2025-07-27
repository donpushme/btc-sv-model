#!/usr/bin/env python3

"""
Test script to verify model loading is working correctly.
Run this on your device to check if the models can be loaded properly.
"""

import os
import sys
from predictor import RealTimeVolatilityPredictor
from config import Config

def test_model_loading():
    """Test loading models for all supported cryptocurrencies."""
    
    print("🧪 Testing Model Loading")
    print("=" * 50)
    
    # Get all model files in the models directory
    models_dir = Config().MODEL_SAVE_PATH
    if not os.path.exists(models_dir):
        print(f"❌ Models directory not found: {models_dir}")
        return
    
    print(f"📁 Models directory: {models_dir}")
    
    # List all files
    all_files = os.listdir(models_dir)
    print(f"📁 All files: {all_files}")
    
    # Find model files for each crypto
    from config import Config
    supported_cryptos = Config.SUPPORTED_CRYPTOS
    
    for crypto_symbol in supported_cryptos:
        print(f"\n🔍 Testing {crypto_symbol} ({supported_cryptos[crypto_symbol]['name']})")
        
        # Find model files
        model_files = [f for f in all_files if f.startswith(f'{crypto_symbol}_model_') and f.endswith('.pth')]
        legacy_files = [f for f in all_files if f == 'best_model.pth' or f.startswith(f'{crypto_symbol}_best_model.pth')]
        
        if model_files:
            print(f"  ✅ Found {len(model_files)} model(s) with new naming: {model_files}")
        elif legacy_files:
            print(f"  ⚠️ Found {len(legacy_files)} legacy model(s): {legacy_files}")
        else:
            print(f"  ❌ No models found for {crypto_symbol}")
            continue
        
        # Try to load the model
        try:
            print(f"  🔧 Attempting to load model...")
            predictor = RealTimeVolatilityPredictor(crypto_symbol=crypto_symbol)
            
            if predictor.model is not None:
                print(f"  ✅ Model loaded successfully!")
                print(f"  📊 Model info: {predictor.get_model_info()}")
            else:
                print(f"  ❌ Model is None after loading")
                
        except Exception as e:
            print(f"  ❌ Failed to load model: {str(e)}")
            import traceback
            print(f"  📋 Traceback: {traceback.format_exc()}")

if __name__ == "__main__":
    test_model_loading() 
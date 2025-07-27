#!/usr/bin/env python3

"""
Test script to verify multi-crypto orchestrator can start properly.
Run this on your device to check if the orchestrator can initialize without errors.
"""

import os
import sys
from multi_crypto_orchestrator import MultiCryptoOrchestrator
from config import Config

def test_orchestrator_initialization():
    """Test initializing the multi-crypto orchestrator."""
    
    print("🧪 Testing Multi-Crypto Orchestrator Initialization")
    print("=" * 60)
    
    # Get all model files in the models directory
    models_dir = Config().MODEL_SAVE_PATH
    if not os.path.exists(models_dir):
        print(f"❌ Models directory not found: {models_dir}")
        return
    
    print(f"📁 Models directory: {models_dir}")
    
    # List all files
    all_files = os.listdir(models_dir)
    print(f"📁 All files: {all_files}")
    
    # Find available models
    from config import Config
    supported_cryptos = Config.SUPPORTED_CRYPTOS
    available_cryptos = []
    
    for crypto_symbol in supported_cryptos:
        print(f"\n🔍 Checking {crypto_symbol} ({supported_cryptos[crypto_symbol]['name']})")
        
        # Find model files
        model_files = [f for f in all_files if f.startswith(f'{crypto_symbol}_model_') and f.endswith('.pth')]
        legacy_files = [f for f in all_files if f == 'best_model.pth' or f.startswith(f'{crypto_symbol}_best_model.pth')]
        
        if model_files:
            print(f"  ✅ Found {len(model_files)} model(s) with new naming: {model_files}")
            available_cryptos.append(crypto_symbol)
        elif legacy_files:
            print(f"  ⚠️ Found {len(legacy_files)} legacy model(s): {legacy_files}")
            available_cryptos.append(crypto_symbol)
        else:
            print(f"  ❌ No models found for {crypto_symbol}")
    
    if not available_cryptos:
        print(f"\n❌ No models found for any cryptocurrency")
        return
    
    print(f"\n🚀 Testing orchestrator initialization with: {available_cryptos}")
    
    try:
        # Initialize orchestrator with available cryptos
        orchestrator = MultiCryptoOrchestrator(crypto_symbols=available_cryptos)
        print(f"✅ Orchestrator initialized successfully!")
        
        # Get status
        status = orchestrator.get_status()
        print(f"📊 Status: {status}")
        
        print(f"\n✅ All tests passed! The orchestrator should work correctly.")
        
    except Exception as e:
        print(f"❌ Failed to initialize orchestrator: {str(e)}")
        import traceback
        print(f"📋 Traceback: {traceback.format_exc()}")

if __name__ == "__main__":
    test_orchestrator_initialization() 
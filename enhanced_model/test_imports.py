#!/usr/bin/env python3
"""
Test script for imports
"""

def main():
    print("🧪 Testing imports...")
    
    try:
        print("📦 Importing config...")
        from config import EnhancedConfig
        print("✅ Config imported successfully")
        
        print("📦 Importing data processor...")
        from data_processor import EnhancedCryptoDataProcessor
        print("✅ Data processor imported successfully")
        
        print("📦 Importing feature engineering...")
        from feature_engineering import EnhancedFeatureEngineer
        print("✅ Feature engineering imported successfully")
        
        print("📦 Importing enhanced model...")
        from enhanced_model import EnhancedVolatilityModel, create_enhanced_model
        print("✅ Enhanced model imported successfully")
        
        print("📦 Importing trainer...")
        from trainer import EnhancedCryptoVolatilityTrainer
        print("✅ Trainer imported successfully")
        
        print("✅ All imports successful!")
        
    except Exception as e:
        print(f"❌ Import failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
Test script for data loading and preprocessing
"""

import os
import sys
from config import EnhancedConfig
from data_processor import EnhancedCryptoDataProcessor
from feature_engineering import EnhancedFeatureEngineer

def main():
    print("🧪 Testing data loading and preprocessing...")
    
    # Test BTC data
    crypto_symbol = 'BTC'
    csv_path = f'../training_data/{crypto_symbol.lower()}_5min.csv'
    
    if not os.path.exists(csv_path):
        print(f"❌ CSV file not found: {csv_path}")
        return
    
    print(f"📁 Using CSV file: {csv_path}")
    
    try:
        # Test data processor
        print("📊 Testing data processor...")
        data_processor = EnhancedCryptoDataProcessor(csv_path, crypto_symbol)
        df = data_processor.preprocess_data()
        print(f"✅ Data preprocessing complete. Shape: {df.shape}")
        print(f"📋 Columns: {list(df.columns)}")
        
        # Test feature engineering
        print("🔧 Testing feature engineering...")
        feature_engineer = EnhancedFeatureEngineer()
        df = feature_engineer.engineer_features(df)
        print(f"✅ Feature engineering complete. Shape: {df.shape}")
        print(f"📋 Columns: {list(df.columns)}")
        
        # Test feature selection
        feature_cols = [col for col in df.columns if col.startswith(('rsi', 'macd', 'bb_', 'stoch_', 'williams_r', 'atr', 'realized_vol', 'parkinson_vol', 'garman_klass_vol', 'vol_of_vol', 'vwap', 'volume_ratio', 'price_efficiency', 'spread_proxy', 'tick_effect', 'vol_rsi_interaction', 'vol_macd_interaction', 'hour_vol_interaction', 'weekend_vol_interaction', 'volume_vol_interaction', 'hour', 'day_of_week', 'is_weekend', 'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos', 'us_trading_hours', 'asian_trading_hours'))]
        target_cols = ['target_volatility', 'target_skewness', 'target_kurtosis']
        
        print(f"🎯 Selected features: {len(feature_cols)}")
        print(f"🎯 Target columns: {target_cols}")
        
        # Check if target columns exist
        missing_targets = [col for col in target_cols if col not in df.columns]
        if missing_targets:
            print(f"❌ Missing target columns: {missing_targets}")
        else:
            print("✅ All target columns found")
        
        print("✅ Data loading and preprocessing test completed successfully!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
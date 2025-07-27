#!/usr/bin/env python3
"""
Debug script to diagnose retraining issues.
"""

import os
import sys
import pandas as pd
from datetime import datetime, timedelta
from database_manager import DatabaseManager
from data_processor import CryptoDataProcessor
from feature_engineering import FeatureEngineer

def debug_training_data_retrieval(crypto_symbol='BTC'):
    """Debug the training data retrieval process."""
    print(f"🔍 Debugging training data retrieval for {crypto_symbol}")
    print("=" * 60)
    
    # Initialize database manager
    db_manager = DatabaseManager(crypto_symbol=crypto_symbol)
    
    # Check training data availability
    print("\n1. Checking training data availability...")
    availability = db_manager.check_training_data_availability()
    
    if not availability:
        print("❌ No training data found in database")
        return
    
    # Try to retrieve training data
    print("\n2. Retrieving training data...")
    training_data = db_manager.get_training_data_for_update(
        hours=168,  # 7 days
        fallback_to_all=True
    )
    
    if len(training_data) == 0:
        print("❌ No training data retrieved")
        return
    
    print(f"✅ Retrieved {len(training_data)} training data points")
    print(f"📊 Columns: {list(training_data.columns)}")
    print(f"📊 Date range: {training_data['timestamp'].min()} to {training_data['timestamp'].max()}")
    
    # Test data preprocessing
    print("\n3. Testing data preprocessing...")
    
    # Save to temporary CSV
    import tempfile
    temp_csv = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
    training_data.to_csv(temp_csv.name, index=False)
    temp_csv.close()
    
    try:
        # Test CryptoDataProcessor
        processor = CryptoDataProcessor(temp_csv.name, crypto_symbol)
        
        print("🔄 Loading data...")
        df = processor.load_data()
        print(f"📊 Loaded {len(df)} data points")
        
        print("🔄 Preprocessing data...")
        processed_df = processor.preprocess_data(
            return_windows=[6, 12, 24, 48],
            prediction_horizon=288
        )
        print(f"📊 After preprocessing: {len(processed_df)} data points")
        
        if len(processed_df) == 0:
            print("❌ All data was filtered out during preprocessing!")
            print("💡 This is likely due to rolling window calculations")
            return
        
        print("🔄 Adding engineered features...")
        feature_engineer = FeatureEngineer()
        final_df = feature_engineer.engineer_features(processed_df)
        print(f"📊 After feature engineering: {len(final_df)} data points")
        
        # Remove NaN values
        initial_rows = len(final_df)
        final_df = final_df.dropna().reset_index(drop=True)
        final_rows = len(final_df)
        
        print(f"📊 After removing NaN values: {final_rows} data points (removed {initial_rows - final_rows})")
        
        if final_rows < 50:
            print("❌ Insufficient data after all processing steps")
            print("💡 Need at least 50 data points for retraining")
        else:
            print("✅ Data processing successful!")
            print(f"📊 Final dataset shape: {final_df.shape}")
            
            # Show feature columns
            feature_cols = [col for col in final_df.columns if col not in 
                           ['timestamp', 'target_volatility', 'target_skewness', 'target_kurtosis']]
            print(f"📊 Number of features: {len(feature_cols)}")
            
    finally:
        # Clean up temporary file
        os.unlink(temp_csv.name)

def main():
    """Main debug function."""
    crypto_symbol = sys.argv[1].upper() if len(sys.argv) > 1 else 'BTC'
    
    print(f"🚀 Debugging retraining for {crypto_symbol}")
    print("=" * 60)
    
    debug_training_data_retrieval(crypto_symbol)
    
    print("\n" + "=" * 60)
    print("💡 Recommendations:")
    print("1. Ensure you have at least 1000+ data points in the database")
    print("2. Check that the data has proper OHLC structure")
    print("3. Consider reducing rolling window sizes for small datasets")
    print("4. Monitor the continuous predictor to collect more data")

if __name__ == "__main__":
    main() 
#!/usr/bin/env python3
"""
Test script to verify data preprocessing works with limited data.
This helps diagnose the "Insufficient data after preprocessing" error.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import sys

# Add the current directory to the path so we can import our modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_processor import CryptoDataProcessor
from config import Config

def create_test_data(num_points: int = 100) -> pd.DataFrame:
    """Create synthetic test data with the required structure."""
    print(f"📊 Creating test data with {num_points} points...")
    
    # Create timestamps
    end_time = datetime.now()
    start_time = end_time - timedelta(minutes=5 * num_points)
    timestamps = pd.date_range(start=start_time, end=end_time, periods=num_points)
    
    # Create synthetic price data
    np.random.seed(42)  # For reproducible results
    
    # Start with a base price
    base_price = 50000.0
    prices = [base_price]
    
    # Generate price movements
    for i in range(1, num_points):
        # Random price movement
        change = np.random.normal(0, 0.02)  # 2% volatility
        new_price = prices[-1] * (1 + change)
        prices.append(max(new_price, 1000))  # Ensure positive price
    
    # Create OHLC data
    data = []
    for i, (timestamp, close_price) in enumerate(zip(timestamps, prices)):
        # Create realistic OHLC from close price
        volatility = np.random.uniform(0.005, 0.02)  # 0.5% to 2% daily volatility
        
        high = close_price * (1 + np.random.uniform(0, volatility))
        low = close_price * (1 - np.random.uniform(0, volatility))
        open_price = np.random.uniform(low, high)
        
        data.append({
            'timestamp': timestamp,
            'open': open_price,
            'high': high,
            'low': low,
            'close': close_price
        })
    
    df = pd.DataFrame(data)
    print(f"✅ Created test data with shape: {df.shape}")
    print(f"📊 Price range: ${df['low'].min():.2f} - ${df['high'].max():.2f}")
    print(f"📊 Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    
    return df

def test_preprocessing_with_limited_data():
    """Test preprocessing with various amounts of limited data."""
    print("🧪 Testing data preprocessing with limited data...")
    print("=" * 60)
    
    # Test with different data sizes
    test_sizes = [50, 100, 200, 500]
    
    for size in test_sizes:
        print(f"\n🔍 Testing with {size} data points:")
        print("-" * 40)
        
        try:
            # Create test data
            test_df = create_test_data(size)
            
            # Save to temporary CSV
            temp_csv = f"temp_test_data_{size}.csv"
            test_df.to_csv(temp_csv, index=False)
            
            # Test preprocessing
            processor = CryptoDataProcessor(temp_csv, 'BTC')
            
            # Test with different rolling windows
            if size < 100:
                windows = [3, 6, 12]
            elif size < 300:
                windows = [6, 12, 24]
            else:
                windows = [6, 12, 24, 48]
            
            print(f"📊 Using rolling windows: {windows}")
            
            # Preprocess data
            processed_df = processor.preprocess_data(
                return_windows=windows,
                prediction_horizon=min(48, size // 4)  # Adaptive prediction horizon
            )
            
            print(f"✅ Preprocessing successful!")
            print(f"📊 Final dataset shape: {processed_df.shape}")
            print(f"📊 Feature columns: {len(processor.get_feature_columns())}")
            print(f"📊 Target columns: {processor.get_target_columns()}")
            
            # Check for target values
            target_stats = processed_df[['target_volatility', 'target_skewness', 'target_kurtosis']].describe()
            print(f"📊 Target statistics:")
            print(target_stats)
            
            # Clean up
            if os.path.exists(temp_csv):
                os.remove(temp_csv)
                
        except Exception as e:
            print(f"❌ Error with {size} data points: {str(e)}")
            if os.path.exists(temp_csv):
                os.remove(temp_csv)

def test_database_data_preprocessing():
    """Test preprocessing with data retrieved from database."""
    print("\n🧪 Testing database data preprocessing...")
    print("=" * 60)
    
    try:
        from database_manager import DatabaseManager
        
        # Initialize database manager
        db_manager = DatabaseManager('BTC')
        
        # Check data availability
        availability = db_manager.check_training_data_availability()
        print(f"📊 Database availability: {availability}")
        
        # Get training data
        training_data = db_manager.get_training_data_for_update(
            hours=168,  # 7 days
            fallback_to_all=True
        )
        
        if len(training_data) == 0:
            print("❌ No training data found in database")
            return
        
        print(f"📊 Retrieved {len(training_data)} data points from database")
        print(f"📊 Columns: {list(training_data.columns)}")
        print(f"📊 Date range: {training_data['timestamp'].min()} to {training_data['timestamp'].max()}")
        
        # Save to temporary CSV
        temp_csv = "temp_db_data.csv"
        training_data.to_csv(temp_csv, index=False)
        
        # Test preprocessing
        processor = CryptoDataProcessor(temp_csv, 'BTC')
        
        # Use appropriate rolling windows based on data size
        if len(training_data) < 200:
            windows = [3, 6, 12]
        elif len(training_data) < 500:
            windows = [6, 12, 24]
        else:
            windows = [6, 12, 24, 48]
        
        print(f"📊 Using rolling windows: {windows}")
        
        # Preprocess data
        processed_df = processor.preprocess_data(
            return_windows=windows,
            prediction_horizon=min(48, len(training_data) // 4)
        )
        
        print(f"✅ Database data preprocessing successful!")
        print(f"📊 Final dataset shape: {processed_df.shape}")
        
        # Clean up
        if os.path.exists(temp_csv):
            os.remove(temp_csv)
            
    except Exception as e:
        print(f"❌ Error testing database data: {str(e)}")
        if os.path.exists(temp_csv):
            os.remove(temp_csv)

def main():
    """Main test function."""
    print("🧪 Limited Data Preprocessing Test")
    print("=" * 60)
    
    # Test with synthetic data
    test_preprocessing_with_limited_data()
    
    # Test with database data
    test_database_data_preprocessing()
    
    print("\n✅ All tests completed!")

if __name__ == "__main__":
    main() 
#!/usr/bin/env python3

"""
Test script to verify data preprocessing works correctly with limited data.
This will help diagnose the retraining issues.
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from data_processor import CryptoDataProcessor
from config import Config

def create_test_data(num_points: int = 620) -> pd.DataFrame:
    """Create test data similar to what's causing the issue."""
    
    print(f"🔧 Creating test data with {num_points} points...")
    
    # Create timestamps (5-minute intervals)
    end_time = datetime.now()
    start_time = end_time - timedelta(minutes=5 * num_points)
    timestamps = pd.date_range(start=start_time, end=end_time, freq='5min')
    
    # Create realistic price data
    np.random.seed(42)
    base_price = 45000  # Bitcoin-like price
    prices = [base_price]
    
    for i in range(1, num_points):
        # Simulate realistic price movements
        change = np.random.normal(0, 0.02)  # 2% daily volatility
        new_price = prices[-1] * (1 + change)
        prices.append(max(new_price, 1000))  # Ensure positive price
    
    # Create OHLC data
    data = []
    for i, (timestamp, close_price) in enumerate(zip(timestamps, prices)):
        # Create realistic OHLC from close price
        volatility = np.random.uniform(0.005, 0.02)  # 0.5% to 2% intra-period volatility
        
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
    print(f"✅ Created test data: {len(df)} points")
    print(f"📊 Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    print(f"📊 Price range: ${df['low'].min():.2f} - ${df['high'].max():.2f}")
    
    return df

def test_preprocessing(crypto_symbol: str = 'BTC', num_points: int = 620):
    """Test the preprocessing pipeline."""
    
    print(f"\n🧪 Testing preprocessing for {crypto_symbol}")
    print("=" * 60)
    
    # Create test data
    test_data = create_test_data(num_points)
    
    # Save to temporary CSV
    temp_csv = f"temp_test_data_{crypto_symbol}.csv"
    test_data.to_csv(temp_csv, index=False)
    print(f"💾 Saved test data to: {temp_csv}")
    
    try:
        # Initialize processor
        processor = CryptoDataProcessor(temp_csv, crypto_symbol)
        
        # Test preprocessing with different configurations
        print(f"\n🔧 Testing preprocessing with {num_points} data points...")
        
        # Test with small windows (like retraining)
        print(f"\n📊 Test 1: Small windows [3, 6, 12, 24]")
        try:
            result1 = processor.preprocess_data(
                return_windows=[3, 6, 12, 24],
                prediction_horizon=155  # Adaptive horizon for 620 points
            )
            print(f"✅ Success: {len(result1)} data points after preprocessing")
            print(f"📊 Result shape: {result1.shape}")
        except Exception as e:
            print(f"❌ Failed: {str(e)}")
        
        # Test with normal windows
        print(f"\n📊 Test 2: Normal windows [6, 12, 24, 48]")
        try:
            result2 = processor.preprocess_data(
                return_windows=[6, 12, 24, 48],
                prediction_horizon=155
            )
            print(f"✅ Success: {len(result2)} data points after preprocessing")
            print(f"📊 Result shape: {result2.shape}")
        except Exception as e:
            print(f"❌ Failed: {str(e)}")
        
        # Test with very limited data
        print(f"\n📊 Test 3: Very limited data (200 points)")
        try:
            limited_data = test_data.tail(200)
            limited_csv = f"temp_limited_{crypto_symbol}.csv"
            limited_data.to_csv(limited_csv, index=False)
            
            processor_limited = CryptoDataProcessor(limited_csv, crypto_symbol)
            result3 = processor_limited.preprocess_data(
                return_windows=[3, 6, 12, 24],
                prediction_horizon=50  # Very short horizon
            )
            print(f"✅ Success: {len(result3)} data points after preprocessing")
            print(f"📊 Result shape: {result3.shape}")
            
            # Clean up
            os.remove(limited_csv)
        except Exception as e:
            print(f"❌ Failed: {str(e)}")
        
    except Exception as e:
        print(f"❌ Test failed: {str(e)}")
        import traceback
        print(f"📋 Traceback: {traceback.format_exc()}")
    
    finally:
        # Clean up
        if os.path.exists(temp_csv):
            os.remove(temp_csv)
            print(f"🧹 Cleaned up: {temp_csv}")

def main():
    """Main test function."""
    print("🧪 Data Preprocessing Test")
    print("=" * 60)
    
    # Test with different data sizes
    test_sizes = [620, 1000, 500, 200]
    
    for size in test_sizes:
        print(f"\n{'='*60}")
        print(f"Testing with {size} data points")
        print(f"{'='*60}")
        test_preprocessing('BTC', size)

if __name__ == "__main__":
    main() 
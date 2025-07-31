#!/usr/bin/env python3

"""
Test script to demonstrate the new database cleanup functionality.
This script shows how the automatic cleanup works when saving data.
"""

import os
import sys
from datetime import datetime, timedelta
import pandas as pd
from database_manager import DatabaseManager

def create_test_data(count: int, start_date: datetime = None) -> pd.DataFrame:
    """Create test price data for demonstration."""
    if start_date is None:
        start_date = datetime.utcnow() - timedelta(days=count)
    
    timestamps = [start_date + timedelta(minutes=5*i) for i in range(count)]
    
    data = []
    base_price = 50000.0
    
    for i, timestamp in enumerate(timestamps):
        # Simulate some price movement
        price_change = (i % 100 - 50) * 10  # Oscillating price
        current_price = base_price + price_change
        
        data.append({
            'timestamp': timestamp,
            'open': current_price - 5,
            'high': current_price + 10,
            'low': current_price - 10,
            'close': current_price,
            'volume': 1000 + (i % 500)
        })
    
    return pd.DataFrame(data)

def create_test_prediction(timestamp: datetime = None) -> dict:
    """Create a test prediction for demonstration."""
    if timestamp is None:
        timestamp = datetime.utcnow()
    
    return {
        'timestamp': timestamp,
        'current_price': 50000.0,
        'predicted_volatility': 0.02,
        'predicted_skewness': 0.1,
        'predicted_kurtosis': 3.2,
        'volatility_annualized': 0.35,
        'confidence_interval_lower': 48000.0,
        'confidence_interval_upper': 52000.0,
        'market_regime': 'normal',
        'risk_assessment': 'low',
        'prediction_period': '24h'
    }

def test_cleanup_functionality():
    """Test the cleanup functionality with sample data."""
    print("🧪 Testing Database Cleanup Functionality")
    print("=" * 50)
    
    try:
        # Initialize database manager
        db_manager = DatabaseManager(crypto_symbol='BTC')
        
        # Show initial stats
        print("\n📊 Initial Database Statistics:")
        initial_stats = db_manager.get_database_stats()
        
        # Test 1: Add training data and see cleanup in action
        print("\n🧪 Test 1: Adding training data...")
        for i in range(5):
            test_data = create_test_data(100, datetime.utcnow() - timedelta(days=i))
            doc_id = db_manager.save_training_data(test_data, f"test_source_{i}")
            print(f"  Added training data batch {i+1}: {doc_id}")
        
        # Show stats after adding training data
        print("\n📊 Database Statistics after adding training data:")
        db_manager.get_database_stats()
        
        # Test 2: Add predictions and see cleanup in action
        print("\n🧪 Test 2: Adding predictions...")
        for i in range(10):
            test_prediction = create_test_prediction(datetime.utcnow() - timedelta(hours=i))
            doc_id = db_manager.save_prediction(test_prediction, f"test_model_v{i}")
            print(f"  Added prediction {i+1}: {doc_id}")
        
        # Show stats after adding predictions
        print("\n📊 Database Statistics after adding predictions:")
        db_manager.get_database_stats()
        
        # Test 3: Manual cleanup
        print("\n🧪 Test 3: Manual cleanup...")
        db_manager.manual_cleanup_by_count()
        
        # Test 4: Add more data to trigger automatic cleanup
        print("\n🧪 Test 4: Adding more data to trigger automatic cleanup...")
        for i in range(3):
            test_data = create_test_data(50, datetime.utcnow() - timedelta(hours=i))
            doc_id = db_manager.save_training_data(test_data, f"auto_cleanup_test_{i}")
            print(f"  Added training data batch {i+1}: {doc_id}")
        
        # Final stats
        print("\n📊 Final Database Statistics:")
        final_stats = db_manager.get_database_stats()
        
        print("\n✅ Cleanup functionality test completed!")
        print("💡 Key features demonstrated:")
        print("  - Automatic cleanup after each save operation")
        print("  - Manual cleanup capability")
        print("  - Configurable limits via environment variables")
        print("  - Maintains minimum required records")
        
        # Close connection
        db_manager.close_connection()
        
    except Exception as e:
        print(f"\n❌ Test failed: {str(e)}")
        print("\n💡 Make sure MongoDB is running and accessible")

def test_environment_configuration():
    """Test different environment configurations."""
    print("\n🧪 Testing Environment Configuration")
    print("=" * 50)
    
    try:
        # Test with custom limits
        os.environ['MAX_TRAINING_RECORDS'] = '5'
        os.environ['MAX_PREDICTION_RECORDS'] = '3'
        
        db_manager = DatabaseManager(crypto_symbol='BTC')
        
        print(f"📊 Using custom limits - Training: {os.getenv('MAX_TRAINING_RECORDS')}, Predictions: {os.getenv('MAX_PREDICTION_RECORDS')}")
        
        # Add some data
        for i in range(10):
            test_data = create_test_data(10, datetime.utcnow() - timedelta(hours=i))
            db_manager.save_training_data(test_data, f"env_test_{i}")
            
            test_prediction = create_test_prediction(datetime.utcnow() - timedelta(hours=i))
            db_manager.save_prediction(test_prediction, f"env_test_model_{i}")
        
        # Show stats
        print("\n📊 Database Statistics with custom limits:")
        db_manager.get_database_stats()
        
        # Manual cleanup
        print("\n🧹 Manual cleanup with custom limits:")
        db_manager.manual_cleanup_by_count()
        
        db_manager.close_connection()
        
    except Exception as e:
        print(f"\n❌ Environment test failed: {str(e)}")

if __name__ == "__main__":
    print("🚀 Database Cleanup Test Suite")
    print("This script demonstrates the new cleanup functionality")
    
    # Run tests
    test_cleanup_functionality()
    test_environment_configuration()
    
    print("\n🎉 All tests completed!")
    print("\n💡 To use in production:")
    print("  1. Set MAX_TRAINING_RECORDS and MAX_PREDICTION_RECORDS in your .env file")
    print("  2. Cleanup happens automatically after each save operation")
    print("  3. Use manual_cleanup_by_count() for manual cleanup")
    print("  4. Monitor usage with get_database_stats()") 
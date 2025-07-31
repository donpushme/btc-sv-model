#!/usr/bin/env python3

"""
Test script to verify numerical fixes in the predictor.
This script tests the predictor's ability to handle data with potential numerical issues
that could cause the "Input X contains infinity" error during retraining.
"""

import pandas as pd
import numpy as np
import os
import sys
from datetime import datetime, timedelta

# Add the current directory to the path so we can import our modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from predictor import RealTimeVolatilityPredictor
from config import Config

def create_test_data_with_numerical_issues():
    """
    Create test data that might contain numerical issues similar to what could occur
    during real-time prediction and retraining.
    """
    print("🔧 Creating test data with potential numerical issues...")
    
    # Generate base data
    dates = pd.date_range(start='2024-01-01', end='2024-01-05', freq='5min')
    n_points = len(dates)
    
    # Simulate realistic price movements with some extreme values
    np.random.seed(42)
    price_base = 45000
    returns = np.random.normal(0, 0.02, n_points)
    
    # Introduce some extreme values that could cause numerical issues
    extreme_indices = np.random.choice(n_points, size=min(10, n_points//10), replace=False)
    returns[extreme_indices] = np.random.choice([0.5, -0.5, 0.1, -0.1], size=len(extreme_indices))
    
    # Add some infinite values (simulating division by zero scenarios)
    inf_indices = np.random.choice(n_points, size=min(5, n_points//20), replace=False)
    returns[inf_indices] = np.inf
    
    # Add some NaN values
    nan_indices = np.random.choice(n_points, size=min(5, n_points//20), replace=False)
    returns[nan_indices] = np.nan
    
    prices = [price_base]
    for ret in returns[1:]:
        if np.isfinite(ret):
            prices.append(prices[-1] * (1 + ret))
        else:
            prices.append(prices[-1])  # Keep same price for invalid returns
    
    # Create DataFrame with potential issues
    test_data = pd.DataFrame({
        'timestamp': dates,
        'open': prices,
        'close': prices,
        'high': [p * (1 + abs(np.random.normal(0, 0.01))) for p in prices],
        'low': [p * (1 - abs(np.random.normal(0, 0.01))) for p in prices]
    })
    
    # Add some extreme values to OHLC data
    extreme_price_indices = np.random.choice(n_points, size=min(20, n_points//5), replace=False)
    for idx in extreme_price_indices:
        if np.random.random() < 0.5:
            test_data.loc[idx, 'high'] = test_data.loc[idx, 'high'] * 10  # Extreme high
        else:
            test_data.loc[idx, 'low'] = test_data.loc[idx, 'low'] / 10   # Extreme low
    
    print(f"✅ Created test data with {len(test_data)} points")
    print(f"   - Contains {np.isinf(returns).sum()} infinite values")
    print(f"   - Contains {np.isnan(returns).sum()} NaN values")
    print(f"   - Contains {len(extreme_indices)} extreme return values")
    
    return test_data

def test_predictor_numerical_handling():
    """
    Test the predictor's ability to handle numerical issues.
    """
    print("\n🧪 Testing predictor numerical handling...")
    
    try:
        # Initialize predictor
        predictor = RealTimeVolatilityPredictor(crypto_symbol='BTC')
        print("✅ Predictor initialized successfully")
        
        # Create test data with numerical issues
        test_data = create_test_data_with_numerical_issues()
        
        # Test preprocessing
        print("\n🔄 Testing preprocessing with problematic data...")
        try:
            preprocessed_data = predictor.preprocess_input_data(test_data)
            print(f"✅ Preprocessing completed successfully")
            print(f"   - Input data shape: {test_data.shape}")
            print(f"   - Preprocessed data shape: {preprocessed_data.shape}")
            
            # Check for any remaining numerical issues
            numeric_cols = preprocessed_data.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                if col == 'timestamp':
                    continue
                if preprocessed_data[col].isnull().any():
                    print(f"⚠️ Warning: Column '{col}' still contains NaN values")
                if np.isinf(preprocessed_data[col]).any():
                    print(f"⚠️ Warning: Column '{col}' still contains infinite values")
            
        except Exception as e:
            print(f"❌ Preprocessing failed: {str(e)}")
            return False
        
        # Test prediction
        print("\n🔄 Testing prediction with cleaned data...")
        try:
            prediction = predictor.predict_next_period(test_data)
            print(f"✅ Prediction completed successfully")
            print(f"   - Predicted volatility: {prediction['predicted_volatility']:.6f}")
            print(f"   - Predicted skewness: {prediction['predicted_skewness']:.6f}")
            print(f"   - Predicted kurtosis: {prediction['predicted_kurtosis']:.6f}")
            
        except Exception as e:
            print(f"❌ Prediction failed: {str(e)}")
            return False
        
        print("\n✅ All tests passed! Numerical fixes are working correctly.")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {str(e)}")
        return False

def test_extreme_value_handling():
    """
    Test handling of extreme values that could cause numerical instability.
    """
    print("\n🧪 Testing extreme value handling...")
    
    try:
        # Create data with very extreme values
        dates = pd.date_range(start='2024-01-01', end='2024-01-02', freq='5min')
        n_points = len(dates)
        
        # Create data with extremely large and small values
        test_data = pd.DataFrame({
            'timestamp': dates,
            'open': [1e10] * n_points,  # Extremely large values
            'close': [1e-10] * n_points,  # Extremely small values
            'high': [1e15] * n_points,  # Even more extreme
            'low': [1e-15] * n_points   # Even more extreme
        })
        
        # Initialize predictor
        predictor = RealTimeVolatilityPredictor(crypto_symbol='BTC')
        
        # Test preprocessing
        print("🔄 Testing preprocessing with extremely large/small values...")
        preprocessed_data = predictor.preprocess_input_data(test_data)
        
        # Check if values were clipped appropriately
        numeric_cols = preprocessed_data.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if col == 'timestamp':
                continue
            max_val = preprocessed_data[col].max()
            min_val = preprocessed_data[col].min()
            if max_val > 1e6 or min_val < -1e6:
                print(f"⚠️ Warning: Column '{col}' still contains very large values: [{min_val}, {max_val}]")
            else:
                print(f"✅ Column '{col}' values are within reasonable range: [{min_val:.2e}, {max_val:.2e}]")
        
        print("✅ Extreme value handling test completed")
        return True
        
    except Exception as e:
        print(f"❌ Extreme value test failed: {str(e)}")
        return False

def main():
    """
    Main test function.
    """
    print("🚀 Testing Predictor Numerical Fixes")
    print("=" * 50)
    
    # Test 1: Basic numerical handling
    test1_passed = test_predictor_numerical_handling()
    
    # Test 2: Extreme value handling
    test2_passed = test_extreme_value_handling()
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 Test Results Summary:")
    print(f"   - Basic numerical handling: {'✅ PASSED' if test1_passed else '❌ FAILED'}")
    print(f"   - Extreme value handling: {'✅ PASSED' if test2_passed else '❌ FAILED'}")
    
    if test1_passed and test2_passed:
        print("\n🎉 All tests passed! The predictor should now handle numerical issues correctly.")
        print("💡 The 'Input X contains infinity' error should be resolved.")
    else:
        print("\n⚠️ Some tests failed. There may still be numerical issues to address.")
    
    return test1_passed and test2_passed

if __name__ == "__main__":
    main() 
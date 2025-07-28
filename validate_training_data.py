#!/usr/bin/env python3
"""
Training Data Validation Script

This script helps validate your training data before retraining the model
to ensure the kurtosis prediction fixes will work properly.
"""

import os
import sys
import pandas as pd
import numpy as np
from utils import analyze_training_data_quality, validate_prediction_bounds

def main():
    print("🔍 Bitcoin Volatility Model - Training Data Validator")
    print("=" * 60)
    
    # Check if training data exists
    training_data_path = "training_data/bitcoin_5min.csv"
    
    if not os.path.exists(training_data_path):
        print(f"❌ Training data not found at: {training_data_path}")
        print("Please ensure your training data is in the correct location.")
        return
    
    print(f"📁 Found training data: {training_data_path}")
    
    # Analyze training data quality
    print("\n" + "="*60)
    analysis = analyze_training_data_quality(training_data_path)
    
    if 'error' in analysis:
        print(f"❌ Error: {analysis['error']}")
        return
    
    # Check if data looks reasonable for retraining
    print("\n" + "="*60)
    print("🎯 Retraining Readiness Check:")
    
    kurtosis_stats = analysis['target_kurtosis']
    
    # Check for extreme kurtosis values
    extreme_count = kurtosis_stats['outliers_5std']
    total_count = kurtosis_stats['count']
    
    if extreme_count > total_count * 0.05:  # More than 5% extreme values
        print("⚠️  Warning: High number of extreme kurtosis values detected")
        print("   The new bounds and log transformation should help with this")
    else:
        print("✅ Kurtosis values look reasonable")
    
    # Check kurtosis range
    kurt_min, kurt_max = kurtosis_stats['min'], kurtosis_stats['max']
    if kurt_max > 30:
        print("⚠️  Warning: Some kurtosis values exceed 30 (will be capped)")
    else:
        print("✅ Kurtosis range is within reasonable bounds")
    
    # Check data size
    if total_count < 10000:
        print("⚠️  Warning: Relatively small dataset for training")
    else:
        print(f"✅ Dataset size looks good ({total_count:,} samples)")
    
    print("\n" + "="*60)
    print("🚀 Ready for Retraining!")
    print("\nThe following fixes have been implemented:")
    print("✅ Log transformation for kurtosis values")
    print("✅ RobustScaler instead of StandardScaler for targets")
    print("✅ Validation bounds (3-30 absolute kurtosis)")
    print("✅ Huber loss for kurtosis prediction")
    print("✅ Prediction validation in real-time")
    
    print("\n📋 Next Steps:")
    print("1. Run: python trainer.py")
    print("2. Monitor training progress")
    print("3. Check that kurtosis predictions are reasonable")
    print("4. Test with: python continuous_predictor.py")
    
    print("\n💡 Expected Improvements:")
    print("- Kurtosis predictions should be between -1 and +10 (excess kurtosis)")
    print("- More stable training due to log transformation")
    print("- Better handling of outliers with RobustScaler")
    print("- No more extreme values like 83.18")

if __name__ == "__main__":
    main() 
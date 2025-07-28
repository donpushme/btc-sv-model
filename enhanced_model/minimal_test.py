#!/usr/bin/env python3
"""
Minimal test script
"""

import pandas as pd
import os

def main():
    print("🧪 Minimal test...")
    
    # Check if file exists
    csv_path = '../training_data/bitcoin_5min.csv'
    print(f"📁 Checking file: {csv_path}")
    print(f"File exists: {os.path.exists(csv_path)}")
    
    if os.path.exists(csv_path):
        # Try to load the CSV
        print("📊 Loading CSV...")
        df = pd.read_csv(csv_path)
        print(f"✅ CSV loaded. Shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        print(f"First few rows:")
        print(df.head())
    else:
        print("❌ File not found")

if __name__ == "__main__":
    main()
#!/usr/bin/env python3

"""
Test Script for Pyth Network API Integration
Demonstrates how to fetch Bitcoin price data using the new API.
"""

import requests
import pandas as pd
import time
from datetime import datetime, timedelta

def test_pyth_api():
    """Test the Pyth Network API for Bitcoin price data."""
    
    print("🧪 Testing Pyth Network API for Bitcoin Price Data")
    print("=" * 60)
    
    try:
        # API configuration
        base_url = "https://benchmarks.pyth.network/v1/shims/tradingview/history"
        symbol = "Crypto.BTC/USD"
        
        # Test 1: Get current Bitcoin price
        print("\n📈 Test 1: Getting Current Bitcoin Price")
        print("-" * 40)
        
        current_time = int(time.time())
        current_url = f"{base_url}?symbol={symbol}&resolution=1&from={current_time-300}&to={current_time}"
        
        print(f"🔗 API URL: {current_url}")
        
        response = requests.get(current_url, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        print(f"📊 API Response: {data}")
        
        if data.get('s') == 'ok' and data.get('c'):
            current_price = data['c'][-1]
            current_timestamp = data['t'][-1]
            current_time_dt = datetime.fromtimestamp(current_timestamp)
            
            print(f"✅ Success!")
            print(f"💰 Current Bitcoin Price: ${current_price:,.2f}")
            print(f"🕐 Timestamp: {current_time_dt}")
            print(f"📡 Source: Pyth Network")
        else:
            print(f"❌ Failed: API returned status '{data.get('s')}'")
            return False
        
        # Test 2: Get historical data (last 24 hours)
        print("\n📊 Test 2: Getting Historical Bitcoin Data (24 hours)")
        print("-" * 50)
        
        end_time = int(time.time())
        start_time = end_time - (24 * 3600)  # 24 hours ago
        
        history_url = f"{base_url}?symbol={symbol}&resolution=5&from={start_time}&to={end_time}"
        print(f"🔗 API URL: {history_url}")
        
        response = requests.get(history_url, timeout=15)
        response.raise_for_status()
        data = response.json()
        
        if data.get('s') == 'ok' and data.get('t'):
            timestamps = data['t']
            opens = data['o']
            highs = data['h']
            lows = data['l']
            closes = data['c']
            
            # Create DataFrame
            df = pd.DataFrame({
                'timestamp': [datetime.fromtimestamp(ts) for ts in timestamps],
                'open': opens,
                'high': highs,
                'low': lows,
                'close': closes
            })
            
            print(f"✅ Success!")
            print(f"📈 Retrieved {len(df)} data points")
            print(f"📅 Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
            print(f"💰 Price range: ${df['close'].min():,.2f} - ${df['close'].max():,.2f}")
            print(f"📊 Latest price: ${df['close'].iloc[-1]:,.2f}")
            
            # Show sample data
            print(f"\n📋 Sample Data (last 5 points):")
            print(df.tail()[['timestamp', 'open', 'high', 'low', 'close']].to_string(index=False))
            
        else:
            print(f"❌ Failed: API returned status '{data.get('s')}'")
            return False
        
        # Test 3: Data validation
        print("\n✅ Test 3: Data Validation")
        print("-" * 30)
        
        # Check data quality
        missing_values = df.isnull().sum()
        if missing_values.any():
            print(f"⚠️ Missing values found: {missing_values.to_dict()}")
        else:
            print("✅ No missing values")
        
        # Check price consistency
        valid_prices = (df['low'] <= df['close']) & (df['close'] <= df['high'])
        if valid_prices.all():
            print("✅ Price data is consistent (low <= close <= high)")
        else:
            print(f"⚠️ Found {(~valid_prices).sum()} inconsistent price entries")
        
        # Check chronological order
        if df['timestamp'].is_monotonic_increasing:
            print("✅ Timestamps are in chronological order")
        else:
            print("⚠️ Timestamps are not in chronological order")
        
        print(f"\n🎉 All tests completed successfully!")
        print(f"✅ Pyth Network API is working correctly for Bitcoin price data")
        
        return True
        
    except requests.exceptions.RequestException as e:
        print(f"❌ Network Error: {str(e)}")
        return False
    except Exception as e:
        print(f"❌ Unexpected Error: {str(e)}")
        return False

def demo_api_formats():
    """Demonstrate different API response formats."""
    
    print("\n🔍 API Response Format Demonstration")
    print("=" * 50)
    
    # Show the expected response format
    example_response = {
        "s": "ok",
        "t": [1753274220, 1753274280, 1753274340],
        "o": [42350.50, 42355.75, 42360.25],
        "h": [42375.00, 42380.50, 42385.75],
        "l": [42340.25, 42350.00, 42355.50],
        "c": [42360.75, 42365.25, 42370.50],
        "v": [0.0, 0.0, 0.0]
    }
    
    print("📋 Expected API Response Format:")
    print(f"   s: Status ('ok' for success)")
    print(f"   t: Timestamps (Unix timestamps)")
    print(f"   o: Open prices")
    print(f"   h: High prices")
    print(f"   l: Low prices")
    print(f"   c: Close prices")
    print(f"   v: Volume (not used in our system)")
    
    print(f"\n📊 Example Response:")
    for key, value in example_response.items():
        print(f"   {key}: {value}")

if __name__ == "__main__":
    demo_api_formats()
    
    print(f"\n" + "="*60)
    test_success = test_pyth_api()
    
    if test_success:
        print(f"\n🚀 Ready to use Pyth Network API in continuous predictor!")
        print(f"💡 Run: python continuous_predictor.py")
    else:
        print(f"\n🔧 Please check your internet connection and API access") 
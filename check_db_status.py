#!/usr/bin/env python3

"""
Simple script to check database status and training data availability
Run this on your other device to diagnose the data retrieval issue
"""

from database_manager import DatabaseManager

def main():
    print("🔍 Checking Database Status")
    print("=" * 40)
    
    try:
        # Initialize database manager
        db_manager = DatabaseManager()
        print("✅ Connected to database")
        
        # Get database stats
        print("\n📊 Database Statistics:")
        stats = db_manager.get_database_stats()
        
        # Check training data availability
        print("\n📊 Training Data Availability:")
        availability = db_manager.check_training_data_availability()
        
        # Test data retrieval
        print("\n🧪 Testing Data Retrieval:")
        
        # Test with different time ranges
        for hours in [1, 24, 168, 720]:
            print(f"\n  Testing {hours} hours:")
            df = db_manager.get_training_data_for_update(hours=hours, fallback_to_all=True)
            print(f"    Retrieved: {len(df)} rows")
            if len(df) > 0:
                print(f"    Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
                print(f"    Columns: {list(df.columns)}")
        
        print("\n" + "=" * 40)
        print("✅ Check complete")
        
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main() 
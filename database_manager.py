#!/usr/bin/env python3

"""
MongoDB Database Manager for Multi-Crypto Volatility Prediction System
Handles storage of predictions, training data, and model metadata for multiple cryptocurrencies.
"""

import os
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import pymongo
from pymongo import MongoClient, ASCENDING, DESCENDING
from bson import ObjectId
import warnings
import dotenv
from config import Config
dotenv.load_dotenv()

class DatabaseManager:
    """
    Manages MongoDB operations for the multi-crypto volatility prediction system.
    """
    
    def __init__(self, crypto_symbol: str = 'BTC', connection_string: str = None, database_name: str = "synth_prediction"):
        """
        Initialize database connection for a specific cryptocurrency.
        
        Args:
            crypto_symbol: Cryptocurrency symbol (BTC, ETH, XAU, SOL)
            connection_string: MongoDB connection string
            database_name: Name of the database to use
        """
        # Validate crypto symbol
        if crypto_symbol not in Config.SUPPORTED_CRYPTOS:
            raise ValueError(f"Unsupported crypto symbol: {crypto_symbol}. Supported: {list(Config.SUPPORTED_CRYPTOS.keys())}")
        
        self.crypto_symbol = crypto_symbol
        self.crypto_config = Config.SUPPORTED_CRYPTOS[crypto_symbol]
        
        # Default connection string for local MongoDB
        if connection_string is None:
            connection_string = os.getenv('MONGODB_URI', 'mongodb://localhost:27017/')
        
        try:
            self.client = MongoClient(connection_string)
            self.db = self.client[database_name]
            
            # Test connection
            self.client.admin.command('ping')
            print(f"✅ Connected to MongoDB database: {database_name} for {self.crypto_config['name']} ({crypto_symbol})")
            
            # Initialize collections
            self._init_collections()
            
        except Exception as e:
            print(f"❌ Failed to connect to MongoDB: {str(e)}")
            print("💡 Tip: Make sure MongoDB is running and accessible")
            raise
    
    def _init_collections(self):
        """Initialize collections and create indexes for the specific cryptocurrency."""
        # Collection names - ALL crypto-specific
        self.predictions_collection = self.db[self.crypto_config['db_table']]
        self.training_data_collection = self.db[f"{self.crypto_symbol.lower()}_training_data"]
        self.models_collection = self.db[f"{self.crypto_symbol.lower()}_models"]
        self.performance_collection = self.db[f"{self.crypto_symbol.lower()}_performance"]
        
        # Create indexes for better query performance
        try:
            # Predictions indexes
            self.predictions_collection.create_index([
                ("timestamp", DESCENDING),
                ("prediction_timestamp", DESCENDING),
                ("crypto_symbol", ASCENDING)
            ])
            
            # Training data indexes
            self.training_data_collection.create_index([
                ("timestamp", DESCENDING),
                ("crypto_symbol", ASCENDING)
            ])
            
            # Models indexes
            self.models_collection.create_index([
                ("created_at", DESCENDING),
                ("model_version", DESCENDING),
                ("crypto_symbol", ASCENDING)
            ])
            
            # Performance indexes
            self.performance_collection.create_index([
                ("timestamp", DESCENDING),
                ("model_version", DESCENDING),
                ("crypto_symbol", ASCENDING)
            ])
            
        except Exception as e:
            print(f"⚠️ Warning: Failed to create indexes: {str(e)}")
    
    def cleanup_old_records_by_count(self, max_training_records: int = None, max_prediction_records: int = None):
        """
        Clean up old records based on count limits while maintaining minimum records.
        Deletes oldest records when maximum count is exceeded.
        
        Args:
            max_training_records: Maximum number of training data records to keep (default from env)
            max_prediction_records: Maximum number of prediction records to keep (default from env)
        """
        try:
            # Get limits from environment variables or use defaults
            if max_training_records is None:
                max_training_records = int(os.getenv('MAX_TRAINING_RECORDS', '1000'))
            if max_prediction_records is None:
                max_prediction_records = int(os.getenv('MAX_PREDICTION_RECORDS', '400'))
            
            # Cleanup training data records
            training_count = self.training_data_collection.count_documents({})
            if training_count > max_training_records:
                # Find the oldest records to delete
                excess_count = training_count - max_training_records
                oldest_records = self.training_data_collection.find().sort("saved_at", ASCENDING).limit(excess_count)
                oldest_ids = [record["_id"] for record in oldest_records]
                
                if oldest_ids:
                    result = self.training_data_collection.delete_many({"_id": {"$in": oldest_ids}})
                    print(f"🧹 Cleaned up {result.deleted_count} old training data records (kept {max_training_records})")
            
            # Cleanup prediction records
            prediction_count = self.predictions_collection.count_documents({})
            if prediction_count > max_prediction_records:
                # Find the oldest records to delete
                excess_count = prediction_count - max_prediction_records
                oldest_records = self.predictions_collection.find().sort("prediction_timestamp", ASCENDING).limit(excess_count)
                oldest_ids = [record["_id"] for record in oldest_records]
                
                if oldest_ids:
                    result = self.predictions_collection.delete_many({"_id": {"$in": oldest_ids}})
                    print(f"🧹 Cleaned up {result.deleted_count} old prediction records (kept {max_prediction_records})")
                    
        except Exception as e:
            print(f"❌ Failed to cleanup old records by count: {str(e)}")

    def save_prediction(self, prediction: Dict, model_version: str = None) -> str:
        """
        Save a prediction to the database.
        
        Args:
            prediction: Prediction dictionary from predictor
            model_version: Version of the model used
            
        Returns:
            str: Document ID of saved prediction
        """
        try:
            # Check if this is a batch record (continuous predictions)
            if prediction.get('prediction_type') == 'continuous_batch':
                return self.save_prediction_batch(prediction, model_version)
            
            # Handle individual predictions (original format)
            doc = {
                "prediction_timestamp": datetime.utcnow(),
                "data_timestamp": pd.to_datetime(prediction['timestamp']),
                "model_version": model_version or "unknown",
                "current_price": float(prediction['current_price']),
                "predicted_volatility": float(prediction['predicted_volatility']),
                "predicted_skewness": float(prediction['predicted_skewness']),
                "predicted_kurtosis": float(prediction['predicted_kurtosis']),
                "volatility_annualized": float(prediction['volatility_annualized']),
                "confidence_interval_lower": float(prediction['confidence_interval_lower']),
                "confidence_interval_upper": float(prediction['confidence_interval_upper']),
                "market_regime": prediction['market_regime'],
                "risk_assessment": prediction['risk_assessment'],
                "prediction_period": prediction['prediction_period'],
                "crypto_symbol": self.crypto_symbol
            }
            
            # Insert document
            result = self.predictions_collection.insert_one(doc)
            
            # Cleanup old records after saving new one
            self.cleanup_old_records_by_count()
            
            return str(result.inserted_id)
            
        except Exception as e:
            print(f"❌ Failed to save prediction: {str(e)}")
            raise
    
    def save_prediction_batch(self, batch_record: Dict, model_version: str = None) -> str:
        """
        Save a batch of 288 predictions as a single database record.
        
        Args:
            batch_record: Batch record containing all 288 predictions
            model_version: Version of the model used
            
        Returns:
            str: Document ID of saved batch record
        """
        try:
            # Prepare batch document
            doc = {
                "prediction_timestamp": datetime.utcnow(),
                "data_timestamp": pd.to_datetime(batch_record['data_timestamp']),
                "model_version": model_version or "unknown",
                "batch_id": batch_record['batch_id'],
                "prediction_type": batch_record['prediction_type'],
                "current_price": float(batch_record['current_price']),
                "predictions_count": int(batch_record['predictions_count']),
                "interval_minutes": int(batch_record.get('interval_minutes', 5)),
                "prediction_horizon_hours": int(batch_record.get('prediction_horizon_hours', 24)),
                "source": batch_record.get('source', 'unknown'),
                "summary_stats": batch_record['summary_stats'],
                "predictions": batch_record['predictions'],  # Array of all 288 predictions
                "crypto_symbol": self.crypto_symbol
            }
            
            # Insert document
            result = self.predictions_collection.insert_one(doc)
            print(f"💾 Saved prediction batch to database: {result.inserted_id}")
            
            # Cleanup old records after saving new one
            self.cleanup_old_records_by_count()
            
            return str(result.inserted_id)
            
        except Exception as e:
            print(f"❌ Failed to save prediction batch: {str(e)}")
            raise
    
    def save_training_data(self, price_data: pd.DataFrame, data_source: str = "realtime") -> str:
        """
        Save training data to the database.
        
        Args:
            price_data: Bitcoin price data DataFrame
            data_source: Source of the data (e.g., "realtime", "historical", "yfinance")
            
        Returns:
            str: Document ID of saved data
        """
        try:
            # Convert DataFrame to records
            records = price_data.to_dict('records')
            
            # Prepare document
            doc = {
                "saved_at": datetime.utcnow(),
                "data_source": data_source,
                "data_count": len(records),
                "date_range": {
                    "start": pd.to_datetime(price_data['timestamp'].min()),
                    "end": pd.to_datetime(price_data['timestamp'].max())
                },
                "data": records,
                "crypto_symbol": self.crypto_symbol
            }
            
            # Insert document
            result = self.training_data_collection.insert_one(doc)
            print(f"💾 Saved {len(records)} training data records: {result.inserted_id}")
            
            # Cleanup old records after saving new one
            self.cleanup_old_records_by_count()
            
            return str(result.inserted_id)
            
        except Exception as e:
            print(f"❌ Failed to save training data: {str(e)}")
            raise
    
    def save_model_metadata(self, model_path: str, training_metrics: Dict, 
                           config: Dict, model_version: str = None) -> str:
        """
        Save model metadata and training results.
        
        Args:
            model_path: Path to saved model file
            training_metrics: Training metrics dictionary
            config: Model configuration
            model_version: Version identifier
            
        Returns:
            str: Document ID of saved metadata
        """
        try:
            if model_version is None:
                model_version = f"{self.crypto_symbol}_model"
            
            # Prepare document
            doc = {
                "model_version": model_version,
                "created_at": datetime.utcnow(),
                "model_path": model_path,
                "config": config,
                "training_metrics": training_metrics,
                "status": "active",
                "crypto_symbol": self.crypto_symbol
            }
            
            # Mark previous models as inactive
            self.models_collection.update_many(
                {"status": "active"},
                {"$set": {"status": "inactive"}}
            )
            
            # Insert new model
            result = self.models_collection.insert_one(doc)
            print(f"💾 Saved model metadata: {model_version} ({result.inserted_id})")
            
            return str(result.inserted_id)
            
        except Exception as e:
            print(f"❌ Failed to save model metadata: {str(e)}")
            raise
    
    def get_recent_predictions(self, hours: int = 24, limit: int = 1000) -> pd.DataFrame:
        """
        Get recent predictions from the database.
        
        Args:
            hours: Number of hours to look back
            limit: Maximum number of predictions to return
            
        Returns:
            pd.DataFrame: Recent predictions
        """
        try:
            cutoff_time = datetime.utcnow() - timedelta(hours=hours)
            
            cursor = self.predictions_collection.find(
                {"prediction_timestamp": {"$gte": cutoff_time}}
            ).sort("prediction_timestamp", DESCENDING).limit(limit)
            
            predictions = list(cursor)
            
            if predictions:
                df = pd.DataFrame(predictions)
                print(f"📊 Retrieved {len(df)} predictions from last {hours} hours")
                return df
            else:
                print(f"📊 No predictions found in last {hours} hours")
                return pd.DataFrame()
                
        except Exception as e:
            print(f"❌ Failed to get recent predictions: {str(e)}")
            return pd.DataFrame()
    
    def get_training_data_for_update(self, hours: int = 168, fallback_to_all: bool = True) -> pd.DataFrame:
        """
        Get recent training data for model updates.
        
        Args:
            hours: Number of hours of data to retrieve (default: 7 days)
            fallback_to_all: If no recent data found, try to get all available data
            
        Returns:
            pd.DataFrame: Combined training data
        """
        try:
            cutoff_time = datetime.utcnow() - timedelta(hours=hours)
            
            # First try to get recent training data
            cursor = self.training_data_collection.find(
                {"saved_at": {"$gte": cutoff_time}}
            ).sort("saved_at", DESCENDING)
            
            all_data = []
            docs_found = 0
            
            for doc in cursor:
                docs_found += 1
                if 'data' in doc and isinstance(doc['data'], list):
                    all_data.extend(doc['data'])
                else:
                    print(f"⚠️ Document {doc.get('_id')} missing or invalid 'data' field")
            
            print(f"📊 Found {docs_found} documents with saved_at >= {cutoff_time}")
            
            # If no recent data found and fallback is enabled, try to get all data
            if not all_data and fallback_to_all:
                print("📊 No recent data found, trying to get all available training data...")
                
                cursor = self.training_data_collection.find().sort("saved_at", DESCENDING)
                all_data = []
                docs_found = 0
                
                for doc in cursor:
                    docs_found += 1
                    if 'data' in doc and isinstance(doc['data'], list):
                        all_data.extend(doc['data'])
                    else:
                        print(f"⚠️ Document {doc.get('_id')} missing or invalid 'data' field")
                
                print(f"📊 Found {docs_found} total documents in training_data collection")
            
            if all_data:
                df = pd.DataFrame(all_data)
                
                # Ensure required columns exist
                required_cols = ['timestamp', 'open', 'close', 'high', 'low']
                missing_cols = [col for col in required_cols if col not in df.columns]
                
                if missing_cols:
                    print(f"❌ Missing required columns in training data: {missing_cols}")
                    print(f"Available columns: {list(df.columns)}")
                    return pd.DataFrame()
                
                # Convert timestamp to datetime
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                
                # Remove duplicates and sort
                df = df.drop_duplicates(subset=['timestamp']).sort_values('timestamp')
                
                # Remove any rows with invalid price data
                initial_count = len(df)
                df = df.dropna(subset=['open', 'close', 'high', 'low'])
                df = df[(df['open'] > 0) & (df['close'] > 0) & (df['high'] > 0) & (df['low'] > 0)]
                final_count = len(df)
                
                if final_count < initial_count:
                    print(f"⚠️ Removed {initial_count - final_count} rows with invalid price data")
                
                print(f"📊 Retrieved {len(df)} training records (from {len(all_data)} total records)")
                print(f"📊 Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
                print(f"📊 Columns: {list(df.columns)}")
                
                return df
            else:
                print(f"📊 No training data found in database")
                return pd.DataFrame()
                
        except Exception as e:
            print(f"❌ Failed to get training data: {str(e)}")
            return pd.DataFrame()
    
    def save_performance_metrics(self, model_version: str, metrics: Dict) -> str:
        """
        Save model performance metrics.
        
        Args:
            model_version: Version of the model
            metrics: Performance metrics dictionary
            
        Returns:
            str: Document ID
        """
        try:
            doc = {
                "model_version": model_version,
                "timestamp": datetime.utcnow(),
                "metrics": metrics,
                "crypto_symbol": self.crypto_symbol
            }
            
            result = self.performance_collection.insert_one(doc)
            print(f"📈 Saved performance metrics for model {model_version}")
            
            return str(result.inserted_id)
            
        except Exception as e:
            print(f"❌ Failed to save performance metrics: {str(e)}")
            raise
    
    def get_model_performance_history(self, model_version: str = None, 
                                    days: int = 30) -> pd.DataFrame:
        """
        Get model performance history.
        
        Args:
            model_version: Specific model version (None for all)
            days: Number of days to look back
            
        Returns:
            pd.DataFrame: Performance history
        """
        try:
            cutoff_time = datetime.utcnow() - timedelta(days=days)
            query = {"timestamp": {"$gte": cutoff_time}}
            
            if model_version:
                query["model_version"] = model_version
            
            cursor = self.performance_collection.find(query).sort("timestamp", ASCENDING)
            performance_data = list(cursor)
            
            if performance_data:
                df = pd.DataFrame(performance_data)
                print(f"📈 Retrieved {len(df)} performance records")
                return df
            else:
                print("📈 No performance data found")
                return pd.DataFrame()
                
        except Exception as e:
            print(f"❌ Failed to get performance history: {str(e)}")
            return pd.DataFrame()
    
    def cleanup_old_data(self, predictions_days: int = 90, 
                        training_data_days: int = 180,
                        performance_days: int = 365):
        """
        Clean up old data to manage database size.
        
        Args:
            predictions_days: Keep predictions for this many days
            training_data_days: Keep training data for this many days
            performance_days: Keep performance data for this many days
        """
        try:
            current_time = datetime.utcnow()
            
            # Cleanup predictions
            pred_cutoff = current_time - timedelta(days=predictions_days)
            pred_result = self.predictions_collection.delete_many(
                {"prediction_timestamp": {"$lt": pred_cutoff}}
            )
            print(f"🧹 Cleaned up {pred_result.deleted_count} old predictions")
            
            # Cleanup training data
            train_cutoff = current_time - timedelta(days=training_data_days)
            train_result = self.training_data_collection.delete_many(
                {"saved_at": {"$lt": train_cutoff}}
            )
            print(f"🧹 Cleaned up {train_result.deleted_count} old training records")
            
            # Cleanup performance data
            perf_cutoff = current_time - timedelta(days=performance_days)
            perf_result = self.performance_collection.delete_many(
                {"timestamp": {"$lt": perf_cutoff}}
            )
            print(f"🧹 Cleaned up {perf_result.deleted_count} old performance records")
            
        except Exception as e:
            print(f"❌ Failed to cleanup old data: {str(e)}")

    def manual_cleanup_by_count(self, max_training_records: int = None, max_prediction_records: int = None):
        """
        Manually trigger cleanup based on record count limits.
        
        Args:
            max_training_records: Maximum number of training data records to keep (default from env)
            max_prediction_records: Maximum number of prediction records to keep (default from env)
        """
        # Get limits from environment variables or use defaults
        if max_training_records is None:
            max_training_records = int(os.getenv('MAX_TRAINING_RECORDS', '1000'))
        if max_prediction_records is None:
            max_prediction_records = int(os.getenv('MAX_PREDICTION_RECORDS', '400'))
            
        print(f"🧹 Manual cleanup triggered - Max training: {max_training_records}, Max predictions: {max_prediction_records}")
        self.cleanup_old_records_by_count(max_training_records, max_prediction_records)
        
        # Show updated stats
        print("\n📊 Database stats after cleanup:")
        self.get_database_stats()
    
    def get_database_stats(self) -> Dict:
        """
        Get database statistics.
        
        Returns:
            Dict: Database statistics
        """
        try:
            predictions_count = self.predictions_collection.count_documents({})
            training_data_count = self.training_data_collection.count_documents({})
            
            # Get limits from environment variables
            max_training_records = int(os.getenv('MAX_TRAINING_RECORDS', '1000'))
            max_prediction_records = int(os.getenv('MAX_PREDICTION_RECORDS', '400'))
            
            stats = {
                "predictions_count": predictions_count,
                "training_data_count": training_data_count,
                "models_count": self.models_collection.count_documents({}),
                "performance_records_count": self.performance_collection.count_documents({}),
                "database_size_mb": self.db.command("dbStats")["dataSize"] / (1024 * 1024),
                "predictions_limit": max_prediction_records,
                "training_data_limit": max_training_records,
                "predictions_usage_percent": (predictions_count / max_prediction_records) * 100 if predictions_count > 0 else 0,
                "training_data_usage_percent": (training_data_count / max_training_records) * 100 if training_data_count > 0 else 0
            }
            
            print("📊 Database Statistics:")
            for key, value in stats.items():
                if "size_mb" in key:
                    print(f"  {key}: {value:.2f} MB")
                elif "usage_percent" in key:
                    print(f"  {key}: {value:.1f}%")
                elif "limit" in key:
                    print(f"  {key}: {value:,}")
                else:
                    print(f"  {key}: {value:,}")
            
            # Show cleanup status
            if predictions_count > max_prediction_records:
                print(f"  ⚠️ Predictions exceed limit: {predictions_count} > {max_prediction_records}")
            if training_data_count > max_training_records:
                print(f"  ⚠️ Training data exceeds limit: {training_data_count} > {max_training_records}")
            
            return stats
            
        except Exception as e:
            print(f"❌ Failed to get database stats: {str(e)}")
            return {}
    
    def check_training_data_availability(self) -> Dict:
        """
        Check training data availability and structure.
        
        Returns:
            Dict: Training data availability information
        """
        try:
            # Get total count
            total_docs = self.training_data_collection.count_documents({})
            
            # Get recent docs
            recent_docs = self.training_data_collection.count_documents({
                "saved_at": {"$gte": datetime.utcnow() - timedelta(hours=168)}
            })
            
            # Check for documents with data field
            docs_with_data = self.training_data_collection.count_documents({
                "data": {"$exists": True}
            })
            
            # Check for documents with valid data arrays
            docs_with_valid_data = self.training_data_collection.count_documents({
                "data": {"$type": "array", "$ne": []}
            })
            
            # Get oldest and newest saved_at timestamps
            oldest_doc = self.training_data_collection.find_one(
                {}, {"saved_at": 1}, sort=[("saved_at", 1)]
            )
            newest_doc = self.training_data_collection.find_one(
                {}, {"saved_at": 1}, sort=[("saved_at", -1)]
            )
            
            info = {
                "total_documents": total_docs,
                "recent_documents_7d": recent_docs,
                "documents_with_data_field": docs_with_data,
                "documents_with_valid_data": docs_with_valid_data,
                "oldest_saved_at": oldest_doc.get("saved_at") if oldest_doc else None,
                "newest_saved_at": newest_doc.get("saved_at") if newest_doc else None
            }
            
            print("📊 Training Data Availability:")
            for key, value in info.items():
                if "saved_at" in key:
                    print(f"  {key}: {value}")
                else:
                    print(f"  {key}: {value}")
            
            return info
            
        except Exception as e:
            print(f"❌ Failed to check training data availability: {str(e)}")
            return {}
    
    def close_connection(self):
        """Close database connection."""
        try:
            self.client.close()
            print("🔐 Database connection closed")
        except Exception as e:
            print(f"❌ Error closing connection: {str(e)}")


def create_database_config():
    """Create a .env file with database configuration."""
    config_content = """# MongoDB Configuration
MONGODB_URI=mongodb://localhost:27017/synth_prediction
DATABASE_NAME=synth_prediction

# Optional: MongoDB Atlas connection (comment out local URI above)
# MONGODB_URI=mongodb+srv://username:password@cluster.mongodb.net/synth_prediction

# Collection Settings
PREDICTION_RETENTION_DAYS=90
TRAINING_DATA_RETENTION_DAYS=180
PERFORMANCE_DATA_RETENTION_DAYS=365

# Record Count Limits (for automatic cleanup)
MAX_TRAINING_RECORDS=1000
MAX_PREDICTION_RECORDS=400

# Online Learning Settings
RETRAIN_INTERVAL_HOURS=24
MIN_NEW_DATA_POINTS=288
PERFORMANCE_THRESHOLD=0.05
"""
    
    try:
        with open('.env', 'w') as f:
            f.write(config_content)
        print("📝 Created .env file with database configuration")
        print("💡 Edit .env file to configure your MongoDB connection")
    except Exception as e:
        print(f"❌ Failed to create .env file: {str(e)}")


if __name__ == "__main__":
    # Demo usage
    print("🚀 Testing Database Manager...")
    
    # Create config file
    create_database_config()
    
    try:
        # Initialize database manager
        db_manager = DatabaseManager()
        
        # Get database statistics
        print("\n📊 Current Database Statistics:")
        stats = db_manager.get_database_stats()
        
        # Test cleanup functionality
        print("\n🧹 Testing cleanup functionality...")
        db_manager.manual_cleanup_by_count()
        
        # Test basic operations
        print("\n✅ Database manager is working correctly!")
        print("💡 Features:")
        print("  - Automatic cleanup after each save operation")
        print("  - Maintains max 1000 training records and 400 prediction records")
        print("  - Manual cleanup available via manual_cleanup_by_count()")
        print("  - You can now use it with the real-time predictor")
        
        # Close connection
        db_manager.close_connection()
        
    except Exception as e:
        print(f"\n❌ Database test failed: {str(e)}")
        print("\n💡 Troubleshooting tips:")
        print("  1. Make sure MongoDB is installed and running")
        print("  2. Check your connection string in .env file")
        print("  3. For Windows: Install MongoDB Community Server")
        print("  4. For cloud: Use MongoDB Atlas and update MONGODB_URI") 
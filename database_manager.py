#!/usr/bin/env python3

"""
MongoDB Database Manager for Bitcoin Volatility Prediction System
Handles storage of predictions, training data, and model metadata.
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

class DatabaseManager:
    """
    Manages MongoDB operations for the Bitcoin volatility prediction system.
    """
    
    def __init__(self, connection_string: str = None, database_name: str = "bitcoin_volatility"):
        """
        Initialize database connection.
        
        Args:
            connection_string: MongoDB connection string
            database_name: Name of the database to use
        """
        # Default connection string for local MongoDB
        if connection_string is None:
            connection_string = os.getenv('MONGODB_URI', 'mongodb://localhost:27017/')
        
        try:
            self.client = MongoClient(connection_string)
            self.db = self.client[database_name]
            
            # Test connection
            self.client.admin.command('ping')
            print(f"✅ Connected to MongoDB database: {database_name}")
            
            # Initialize collections
            self._init_collections()
            
        except Exception as e:
            print(f"❌ Failed to connect to MongoDB: {str(e)}")
            print("💡 Tip: Make sure MongoDB is running and accessible")
            raise
    
    def _init_collections(self):
        """Initialize collections and create indexes."""
        # Collection names
        self.predictions_collection = self.db.predictions
        self.training_data_collection = self.db.training_data
        self.models_collection = self.db.models
        self.performance_collection = self.db.performance
        
        # Create indexes for better query performance
        try:
            # Predictions indexes
            self.predictions_collection.create_index([
                ("timestamp", DESCENDING),
                ("prediction_timestamp", DESCENDING)
            ])
            
            # Training data indexes
            self.training_data_collection.create_index([
                ("timestamp", DESCENDING)
            ])
            
            # Models indexes
            self.models_collection.create_index([
                ("created_at", DESCENDING),
                ("model_version", DESCENDING)
            ])
            
            # Performance indexes
            self.performance_collection.create_index([
                ("timestamp", DESCENDING),
                ("model_version", DESCENDING)
            ])
            
        except Exception as e:
            print(f"⚠️ Warning: Failed to create indexes: {str(e)}")
    
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
            # Prepare document
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
                "prediction_period": prediction['prediction_period']
            }
            
            # Insert document
            result = self.predictions_collection.insert_one(doc)
            print(f"💾 Saved prediction to database: {result.inserted_id}")
            
            return str(result.inserted_id)
            
        except Exception as e:
            print(f"❌ Failed to save prediction: {str(e)}")
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
                "data": records
            }
            
            # Insert document
            result = self.training_data_collection.insert_one(doc)
            print(f"💾 Saved {len(records)} training data records: {result.inserted_id}")
            
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
                model_version = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            
            # Prepare document
            doc = {
                "model_version": model_version,
                "created_at": datetime.utcnow(),
                "model_path": model_path,
                "config": config,
                "training_metrics": training_metrics,
                "status": "active"
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
    
    def get_training_data_for_update(self, hours: int = 168) -> pd.DataFrame:
        """
        Get recent training data for model updates.
        
        Args:
            hours: Number of hours of data to retrieve (default: 7 days)
            
        Returns:
            pd.DataFrame: Combined training data
        """
        try:
            cutoff_time = datetime.utcnow() - timedelta(hours=hours)
            
            # Get recent training data
            cursor = self.training_data_collection.find(
                {"saved_at": {"$gte": cutoff_time}}
            ).sort("saved_at", DESCENDING)
            
            all_data = []
            for doc in cursor:
                if 'data' in doc:
                    all_data.extend(doc['data'])
            
            if all_data:
                df = pd.DataFrame(all_data)
                # Convert timestamp to datetime
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                # Remove duplicates and sort
                df = df.drop_duplicates(subset=['timestamp']).sort_values('timestamp')
                
                print(f"📊 Retrieved {len(df)} training records from last {hours} hours")
                return df
            else:
                print(f"📊 No training data found in last {hours} hours")
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
                "metrics": metrics
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
    
    def get_database_stats(self) -> Dict:
        """
        Get database statistics.
        
        Returns:
            Dict: Database statistics
        """
        try:
            stats = {
                "predictions_count": self.predictions_collection.count_documents({}),
                "training_data_count": self.training_data_collection.count_documents({}),
                "models_count": self.models_collection.count_documents({}),
                "performance_records_count": self.performance_collection.count_documents({}),
                "database_size_mb": self.db.command("dbStats")["dataSize"] / (1024 * 1024)
            }
            
            print("📊 Database Statistics:")
            for key, value in stats.items():
                if "size_mb" in key:
                    print(f"  {key}: {value:.2f} MB")
                else:
                    print(f"  {key}: {value:,}")
            
            return stats
            
        except Exception as e:
            print(f"❌ Failed to get database stats: {str(e)}")
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
MONGODB_URI=mongodb://localhost:27017/
DATABASE_NAME=bitcoin_volatility

# Optional: MongoDB Atlas connection (comment out local URI above)
# MONGODB_URI=mongodb+srv://username:password@cluster.mongodb.net/

# Collection Settings
PREDICTION_RETENTION_DAYS=90
TRAINING_DATA_RETENTION_DAYS=180
PERFORMANCE_DATA_RETENTION_DAYS=365

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
        stats = db_manager.get_database_stats()
        
        # Test basic operations
        print("\n✅ Database manager is working correctly!")
        print("💡 You can now use it with the real-time predictor")
        
        # Close connection
        db_manager.close_connection()
        
    except Exception as e:
        print(f"\n❌ Database test failed: {str(e)}")
        print("\n💡 Troubleshooting tips:")
        print("  1. Make sure MongoDB is installed and running")
        print("  2. Check your connection string in .env file")
        print("  3. For Windows: Install MongoDB Community Server")
        print("  4. For cloud: Use MongoDB Atlas and update MONGODB_URI") 
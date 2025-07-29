#!/usr/bin/env python3
"""
Enhanced Database Manager for MongoDB operations.
Handles storage and retrieval of enhanced model predictions and training data.
"""

import pymongo
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class DatabaseManager:
    """
    Enhanced database manager for MongoDB operations.
    """
    
    def __init__(self, crypto_symbol: str = 'BTC', connection_string: str = None, database_name: str = "synth_prediction_enhanced"):
        """
        Initialize database manager.
        
        Args:
            crypto_symbol: Cryptocurrency symbol
            connection_string: MongoDB connection string
            database_name: Database name
        """
        self.crypto_symbol = crypto_symbol
        
        # Get connection string from environment or use default
        if connection_string is None:
            connection_string = os.getenv('MONGODB_CONNECTION_STRING', 'mongodb://localhost:27017/')
        
        # Connect to MongoDB
        try:
            self.client = pymongo.MongoClient(connection_string)
            self.db = self.client[database_name]
            
            # Create collections
            self.predictions_collection = self.db[f"{crypto_symbol.lower()}_enhanced_predictions"]
            self.training_data_collection = self.db[f"{crypto_symbol.lower()}_enhanced_training_data"]
            self.models_collection = self.db[f"{crypto_symbol.lower()}_enhanced_models"]
            self.performance_collection = self.db[f"{crypto_symbol.lower()}_enhanced_performance"]
            
            # Create indexes
            self._create_indexes()
            
            print(f"✅ Enhanced database manager initialized for {crypto_symbol}")
            
        except Exception as e:
            print(f"❌ Database connection failed: {str(e)}")
            raise
    
    def _create_indexes(self):
        """Create database indexes for better performance."""
        try:
            # Predictions collection indexes
            self.predictions_collection.create_index([("prediction_time", pymongo.DESCENDING)])
            self.predictions_collection.create_index([("crypto_symbol", pymongo.ASCENDING)])
            self.predictions_collection.create_index([("model_version", pymongo.ASCENDING)])
            
            # Training data collection indexes
            self.training_data_collection.create_index([("timestamp", pymongo.DESCENDING)])
            self.training_data_collection.create_index([("crypto_symbol", pymongo.ASCENDING)])
            
            # Models collection indexes
            self.models_collection.create_index([("timestamp", pymongo.DESCENDING)])
            self.models_collection.create_index([("crypto_symbol", pymongo.ASCENDING)])
            self.models_collection.create_index([("model_version", pymongo.ASCENDING)])
            
            # Performance collection indexes
            self.performance_collection.create_index([("timestamp", pymongo.DESCENDING)])
            self.performance_collection.create_index([("crypto_symbol", pymongo.ASCENDING)])
            
        except Exception as e:
            print(f"⚠️ Error creating indexes: {str(e)}")
    
    def save_prediction(self, prediction_data: Dict[str, Any]) -> str:
        """
        Save prediction results to database.
        
        Args:
            prediction_data: Prediction data dictionary
            
        Returns:
            Database ID of saved prediction
        """
        try:
            # Add metadata
            prediction_data['crypto_symbol'] = self.crypto_symbol
            prediction_data['created_at'] = datetime.now()
            
            # Insert into database
            result = self.predictions_collection.insert_one(prediction_data)
            
            return str(result.inserted_id)
            
        except Exception as e:
            print(f"❌ Error saving prediction: {str(e)}")
            return f"error_{str(e)}"
    
    def save_training_data(self, training_data: Dict[str, Any]) -> str:
        """
        Save training data to database.
        
        Args:
            training_data: Training data dictionary
            
        Returns:
            Database ID of saved training data
        """
        try:
            # Add metadata
            training_data['crypto_symbol'] = self.crypto_symbol
            training_data['created_at'] = datetime.now()
            
            # Insert into database
            result = self.training_data_collection.insert_one(training_data)
            
            return str(result.inserted_id)
            
        except Exception as e:
            print(f"❌ Error saving training data: {str(e)}")
            return f"error_{str(e)}"
    
    def save_model_metadata(self, model_data: Dict[str, Any]) -> str:
        """
        Save model metadata to database.
        
        Args:
            model_data: Model metadata dictionary
            
        Returns:
            Database ID of saved model metadata
        """
        try:
            # Add metadata
            model_data['crypto_symbol'] = self.crypto_symbol
            model_data['created_at'] = datetime.now()
            
            # Insert into database
            result = self.models_collection.insert_one(model_data)
            
            return str(result.inserted_id)
            
        except Exception as e:
            print(f"❌ Error saving model metadata: {str(e)}")
            return f"error_{str(e)}"
    
    def save_performance_metrics(self, performance_data: Dict[str, Any]) -> str:
        """
        Save performance metrics to database.
        
        Args:
            performance_data: Performance metrics dictionary
            
        Returns:
            Database ID of saved performance metrics
        """
        try:
            # Add metadata
            performance_data['crypto_symbol'] = self.crypto_symbol
            performance_data['created_at'] = datetime.now()
            
            # Insert into database
            result = self.performance_collection.insert_one(performance_data)
            
            return str(result.inserted_id)
            
        except Exception as e:
            print(f"❌ Error saving performance metrics: {str(e)}")
            return f"error_{str(e)}"
    
    def get_recent_predictions(self, hours: int = 24, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Get recent predictions from database.
        
        Args:
            hours: Number of hours to look back
            limit: Maximum number of predictions to return
            
        Returns:
            List of prediction dictionaries
        """
        try:
            cutoff_time = datetime.now() - timedelta(hours=hours)
            
            cursor = self.predictions_collection.find({
                'prediction_time': {'$gte': cutoff_time}
            }).sort('prediction_time', pymongo.DESCENDING).limit(limit)
            
            return list(cursor)
            
        except Exception as e:
            print(f"❌ Error getting recent predictions: {str(e)}")
            return []
    
    def get_recent_training_data(self, hours: int = 24) -> List[Dict[str, Any]]:
        """
        Get recent training data from database.
        
        Args:
            hours: Number of hours to look back
            
        Returns:
            List of training data dictionaries
        """
        try:
            cutoff_time = datetime.now() - timedelta(hours=hours)
            
            cursor = self.training_data_collection.find({
                'timestamp': {'$gte': cutoff_time}
            }).sort('timestamp', pymongo.DESCENDING)
            
            return list(cursor)
            
        except Exception as e:
            print(f"❌ Error getting recent training data: {str(e)}")
            return []
    
    def get_recent_training_data_count(self, hours: int = 24) -> int:
        """
        Get count of recent training data points.
        
        Args:
            hours: Number of hours to look back
            
        Returns:
            Number of training data points
        """
        try:
            cutoff_time = datetime.now() - timedelta(hours=hours)
            
            count = self.training_data_collection.count_documents({
                'timestamp': {'$gte': cutoff_time}
            })
            
            return count
            
        except Exception as e:
            print(f"❌ Error getting training data count: {str(e)}")
            return 0
    
    def get_latest_model_metadata(self) -> Optional[Dict[str, Any]]:
        """
        Get latest model metadata from database.
        
        Returns:
            Latest model metadata dictionary or None
        """
        try:
            cursor = self.models_collection.find().sort('timestamp', pymongo.DESCENDING).limit(1)
            result = list(cursor)
            
            return result[0] if result else None
            
        except Exception as e:
            print(f"❌ Error getting latest model metadata: {str(e)}")
            return None
    
    def get_performance_history(self, days: int = 30) -> List[Dict[str, Any]]:
        """
        Get performance history from database.
        
        Args:
            days: Number of days to look back
            
        Returns:
            List of performance metrics dictionaries
        """
        try:
            cutoff_time = datetime.now() - timedelta(days=days)
            
            cursor = self.performance_collection.find({
                'timestamp': {'$gte': cutoff_time}
            }).sort('timestamp', pymongo.DESCENDING)
            
            return list(cursor)
            
        except Exception as e:
            print(f"❌ Error getting performance history: {str(e)}")
            return []
    
    def cleanup_old_data(self, days: int = 90):
        """
        Clean up old data from database.
        
        Args:
            days: Number of days to keep
        """
        try:
            cutoff_time = datetime.now() - timedelta(days=days)
            
            # Clean up old predictions
            predictions_deleted = self.predictions_collection.delete_many({
                'prediction_time': {'$lt': cutoff_time}
            })
            
            # Clean up old training data
            training_deleted = self.training_data_collection.delete_many({
                'timestamp': {'$lt': cutoff_time}
            })
            
            # Clean up old performance data
            performance_deleted = self.performance_collection.delete_many({
                'timestamp': {'$lt': cutoff_time}
            })
            
            print(f"🧹 Cleaned up old data:")
            print(f"   Predictions: {predictions_deleted.deleted_count}")
            print(f"   Training data: {training_deleted.deleted_count}")
            print(f"   Performance data: {performance_deleted.deleted_count}")
            
        except Exception as e:
            print(f"❌ Error cleaning up old data: {str(e)}")
    
    def get_database_stats(self) -> Dict[str, Any]:
        """
        Get database statistics.
        
        Returns:
            Dictionary with database statistics
        """
        try:
            stats = {
                'crypto_symbol': self.crypto_symbol,
                'predictions_count': self.predictions_collection.count_documents({}),
                'training_data_count': self.training_data_collection.count_documents({}),
                'models_count': self.models_collection.count_documents({}),
                'performance_count': self.performance_collection.count_documents({}),
                'latest_prediction': None,
                'latest_training_data': None,
                'latest_model': None
            }
            
            # Get latest records
            latest_prediction = self.predictions_collection.find().sort('prediction_time', pymongo.DESCENDING).limit(1)
            latest_prediction_list = list(latest_prediction)
            if latest_prediction_list:
                stats['latest_prediction'] = latest_prediction_list[0]['prediction_time']
            
            latest_training = self.training_data_collection.find().sort('timestamp', pymongo.DESCENDING).limit(1)
            latest_training_list = list(latest_training)
            if latest_training_list:
                stats['latest_training_data'] = latest_training_list[0]['timestamp']
            
            latest_model = self.models_collection.find().sort('timestamp', pymongo.DESCENDING).limit(1)
            latest_model_list = list(latest_model)
            if latest_model_list:
                stats['latest_model'] = latest_model_list[0]['timestamp']
            
            return stats
            
        except Exception as e:
            print(f"❌ Error getting database stats: {str(e)}")
            return {'error': str(e)}
    
    def close(self):
        """Close database connection."""
        try:
            self.client.close()
            print(f"✅ Database connection closed for {self.crypto_symbol}")
        except Exception as e:
            print(f"❌ Error closing database connection: {str(e)}")
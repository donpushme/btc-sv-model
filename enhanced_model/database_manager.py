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
            connection_string = os.getenv('MONGODB_URI', 'mongodb://localhost:27017/')
        
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
        Save prediction results to database in the specified format.
        
        Args:
            prediction_data: Prediction data dictionary
            
        Returns:
            Database ID of saved prediction
        """
        try:
            import time
            from datetime import datetime
            
            # Generate batch ID
            batch_id = f"continuous_{int(time.time())}"
            
            # Format the prediction data according to the specified structure
            formatted_data = {
                "prediction_timestamp": datetime.utcnow(),
                "data_timestamp": prediction_data.get('data_timestamp', datetime.utcnow()),
                "model_version": f"{self.crypto_symbol}_model",
                "batch_id": batch_id,
                "prediction_type": "continuous_batch",
                "current_price": float(prediction_data.get('current_price', 0)),
                "predictions_count": int(prediction_data.get('prediction_count', 0)),
                "interval_minutes": 5,
                "prediction_horizon_hours": 24,
                "source": "Pyth Network",
                "summary_stats": prediction_data.get('summary_stats', {}),
                "predictions": [],
                "crypto_symbol": self.crypto_symbol
            }
            
            # Format individual predictions
            if 'predictions' in prediction_data:
                for i, pred in enumerate(prediction_data['predictions'], 1):
                    # Parse timestamp
                    pred_timestamp = pred.get('timestamp')
                    if isinstance(pred_timestamp, str):
                        from datetime import datetime
                        try:
                            pred_dt = datetime.fromisoformat(pred_timestamp.replace('Z', '+00:00'))
                        except:
                            pred_dt = datetime.utcnow()
                    else:
                        pred_dt = datetime.utcnow()
                    
                    # Calculate minutes ahead
                    data_timestamp = formatted_data['data_timestamp']
                    if isinstance(data_timestamp, str):
                        try:
                            data_dt = datetime.fromisoformat(data_timestamp.replace('Z', '+00:00'))
                        except:
                            data_dt = datetime.utcnow()
                    else:
                        data_dt = data_timestamp
                    
                    minutes_ahead = int((pred_dt - data_dt).total_seconds() / 60)
                    
                    # Calculate time-based features
                    hour_utc = pred_dt.hour
                    is_us_trading_hours = 9 <= hour_utc <= 16  # 9 AM to 4 PM UTC
                    is_weekend = pred_dt.weekday() >= 5  # Saturday = 5, Sunday = 6
                    
                    # Calculate multipliers (simplified - you may want to adjust these)
                    volatility_multiplier = 1.0 + (pred.get('predicted_volatility', 0) * 0.1)
                    skewness_multiplier = 1.0 + (pred.get('predicted_skewness', 0) * 0.05)
                    kurtosis_multiplier = 1.0 + (pred.get('predicted_kurtosis', 0) * 0.02)
                    
                    # Calculate confidence intervals (simplified)
                    current_price = formatted_data['current_price']
                    volatility = pred.get('predicted_volatility', 0)
                    confidence_range = current_price * volatility * 2  # 2 standard deviations
                    
                    # Determine market regime and risk assessment
                    volatility_val = pred.get('predicted_volatility', 0)
                    skewness_val = pred.get('predicted_skewness', 0)
                    kurtosis_val = pred.get('predicted_kurtosis', 0)
                    
                    if volatility_val > 0.08:
                        if skewness_val > 1.5:
                            market_regime = "high_volatility_skewed"
                        else:
                            market_regime = "high_volatility_normal"
                    else:
                        if skewness_val > 1.5:
                            market_regime = "low_volatility_skewed"
                        else:
                            market_regime = "low_volatility_normal"
                    
                    if volatility_val > 0.1 or kurtosis_val > 8:
                        risk_assessment = "very_high"
                    elif volatility_val > 0.08 or kurtosis_val > 6:
                        risk_assessment = "high"
                    elif volatility_val > 0.06 or kurtosis_val > 4:
                        risk_assessment = "medium"
                    else:
                        risk_assessment = "low"
                    
                    # Include both existing uncertainty fields and new format fields
                    formatted_pred = {
                        "sequence_number": i,
                        "timestamp": pred_timestamp,
                        "minutes_ahead": minutes_ahead,
                        "predicted_volatility": float(pred.get('predicted_volatility', 0)),
                        "predicted_skewness": float(pred.get('predicted_skewness', 0)),
                        "predicted_kurtosis": float(pred.get('predicted_kurtosis', 0)),
                        "volatility_annualized": float(pred.get('volatility_annualized', 0)),
                        "volatility_multiplier": float(volatility_multiplier),
                        "skewness_multiplier": float(skewness_multiplier),
                        "kurtosis_multiplier": float(kurtosis_multiplier),
                        "hour_utc": hour_utc,
                        "is_us_trading_hours": is_us_trading_hours,
                        "is_weekend": is_weekend,
                        "current_price": float(current_price),
                        "confidence_interval_lower": float(current_price - confidence_range),
                        "confidence_interval_upper": float(current_price + confidence_range),
                        "market_regime": market_regime,
                        "risk_assessment": risk_assessment,
                        "prediction_period": "5_minutes",
                        "data_timestamp": formatted_data['data_timestamp'].isoformat() if hasattr(formatted_data['data_timestamp'], 'isoformat') else str(formatted_data['data_timestamp']),
                        "model_version": formatted_data['model_version'],
                        "prediction_type": "continuous_5min_varying",
                        # Keep existing uncertainty fields for backward compatibility
                        "uncertainty_volatility": float(pred.get('uncertainty_volatility', 0)),
                        "uncertainty_skewness": float(pred.get('uncertainty_skewness', 0)),
                        "uncertainty_kurtosis": float(pred.get('uncertainty_kurtosis', 0)),
                        "confidence": float(pred.get('confidence', 0)),
                        "prediction_horizon_minutes": int(pred.get('prediction_horizon_minutes', 0))
                    }
                    
                    formatted_data['predictions'].append(formatted_pred)
            
            # Insert into database
            result = self.predictions_collection.insert_one(formatted_data)
            
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
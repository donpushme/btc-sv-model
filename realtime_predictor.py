#!/usr/bin/env python3

"""
Enhanced Real-time Bitcoin Volatility Predictor with Database Integration and Online Learning
Combines real-time predictions with continuous model improvement and data persistence.
"""

import os
import torch
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
from datetime import datetime, timedelta
import schedule
import time
import threading
import json
from dotenv import load_dotenv

from config import Config
from predictor import RealTimeVolatilityPredictor
from trainer import BitcoinVolatilityTrainer
from database_manager import DatabaseManager
from utils import validate_bitcoin_data, format_prediction_output

# Load environment variables
load_dotenv()

class EnhancedRealTimePredictor:
    """
    Enhanced real-time predictor with database integration and continuous learning.
    """
    
    def __init__(self, model_path: str = None, enable_database: bool = True, 
                 enable_online_learning: bool = True):
        """
        Initialize the enhanced predictor.
        
        Args:
            model_path: Path to trained model
            enable_database: Enable MongoDB integration
            enable_online_learning: Enable continuous learning
        """
        self.config = Config()
        self.enable_database = enable_database and self.config.ENABLE_DATABASE
        self.enable_online_learning = enable_online_learning and self.config.ENABLE_ONLINE_LEARNING
        
        # Initialize base predictor
        self.predictor = RealTimeVolatilityPredictor(model_path)
        
        # Initialize database manager
        self.db_manager = None
        if self.enable_database:
            try:
                self.db_manager = DatabaseManager(
                    database_name=self.config.DATABASE_NAME
                )
            except Exception as e:
                print(f"⚠️ Database initialization failed: {str(e)}")
                print("🔄 Continuing without database features...")
                self.enable_database = False
        
        # Initialize trainer for online learning
        self.trainer = None
        if self.enable_online_learning:
            self.trainer = BitcoinVolatilityTrainer(self.config)
        
        # Online learning state
        self.last_retrain_time = datetime.utcnow()
        self.new_data_count = 0
        self.current_model_version = self._get_model_version()
        
        # Background tasks
        self.scheduler_thread = None
        self.running = False
        
        print(f"🚀 Enhanced Real-time Predictor initialized")
        print(f"   📊 Database: {'✅ Enabled' if self.enable_database else '❌ Disabled'}")
        print(f"   🧠 Online Learning: {'✅ Enabled' if self.enable_online_learning else '❌ Disabled'}")
        print(f"   🔧 Model Version: {self.current_model_version}")
    
    def _get_model_version(self) -> str:
        """Get current model version."""
        try:
            if hasattr(self.predictor, 'model') and self.predictor.model is not None:
                # Try to get version from model metadata
                return datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            else:
                return "unknown"
        except:
            return "unknown"
    
    def predict_and_save(self, price_data: pd.DataFrame, 
                        current_price: Optional[float] = None,
                        save_to_db: bool = True) -> Dict[str, float]:
        """
        Make prediction and optionally save to database.
        
        Args:
            price_data: Historical price data
            current_price: Current Bitcoin price
            save_to_db: Whether to save prediction to database
            
        Returns:
            Dict: Prediction results
        """
        try:
            # Make prediction
            prediction = self.predictor.predict_next_period(price_data, current_price)
            
            # Add model version
            prediction['model_version'] = self.current_model_version
            prediction['prediction_id'] = f"{self.current_model_version}_{int(time.time())}"
            
            # Save to database
            if save_to_db and self.enable_database and self.db_manager:
                try:
                    prediction_id = self.db_manager.save_prediction(
                        prediction, self.current_model_version
                    )
                    prediction['database_id'] = prediction_id
                except Exception as e:
                    print(f"⚠️ Failed to save prediction to database: {str(e)}")
            
            # Save training data for online learning
            if self.enable_online_learning and self.enable_database and self.db_manager:
                try:
                    # Save recent data for future training
                    recent_data = price_data.tail(100)  # Last 100 data points
                    self.db_manager.save_training_data(recent_data, "realtime")
                    self.new_data_count += len(recent_data)
                except Exception as e:
                    print(f"⚠️ Failed to save training data: {str(e)}")
            
            # Check if retraining is needed
            if self.enable_online_learning:
                self._check_retrain_condition()
            
            return prediction
            
        except Exception as e:
            print(f"❌ Prediction failed: {str(e)}")
            raise
    
    def batch_predict_and_save(self, price_data_list: List[pd.DataFrame]) -> List[Dict]:
        """
        Make batch predictions and save to database.
        
        Args:
            price_data_list: List of price data DataFrames
            
        Returns:
            List[Dict]: List of prediction results
        """
        results = []
        for i, price_data in enumerate(price_data_list):
            try:
                prediction = self.predict_and_save(price_data)
                results.append(prediction)
                print(f"📊 Completed prediction {i+1}/{len(price_data_list)}")
            except Exception as e:
                error_result = {
                    'error': str(e),
                    'batch_index': i,
                    'timestamp': datetime.utcnow().isoformat()
                }
                results.append(error_result)
                print(f"❌ Batch prediction {i+1} failed: {str(e)}")
        
        return results
    
    def _check_retrain_condition(self):
        """Check if model retraining is needed."""
        if not self.enable_online_learning:
            return
        
        try:
            current_time = datetime.utcnow()
            time_since_retrain = current_time - self.last_retrain_time
            
            # Check time-based condition
            time_condition = time_since_retrain.total_seconds() / 3600 >= self.config.RETRAIN_INTERVAL_HOURS
            
            # Check data-based condition
            data_condition = self.new_data_count >= self.config.MIN_NEW_DATA_POINTS
            
            if time_condition or data_condition:
                print(f"🔄 Retraining condition met:")
                print(f"   ⏰ Time since last retrain: {time_since_retrain}")
                print(f"   📊 New data points: {self.new_data_count}")
                
                # Schedule retraining in background
                self._schedule_retrain()
        
        except Exception as e:
            print(f"⚠️ Error checking retrain condition: {str(e)}")
    
    def _schedule_retrain(self):
        """Schedule model retraining in background."""
        def retrain_task():
            try:
                print("🔄 Starting background model retraining...")
                success = self._retrain_model()
                if success:
                    print("✅ Model retraining completed successfully")
                else:
                    print("❌ Model retraining failed")
            except Exception as e:
                print(f"❌ Background retraining error: {str(e)}")
        
        # Run retraining in a separate thread
        retrain_thread = threading.Thread(target=retrain_task, daemon=True)
        retrain_thread.start()
    
    def _retrain_model(self) -> bool:
        """
        Retrain the model with new data.
        
        Returns:
            bool: True if retraining successful
        """
        try:
            if not self.enable_database or not self.db_manager:
                print("❌ Database required for online learning")
                return False
            
            # Get recent training data
            print("📊 Collecting recent training data...")
            training_data = self.db_manager.get_training_data_for_update(
                hours=self.config.MAX_TRAINING_DATA_HOURS
            )
            
            if len(training_data) < self.config.MIN_NEW_DATA_POINTS:
                print(f"❌ Insufficient training data: {len(training_data)} points")
                return False
            
            # Validate data
            validation_result = validate_bitcoin_data(training_data)
            if validation_result['errors']:
                print(f"❌ Data validation failed: {validation_result['errors']}")
                return False
            
            # Create temporary CSV for training
            temp_csv_path = 'data/temp_retrain_data.csv'
            os.makedirs('data', exist_ok=True)
            training_data.to_csv(temp_csv_path, index=False)
            
            # Backup current model
            backup_path = f"models/backup_{self.current_model_version}.pth"
            if os.path.exists(self.predictor.model):
                import shutil
                shutil.copy2(self.predictor.model, backup_path)
            
            # Initialize trainer and retrain
            print("🧠 Starting model retraining...")
            retrain_config = Config()
            retrain_config.NUM_EPOCHS = 50  # Fewer epochs for online learning
            retrain_config.EARLY_STOPPING_PATIENCE = 10
            
            trainer = BitcoinVolatilityTrainer(retrain_config)
            training_results = trainer.train(temp_csv_path)
            
            # Update model version
            new_model_version = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            
            # Save model metadata
            if self.enable_database:
                self.db_manager.save_model_metadata(
                    model_path=f"models/best_model.pth",
                    training_metrics=training_results,
                    config=vars(retrain_config),
                    model_version=new_model_version
                )
            
            # Reload predictor with new model
            print("🔄 Reloading predictor with new model...")
            self.predictor.load_model("models/best_model.pth")
            self.current_model_version = new_model_version
            
            # Update state
            self.last_retrain_time = datetime.utcnow()
            self.new_data_count = 0
            
            # Cleanup
            try:
                os.remove(temp_csv_path)
            except:
                pass
            
            print(f"✅ Model retrained successfully! New version: {new_model_version}")
            return True
            
        except Exception as e:
            print(f"❌ Model retraining failed: {str(e)}")
            
            # Try to restore backup if available
            try:
                backup_path = f"models/backup_{self.current_model_version}.pth"
                if os.path.exists(backup_path):
                    import shutil
                    shutil.copy2(backup_path, "models/best_model.pth")
                    print("🔄 Restored backup model")
            except:
                pass
            
            return False
    
    def get_prediction_history(self, hours: int = 24) -> pd.DataFrame:
        """
        Get prediction history from database.
        
        Args:
            hours: Number of hours to look back
            
        Returns:
            pd.DataFrame: Prediction history
        """
        if not self.enable_database or not self.db_manager:
            print("❌ Database not available")
            return pd.DataFrame()
        
        return self.db_manager.get_recent_predictions(hours=hours)
    
    def get_model_performance(self, days: int = 7) -> pd.DataFrame:
        """
        Get model performance history.
        
        Args:
            days: Number of days to look back
            
        Returns:
            pd.DataFrame: Performance history
        """
        if not self.enable_database or not self.db_manager:
            print("❌ Database not available")
            return pd.DataFrame()
        
        return self.db_manager.get_model_performance_history(days=days)
    
    def start_scheduled_tasks(self):
        """Start background scheduled tasks."""
        if self.running:
            print("⚠️ Scheduled tasks already running")
            return
        
        self.running = True
        
        # Schedule periodic tasks
        if self.enable_database:
            # Database cleanup
            schedule.every().day.at("02:00").do(self._cleanup_database)
            
            # Performance monitoring
            schedule.every().hour.do(self._monitor_performance)
        
        if self.enable_online_learning:
            # Check retrain conditions
            schedule.every(30).minutes.do(self._check_retrain_condition)
        
        # Start scheduler thread
        def run_scheduler():
            while self.running:
                schedule.run_pending()
                time.sleep(60)  # Check every minute
        
        self.scheduler_thread = threading.Thread(target=run_scheduler, daemon=True)
        self.scheduler_thread.start()
        
        print("⏰ Background scheduled tasks started")
    
    def stop_scheduled_tasks(self):
        """Stop background scheduled tasks."""
        self.running = False
        if self.scheduler_thread:
            self.scheduler_thread.join(timeout=5)
        schedule.clear()
        print("⏰ Background scheduled tasks stopped")
    
    def _cleanup_database(self):
        """Periodic database cleanup."""
        if self.enable_database and self.db_manager:
            try:
                print("🧹 Running database cleanup...")
                self.db_manager.cleanup_old_data()
            except Exception as e:
                print(f"❌ Database cleanup failed: {str(e)}")
    
    def _monitor_performance(self):
        """Monitor model performance."""
        try:
            if not self.enable_database or not self.db_manager:
                return
            
            # Get recent predictions
            recent_predictions = self.db_manager.get_recent_predictions(hours=24)
            
            if len(recent_predictions) > 0:
                # Calculate basic performance metrics
                metrics = {
                    "predictions_count": len(recent_predictions),
                    "average_volatility": float(recent_predictions['predicted_volatility'].mean()),
                    "average_risk_score": len(recent_predictions[recent_predictions['risk_assessment'].isin(['high', 'very_high'])]) / len(recent_predictions),
                    "timestamp": datetime.utcnow().isoformat()
                }
                
                # Save performance metrics
                self.db_manager.save_performance_metrics(
                    self.current_model_version, metrics
                )
        
        except Exception as e:
            print(f"❌ Performance monitoring failed: {str(e)}")
    
    def get_system_status(self) -> Dict:
        """Get system status and statistics."""
        status = {
            "timestamp": datetime.utcnow().isoformat(),
            "model_version": self.current_model_version,
            "database_enabled": self.enable_database,
            "online_learning_enabled": self.enable_online_learning,
            "running": self.running,
            "last_retrain": self.last_retrain_time.isoformat(),
            "new_data_count": self.new_data_count
        }
        
        if self.enable_database and self.db_manager:
            try:
                db_stats = self.db_manager.get_database_stats()
                status["database_stats"] = db_stats
            except:
                status["database_stats"] = {"error": "Failed to get stats"}
        
        return status
    
    def force_retrain(self) -> bool:
        """Force model retraining."""
        if not self.enable_online_learning:
            print("❌ Online learning not enabled")
            return False
        
        print("🔄 Forcing model retraining...")
        return self._retrain_model()
    
    def close(self):
        """Clean shutdown."""
        self.stop_scheduled_tasks()
        
        if self.enable_database and self.db_manager:
            self.db_manager.close_connection()
        
        print("🔐 Enhanced predictor shutdown complete")


def demo_enhanced_predictor():
    """Demonstrate the enhanced predictor functionality."""
    print("🚀 Enhanced Real-time Predictor Demo")
    print("=====================================")
    
    try:
        # Initialize enhanced predictor
        predictor = EnhancedRealTimePredictor()
        
        # Get system status
        status = predictor.get_system_status()
        print(f"\n📊 System Status:")
        for key, value in status.items():
            print(f"  {key}: {value}")
        
        # Create sample data for demo
        print(f"\n📈 Creating sample prediction...")
        dates = pd.date_range(start='2024-01-01', periods=500, freq='5min')
        
        # Simulate Bitcoin price data
        np.random.seed(42)
        prices = [45000]
        for _ in range(len(dates) - 1):
            change = np.random.normal(0, 0.02)
            prices.append(prices[-1] * (1 + change))
        
        sample_data = pd.DataFrame({
            'timestamp': dates,
            'open': prices,
            'close': prices,
            'high': [p * (1 + abs(np.random.normal(0, 0.01))) for p in prices],
            'low': [p * (1 - abs(np.random.normal(0, 0.01))) for p in prices]
        })
        
        # Make prediction
        prediction = predictor.predict_and_save(sample_data)
        
        print(f"\n🔮 Prediction Results:")
        print(format_prediction_output(prediction))
        
        if predictor.enable_database:
            # Get prediction history
            history = predictor.get_prediction_history(hours=1)
            print(f"\n📚 Recent predictions in database: {len(history)}")
        
        # Start background tasks
        predictor.start_scheduled_tasks()
        print(f"\n⏰ Background tasks started")
        
        # Wait a bit for demo
        print(f"\n⏳ Running for 10 seconds to demonstrate background tasks...")
        time.sleep(10)
        
        # Shutdown
        predictor.close()
        print(f"\n✅ Demo completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Demo failed: {str(e)}")
        print(f"\n💡 Make sure MongoDB is running for full functionality")


if __name__ == "__main__":
    demo_enhanced_predictor() 
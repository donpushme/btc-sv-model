import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')
import pickle

from config import Config
from model import VolatilityLSTM, VolatilityLoss, init_model
from feature_engineering import FeatureEngineer, BitcoinDataset, create_train_val_split
from data_processor import BitcoinDataProcessor

class BitcoinVolatilityTrainer:
    """
    Training pipeline for Bitcoin volatility prediction model.
    """
    
    def __init__(self, config: Config):
        self.config = config
        self.device = config.DEVICE
        self.model = None
        self.feature_engineer = FeatureEngineer()
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        
        # Create directories
        os.makedirs(config.MODEL_SAVE_PATH, exist_ok=True)
        os.makedirs(config.RESULTS_PATH, exist_ok=True)
        
        print(f"Using device: {self.device}")
    
    def prepare_data(self, csv_path: str) -> Tuple[DataLoader, DataLoader, List[str], List[str]]:
        """
        Prepare training and validation data loaders.
        """
        print("Loading and preprocessing data...")
        
        # Load and preprocess data
        processor = BitcoinDataProcessor(csv_path)
        df = processor.preprocess_data(
            return_windows=self.config.RETURN_WINDOWS,
            prediction_horizon=self.config.PREDICTION_HORIZON
        )
        
        # Add additional features
        df = self.feature_engineer.engineer_features(df)
        
        # Remove any remaining NaN values
        df = df.dropna().reset_index(drop=True)
        print(f"Final dataset shape after feature engineering: {df.shape}")
        
        # Get feature and target columns
        feature_cols = [col for col in df.columns if col not in 
                       ['timestamp', 'target_volatility', 'target_skewness', 'target_kurtosis']]
        target_cols = ['target_volatility', 'target_skewness', 'target_kurtosis']
        
        print(f"Number of features: {len(feature_cols)}")
        print(f"Features: {feature_cols[:10]}...")  # Show first 10 features
        
        # Update config with actual input size
        self.config.INPUT_SIZE = len(feature_cols)
        
        # Fit scalers on training data
        train_size = int(len(df) * (1 - self.config.VALIDATION_SPLIT))
        train_df = df.iloc[:train_size]
        self.feature_engineer.fit_scalers(train_df, feature_cols, target_cols)
        
        # Scale the data
        df_scaled = self.feature_engineer.transform_data(df, feature_cols, target_cols)
        
        # Prepare sequences
        X, y = self.feature_engineer.prepare_sequences(
            df_scaled, feature_cols, target_cols, self.config.SEQUENCE_LENGTH
        )
        
        print(f"Sequence shape: X={X.shape}, y={y.shape}")
        
        # Train/validation split
        X_train, X_val, y_train, y_val = create_train_val_split(
            X, y, self.config.VALIDATION_SPLIT
        )
        
        print(f"Train: X={X_train.shape}, y={y_train.shape}")
        print(f"Validation: X={X_val.shape}, y={y_val.shape}")
        
        # Create datasets and data loaders
        train_dataset = BitcoinDataset(X_train, y_train)
        val_dataset = BitcoinDataset(X_val, y_val)
        
        train_loader = DataLoader(
            train_dataset, 
            batch_size=self.config.BATCH_SIZE, 
            shuffle=True,
            num_workers=0
        )
        
        val_loader = DataLoader(
            val_dataset, 
            batch_size=self.config.BATCH_SIZE, 
            shuffle=False,
            num_workers=0
        )
        
        return train_loader, val_loader, feature_cols, target_cols
    
    def train_epoch(self, model: nn.Module, train_loader: DataLoader, 
                   optimizer: torch.optim.Optimizer, criterion: nn.Module) -> Dict[str, float]:
        """
        Train model for one epoch.
        """
        model.train()
        total_loss = 0.0
        total_vol_loss = 0.0
        total_skew_loss = 0.0
        total_kurt_loss = 0.0
        num_batches = 0
        
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(self.device)
            batch_y = batch_y.to(self.device)
            
            optimizer.zero_grad()
            
            # Forward pass
            predictions = model(batch_x)
            loss, vol_loss, skew_loss, kurt_loss = criterion(predictions, batch_y)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            total_loss += loss.item()
            total_vol_loss += vol_loss.item()
            total_skew_loss += skew_loss.item()
            total_kurt_loss += kurt_loss.item()
            num_batches += 1
        
        return {
            'total_loss': total_loss / num_batches,
            'vol_loss': total_vol_loss / num_batches,
            'skew_loss': total_skew_loss / num_batches,
            'kurt_loss': total_kurt_loss / num_batches
        }
    
    def validate_epoch(self, model: nn.Module, val_loader: DataLoader, 
                      criterion: nn.Module) -> Dict[str, float]:
        """
        Validate model for one epoch.
        """
        model.eval()
        total_loss = 0.0
        total_vol_loss = 0.0
        total_skew_loss = 0.0
        total_kurt_loss = 0.0
        num_batches = 0
        
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                
                predictions = model(batch_x)
                loss, vol_loss, skew_loss, kurt_loss = criterion(predictions, batch_y)
                
                total_loss += loss.item()
                total_vol_loss += vol_loss.item()
                total_skew_loss += skew_loss.item()
                total_kurt_loss += kurt_loss.item()
                num_batches += 1
                
                # Store for detailed metrics
                all_predictions.append(predictions.cpu().numpy())
                all_targets.append(batch_y.cpu().numpy())
        
        # Calculate detailed metrics
        all_predictions = np.concatenate(all_predictions, axis=0)
        all_targets = np.concatenate(all_targets, axis=0)
        
        # Inverse transform to original scale
        predictions_original = self.feature_engineer.inverse_transform_targets(all_predictions)
        targets_original = self.feature_engineer.inverse_transform_targets(all_targets)
        
        # Calculate R² scores for each target
        r2_vol = r2_score(targets_original[:, 0], predictions_original[:, 0])
        r2_skew = r2_score(targets_original[:, 1], predictions_original[:, 1])
        r2_kurt = r2_score(targets_original[:, 2], predictions_original[:, 2])
        
        return {
            'total_loss': total_loss / num_batches,
            'vol_loss': total_vol_loss / num_batches,
            'skew_loss': total_skew_loss / num_batches,
            'kurt_loss': total_kurt_loss / num_batches,
            'r2_vol': r2_vol,
            'r2_skew': r2_skew,
            'r2_kurt': r2_kurt
        }
    
    def train(self, csv_path: str) -> Dict:
        """
        Complete training pipeline.
        """
        print("Starting training...")
        
        # Prepare data
        train_loader, val_loader, feature_cols, target_cols = self.prepare_data(csv_path)
        
        # Initialize model
        self.model = init_model(self.config).to(self.device)
        
        # Initialize optimizer and criterion
        optimizer = torch.optim.AdamW(
            self.model.parameters(), 
            lr=self.config.LEARNING_RATE,
            weight_decay=1e-5
        )
        
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5
        )
        
        criterion = VolatilityLoss()
        
        # Training loop
        training_history = {
            'train_losses': [],
            'val_losses': [],
            'train_metrics': [],
            'val_metrics': []
        }
        
        for epoch in range(self.config.NUM_EPOCHS):
            # Train
            train_metrics = self.train_epoch(self.model, train_loader, optimizer, criterion)
            
            # Validate
            val_metrics = self.validate_epoch(self.model, val_loader, criterion)
            
            # Update learning rate
            scheduler.step(val_metrics['total_loss'])
            
            # Store metrics
            training_history['train_losses'].append(train_metrics['total_loss'])
            training_history['val_losses'].append(val_metrics['total_loss'])
            training_history['train_metrics'].append(train_metrics)
            training_history['val_metrics'].append(val_metrics)
            
            # Print progress
            if epoch % 10 == 0 or epoch == self.config.NUM_EPOCHS - 1:
                print(f"Epoch {epoch}/{self.config.NUM_EPOCHS}")
                print(f"  Train Loss: {train_metrics['total_loss']:.6f}")
                print(f"  Val Loss: {val_metrics['total_loss']:.6f}")
                print(f"  Val R² - Vol: {val_metrics['r2_vol']:.4f}, "
                      f"Skew: {val_metrics['r2_skew']:.4f}, "
                      f"Kurt: {val_metrics['r2_kurt']:.4f}")
            
            # Early stopping and checkpointing
            if val_metrics['total_loss'] < self.best_val_loss:
                self.best_val_loss = val_metrics['total_loss']
                self.patience_counter = 0
                
                # Save best model
                self.save_model(epoch, train_metrics, val_metrics, feature_cols, target_cols)
            else:
                self.patience_counter += 1
                
            if self.patience_counter >= self.config.EARLY_STOPPING_PATIENCE:
                print(f"Early stopping triggered after epoch {epoch}")
                break
        
        # Save training history
        self.save_training_history(training_history)
        
        # Generate training plots
        self.plot_training_history(training_history)
        
        print("Training completed!")
        return training_history
    
    def save_model(self, epoch: int, train_metrics: Dict, val_metrics: Dict, 
                  feature_cols: List[str], target_cols: List[str], suffix: str = "") -> None:
        """Save the trained model and metadata."""
        
        # Create model directory if it doesn't exist
        os.makedirs(self.config.MODEL_SAVE_PATH, exist_ok=True)
        
        # Generate timestamp for model version
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_version = f"{timestamp}{suffix}"
        
        # Save model state
        model_path = os.path.join(self.config.MODEL_SAVE_PATH, f"model_{model_version}.pth")
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'config': self.config,
            'feature_cols': feature_cols,
            'target_cols': target_cols,
            'epoch': epoch,
            'train_metrics': train_metrics,
            'val_metrics': val_metrics,
            'model_version': model_version
        }, model_path)
        
        print(f"Model saved to {model_path}")
        
        # Save feature engineer state
        feature_engineer_path = os.path.join(self.config.MODEL_SAVE_PATH, f"feature_engineer_{model_version}.pkl")
        with open(feature_engineer_path, 'wb') as f:
            pickle.dump(self.feature_engineer, f)
        
        print(f"Feature engineer saved to {feature_engineer_path}")
        
        # Save model metadata
        metadata = {
            'model_version': model_version,
            'timestamp': timestamp,
            'epoch': epoch,
            'feature_cols': feature_cols,
            'target_cols': target_cols,
            'num_features': len(feature_cols),
            'train_metrics': train_metrics,
            'val_metrics': val_metrics,
            'model_path': model_path,
            'feature_engineer_path': feature_engineer_path
        }
        
        metadata_path = os.path.join(self.config.MODEL_SAVE_PATH, f"metadata_{model_version}.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"Model metadata saved to {metadata_path}")
    
    def save_training_history(self, history: Dict):
        """
        Save training history to JSON.
        """
        with open(os.path.join(self.config.RESULTS_PATH, 'training_history.json'), 'w') as f:
            json.dump(history, f, indent=2)
    
    def plot_training_history(self, history: Dict):
        """
        Plot training and validation metrics.
        """
        plt.style.use('seaborn-v0_8')
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Loss curves
        axes[0, 0].plot(history['train_losses'], label='Training Loss', color='blue')
        axes[0, 0].plot(history['val_losses'], label='Validation Loss', color='red')
        axes[0, 0].set_title('Training and Validation Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # R² scores over time
        val_r2_vol = [m['r2_vol'] for m in history['val_metrics']]
        val_r2_skew = [m['r2_skew'] for m in history['val_metrics']]
        val_r2_kurt = [m['r2_kurt'] for m in history['val_metrics']]
        
        axes[0, 1].plot(val_r2_vol, label='Volatility R²', color='green')
        axes[0, 1].plot(val_r2_skew, label='Skewness R²', color='orange')
        axes[0, 1].plot(val_r2_kurt, label='Kurtosis R²', color='purple')
        axes[0, 1].set_title('Validation R² Scores')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('R² Score')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Individual loss components
        train_vol_losses = [m['vol_loss'] for m in history['train_metrics']]
        train_skew_losses = [m['skew_loss'] for m in history['train_metrics']]
        train_kurt_losses = [m['kurt_loss'] for m in history['train_metrics']]
        
        axes[1, 0].plot(train_vol_losses, label='Volatility Loss', color='blue')
        axes[1, 0].plot(train_skew_losses, label='Skewness Loss', color='orange')
        axes[1, 0].plot(train_kurt_losses, label='Kurtosis Loss', color='green')
        axes[1, 0].set_title('Training Loss Components')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Loss')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # Validation loss components
        val_vol_losses = [m['vol_loss'] for m in history['val_metrics']]
        val_skew_losses = [m['skew_loss'] for m in history['val_metrics']]
        val_kurt_losses = [m['kurt_loss'] for m in history['val_metrics']]
        
        axes[1, 1].plot(val_vol_losses, label='Volatility Loss', color='blue')
        axes[1, 1].plot(val_skew_losses, label='Skewness Loss', color='orange')
        axes[1, 1].plot(val_kurt_losses, label='Kurtosis Loss', color='green')
        axes[1, 1].set_title('Validation Loss Components')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Loss')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.config.RESULTS_PATH, 'training_history.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Training plots saved to {self.config.RESULTS_PATH}")

    def retrain_with_current_data(self, csv_path: str, days_back: int = 30, 
                                 preserve_weights: bool = True) -> Dict:
        """
        Retrain the model with only recent data for adaptation to current market conditions.
        
        Args:
            csv_path: Path to the data CSV file
            days_back: Number of days of recent data to use for retraining
            preserve_weights: Whether to preserve existing model weights as initialization
            
        Returns:
            Dictionary with retraining results
        """
        print(f"🔄 Retraining with recent data (last {days_back} days)...")
        
        # Load the data
        processor = BitcoinDataProcessor(csv_path)
        df = processor.load_data()
        
        # Filter to recent data only
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        cutoff_date = df['timestamp'].max() - pd.Timedelta(days=days_back)
        recent_df = df[df['timestamp'] >= cutoff_date].copy()
        
        print(f"📊 Using {len(recent_df):,} data points from {cutoff_date.strftime('%Y-%m-%d')} to {df['timestamp'].max().strftime('%Y-%m-%d')}")
        
        # Check if we have enough data
        if len(recent_df) < 100:
            print(f"❌ Insufficient data for retraining: {len(recent_df)} < 100 minimum required")
            return {
                'success': False,
                'error': f'Insufficient data: {len(recent_df)} < 100 minimum required',
                'data_points': len(recent_df)
            }
        
        if len(recent_df) < 1000:
            print("⚠️  Warning: Very small dataset for retraining. Consider using more days_back.")
        
        # Preprocess the recent data
        recent_df = processor.preprocess_data(
            return_windows=self.config.RETURN_WINDOWS,
            prediction_horizon=self.config.PREDICTION_HORIZON
        )
        
        # Add additional features
        recent_df = self.feature_engineer.engineer_features(recent_df)
        
        # Remove any remaining NaN values
        recent_df = recent_df.dropna().reset_index(drop=True)
        print(f"Final recent dataset shape: {recent_df.shape}")
        
        # Check if we still have enough data after preprocessing
        if len(recent_df) < 50:
            print(f"❌ Insufficient data after preprocessing: {len(recent_df)} < 50 minimum required")
            return {
                'success': False,
                'error': f'Insufficient data after preprocessing: {len(recent_df)} < 50 minimum required',
                'data_points': len(recent_df)
            }
        
        # Get feature and target columns
        feature_cols = [col for col in recent_df.columns if col not in 
                       ['timestamp', 'target_volatility', 'target_skewness', 'target_kurtosis']]
        target_cols = ['target_volatility', 'target_skewness', 'target_kurtosis']
        
        print(f"Number of features: {len(feature_cols)}")
        
        # Update config with actual input size
        self.config.INPUT_SIZE = len(feature_cols)
        
        # Fit scalers on the recent data
        self.feature_engineer.fit_scalers(recent_df, feature_cols, target_cols)
        
        # Scale the data
        recent_df_scaled = self.feature_engineer.transform_data(recent_df, feature_cols, target_cols)
        
        # Prepare sequences
        X, y = self.feature_engineer.prepare_sequences(
            recent_df_scaled, feature_cols, target_cols, self.config.SEQUENCE_LENGTH
        )
        
        print(f"Recent data sequence shape: X={X.shape}, y={y.shape}")
        
        # Check if we have enough sequences
        if len(X) < 20:
            print(f"❌ Insufficient sequences for retraining: {len(X)} < 20 minimum required")
            return {
                'success': False,
                'error': f'Insufficient sequences: {len(X)} < 20 minimum required',
                'data_points': len(X)
            }
        
        # Create train/validation split
        train_size = int(len(X) * (1 - self.config.VALIDATION_SPLIT))
        X_train, X_val = X[:train_size], X[train_size:]
        y_train, y_val = y[:train_size], y[train_size:]
        
        # Create data loaders
        train_dataset = BitcoinDataset(X_train, y_train)
        val_dataset = BitcoinDataset(X_val, y_val)
        
        train_loader = DataLoader(
            train_dataset, 
            batch_size=self.config.BATCH_SIZE, 
            shuffle=True
        )
        val_loader = DataLoader(
            val_dataset, 
            batch_size=self.config.BATCH_SIZE, 
            shuffle=False
        )
        
        # Initialize or load existing model
        if self.model is None:
            self.model = init_model(self.config).to(self.device)
            print("🆕 Initialized new model")
        else:
            if preserve_weights:
                print("🔄 Preserving existing model weights as initialization")
            else:
                print("🔄 Reinitializing model weights")
                self.model = init_model(self.config).to(self.device)
        
        # Initialize optimizer and criterion
        optimizer = torch.optim.AdamW(
            self.model.parameters(), 
            lr=self.config.LEARNING_RATE * 0.5,  # Lower learning rate for retraining
            weight_decay=1e-5
        )
        
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=3  # Shorter patience for retraining
        )
        
        criterion = VolatilityLoss()
        
        # Retraining loop with fewer epochs
        retrain_epochs = min(self.config.NUM_EPOCHS // 2, 20)  # Use fewer epochs for retraining
        
        training_history = {
            'train_losses': [],
            'val_losses': [],
            'train_metrics': [],
            'val_metrics': []
        }
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        print(f"🔄 Starting retraining for {retrain_epochs} epochs...")
        
        for epoch in range(retrain_epochs):
            # Train
            train_metrics = self.train_epoch(self.model, train_loader, optimizer, criterion)
            
            # Validate
            val_metrics = self.validate_epoch(self.model, val_loader, criterion)
            
            # Update learning rate
            scheduler.step(val_metrics['total_loss'])
            
            # Store metrics
            training_history['train_losses'].append(train_metrics['total_loss'])
            training_history['val_losses'].append(val_metrics['total_loss'])
            training_history['train_metrics'].append(train_metrics)
            training_history['val_metrics'].append(val_metrics)
            
            # Print progress
            if epoch % 5 == 0 or epoch == retrain_epochs - 1:
                print(f"Retrain Epoch {epoch}/{retrain_epochs}")
                print(f"  Train Loss: {train_metrics['total_loss']:.6f}")
                print(f"  Val Loss: {val_metrics['total_loss']:.6f}")
                print(f"  Val R² - Vol: {val_metrics['r2_vol']:.4f}, "
                      f"Skew: {val_metrics['r2_skew']:.4f}, "
                      f"Kurt: {val_metrics['r2_kurt']:.4f}")
            
            # Early stopping and checkpointing
            if val_metrics['total_loss'] < best_val_loss:
                best_val_loss = val_metrics['total_loss']
                patience_counter = 0
                
                # Save retrained model
                self.save_model(epoch, train_metrics, val_metrics, feature_cols, target_cols, 
                              suffix=f"_retrained_{days_back}days")
            else:
                patience_counter += 1
                
            if patience_counter >= 5:  # Shorter patience for retraining
                print(f"Early stopping triggered after epoch {epoch}")
                break
        
        # Save retraining history
        retrain_history_path = os.path.join(self.config.RESULTS_PATH, 
                                           f'retraining_history_{days_back}days.json')
        with open(retrain_history_path, 'w') as f:
            json.dump(training_history, f, indent=2)
        
        print(f"✅ Retraining completed!")
        print(f"📁 Retrained model saved with suffix '_retrained_{days_back}days'")
        print(f"📊 Retraining history saved to {retrain_history_path}")
        
        return {
            'success': True,
            'retrain_epochs': retrain_epochs,
            'final_train_loss': training_history['train_losses'][-1],
            'final_val_loss': training_history['val_losses'][-1],
            'best_val_loss': best_val_loss,
            'data_points_used': len(recent_df),
            'days_back': days_back,
            'model_suffix': f"_retrained_{days_back}days"
        }


def main():
    """
    Main training function.
    """
    config = Config()
    trainer = BitcoinVolatilityTrainer(config)
    
    # Replace with your CSV file path
    csv_path = 'training_data/bitcoin_5min.csv'
    
    if not os.path.exists(csv_path):
        print(f"Please place your Bitcoin price data CSV file at: {csv_path}")
        print("Expected columns: timestamp, open, close, high, low")
        return
    
    try:
        training_history = trainer.train(csv_path)
        print("Training completed successfully!")
        print(f"Best validation loss: {min(training_history['val_losses']):.6f}")
        
    except Exception as e:
        print(f"Training failed with error: {str(e)}")
        raise


if __name__ == "__main__":
    main() 
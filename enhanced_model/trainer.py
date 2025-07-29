#!/usr/bin/env python3
"""
Realistic Model Trainer for Cryptocurrency Prediction

This module provides training capabilities for the realistic enhanced model
with time-aware features and market-aware constraints.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error
import os
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Import realistic components
from enhanced_model import create_realistic_enhanced_model, RealisticLoss
from feature_engineering import RealisticFeatureEngineer, RealisticCryptoDataset, create_train_val_split
from config import RealisticConfig

class RealisticModelTrainer:
    """
    Trainer for the realistic enhanced model with time-aware features.
    """
    
    def __init__(self, config: RealisticConfig, crypto_symbol: str = 'BTC'):
        self.config = config
        self.crypto_symbol = crypto_symbol
        self.device = config.DEVICE
        
        # Initialize components
        self.feature_engineer = RealisticFeatureEngineer()
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.criterion = None
        
        # Training history
        self.train_losses = []
        self.val_losses = []
        self.train_metrics = []
        self.val_metrics = []
        
        print(f"🚀 Initialized Realistic Model Trainer for {crypto_symbol}")
    
    def load_and_preprocess_data(self) -> pd.DataFrame:
        """
        Load and preprocess data with realistic feature engineering.
        """
        print(f"📊 Loading data for {self.crypto_symbol}...")
        
        # Load data
        data_file = os.path.join(self.config.DATA_PATH, self.config.SUPPORTED_CRYPTOS[self.crypto_symbol]['data_file'])
        df = pd.read_csv(data_file)
        
        print(f"📈 Loaded {len(df)} rows of data")
        print(f"📋 Columns: {list(df.columns)}")
        
        # Ensure timestamp column exists
        if 'timestamp' not in df.columns:
            print("⚠️ No timestamp column found, creating one...")
            df['timestamp'] = pd.date_range(start='2022-01-01', periods=len(df), freq='5T')
        
        # Engineer realistic features
        df = self.feature_engineer.engineer_features(df)
        
        # Calculate targets
        df = self.calculate_targets(df)
        
        print(f"✅ Data preprocessing complete. Final shape: {df.shape}")
        return df
    
    def calculate_targets(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate realistic targets for volatility, skewness, and kurtosis.
        """
        print("🎯 Calculating realistic targets...")
        
        # Calculate rolling statistics for targets
        window = 24  # 2 hours for target calculation
        
        # Volatility target (realized volatility)
        df['target_volatility'] = df['log_return'].rolling(window=window).std()
        
        # Skewness target
        df['target_skewness'] = df['log_return'].rolling(window=window).skew()
        
        # Kurtosis target (excess kurtosis)
        df['target_kurtosis'] = df['log_return'].rolling(window=window).kurt()
        
        # Fill NaN values
        df['target_volatility'] = df['target_volatility'].fillna(method='ffill').fillna(0.01)
        df['target_skewness'] = df['target_skewness'].fillna(method='ffill').fillna(0.0)
        df['target_kurtosis'] = df['target_kurtosis'].fillna(method='ffill').fillna(0.0)
        
        # Apply realistic constraints to targets
        df['target_volatility'] = np.clip(df['target_volatility'], 0.001, 0.5)
        df['target_skewness'] = np.clip(df['target_skewness'], -0.8, 0.8)
        df['target_kurtosis'] = np.clip(df['target_kurtosis'], 0.1, 10.0)
        
        print(f"✅ Targets calculated with realistic constraints")
        return df
    
    def prepare_training_data(self, df: pd.DataFrame) -> tuple:
        """
        Prepare training data with time-aware features.
        """
        print("🔄 Preparing training data...")
        
        # Define feature columns (all numeric columns except targets and timestamp)
        exclude_cols = ['timestamp', 'target_volatility', 'target_skewness', 'target_kurtosis']
        feature_cols = [col for col in df.columns if col not in exclude_cols and df[col].dtype in ['float64', 'int64']]
        
        target_cols = ['target_volatility', 'target_skewness', 'target_kurtosis']
        
        print(f"🎯 Selected {len(feature_cols)} feature columns")
        print(f"🎯 Feature columns: {feature_cols[:10]}...")  # Show first 10
        
        # Fit scalers
        self.feature_engineer.fit_scalers(df, feature_cols, target_cols)
        
        # Transform data
        df_transformed = self.feature_engineer.transform_data(df, feature_cols, target_cols)
        
        # Prepare sequences
        X, y = self.feature_engineer.prepare_sequences(
            df_transformed, feature_cols, target_cols, self.config.SEQUENCE_LENGTH
        )
        
        # Extract time features for the sequences
        time_features = self.extract_time_features(df_transformed, X.shape[0])
        
        # Create train-validation split
        if time_features is not None:
            result = create_train_val_split(
                X, y, time_features, self.config.VALIDATION_SPLIT
            )
            X_train, X_val, y_train, y_val, time_train, time_val = result
        else:
            result = create_train_val_split(
                X, y, validation_split=self.config.VALIDATION_SPLIT
            )
            X_train, X_val, y_train, y_val = result
            time_train, time_val = None, None
        
        print(f"✅ Training data prepared:")
        print(f"   X_train: {X_train.shape}, X_val: {X_val.shape}")
        print(f"   y_train: {y_train.shape}, y_val: {y_val.shape}")
        if time_train is not None:
            print(f"   time_train: {time_train.shape}, time_val: {time_val.shape}")
        
        return X_train, X_val, y_train, y_val, time_train, time_val, feature_cols
    
    def extract_time_features(self, df: pd.DataFrame, n_sequences: int) -> np.ndarray:
        """
        Extract time features for the sequences.
        """
        if 'hour_sin' not in df.columns or 'hour_cos' not in df.columns:
            print("⚠️ No time features found, skipping time-aware training")
            return None
        
        # Extract time features for the sequences
        time_features = []
        for i in range(self.config.SEQUENCE_LENGTH, len(df)):
            seq_time_features = df[['hour_sin', 'hour_cos']].iloc[i-self.config.SEQUENCE_LENGTH:i].values
            time_features.append(seq_time_features)
        
        time_features = np.array(time_features)
        print(f"⏰ Extracted time features: {time_features.shape}")
        return time_features
    
    def create_model(self, input_size: int):
        """
        Create the realistic enhanced model.
        """
        print(f"🏗️ Creating realistic model with input size {input_size}...")
        
        self.model = create_realistic_enhanced_model(self.config)
        self.model = self.model.to(self.device)
        
        # Initialize optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.config.LEARNING_RATE,
            weight_decay=1e-5
        )
        
        # Initialize scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=10
        )
        
        # Initialize loss function
        self.criterion = RealisticLoss(
            volatility_weight=self.config.VOLATILITY_WEIGHT,
            skewness_weight=self.config.SKEWNESS_WEIGHT,
            kurtosis_weight=self.config.KURTOSIS_WEIGHT,
            uncertainty_weight=self.config.UNCERTAINTY_WEIGHT,
            consistency_weight=self.config.CONSISTENCY_WEIGHT,
            regime_weight=self.config.REGIME_WEIGHT
        )
        
        print(f"✅ Model created with {sum(p.numel() for p in self.model.parameters()):,} parameters")
    
    def train_epoch(self, train_loader: DataLoader, time_train_loader: DataLoader = None) -> tuple:
        """
        Train for one epoch with time-aware features.
        """
        self.model.train()
        total_loss = 0
        total_metrics = {'volatility_mse': 0, 'skewness_mse': 0, 'kurtosis_mse': 0}
        
        for batch_idx, (data, targets) in enumerate(train_loader):
            data = data.to(self.device)
            targets = targets.to(self.device)
            
            # Get time features if available
            time_features = None
            if time_train_loader is not None:
                time_features = next(time_train_loader).to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            predictions = self.model(data, time_features)
            
            # Calculate loss
            loss, loss_components = self.criterion(predictions, targets, time_features)
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            # Calculate metrics
            pred_values = predictions['point_predictions']
            with torch.no_grad():
                vol_mse = mean_squared_error(targets[:, 0].cpu(), pred_values[:, 0].cpu())
                skew_mse = mean_squared_error(targets[:, 1].cpu(), pred_values[:, 1].cpu())
                kurt_mse = mean_squared_error(targets[:, 2].cpu(), pred_values[:, 2].cpu())
                
                total_metrics['volatility_mse'] += vol_mse
                total_metrics['skewness_mse'] += skew_mse
                total_metrics['kurtosis_mse'] += kurt_mse
            
            total_loss += loss.item()
            
            if batch_idx % 10 == 0:
                print(f"   Batch {batch_idx}/{len(train_loader)}: Loss = {loss.item():.6f}")
        
        # Average metrics
        num_batches = len(train_loader)
        avg_loss = total_loss / num_batches
        avg_metrics = {k: v / num_batches for k, v in total_metrics.items()}
        
        return avg_loss, avg_metrics
    
    def validate_epoch(self, val_loader: DataLoader, time_val_loader: DataLoader = None) -> tuple:
        """
        Validate for one epoch with time-aware features.
        """
        self.model.eval()
        total_loss = 0
        total_metrics = {'volatility_mse': 0, 'skewness_mse': 0, 'kurtosis_mse': 0}
        
        with torch.no_grad():
            for data, targets in val_loader:
                data = data.to(self.device)
                targets = targets.to(self.device)
                
                # Get time features if available
                time_features = None
                if time_val_loader is not None:
                    time_features = next(time_val_loader).to(self.device)
                
                # Forward pass
                predictions = self.model(data, time_features)
                
                # Calculate loss
                loss, loss_components = self.criterion(predictions, targets, time_features)
                
                # Calculate metrics
                pred_values = predictions['point_predictions']
                vol_mse = mean_squared_error(targets[:, 0].cpu(), pred_values[:, 0].cpu())
                skew_mse = mean_squared_error(targets[:, 1].cpu(), pred_values[:, 1].cpu())
                kurt_mse = mean_squared_error(targets[:, 2].cpu(), pred_values[:, 2].cpu())
                
                total_metrics['volatility_mse'] += vol_mse
                total_metrics['skewness_mse'] += skew_mse
                total_metrics['kurtosis_mse'] += kurt_mse
                
                total_loss += loss.item()
        
        # Average metrics
        num_batches = len(val_loader)
        avg_loss = total_loss / num_batches
        avg_metrics = {k: v / num_batches for k, v in total_metrics.items()}
        
        return avg_loss, avg_metrics
    
    def train(self, X_train: np.ndarray, X_val: np.ndarray, y_train: np.ndarray, y_val: np.ndarray,
              time_train: np.ndarray = None, time_val: np.ndarray = None, feature_cols: list = None):
        """
        Train the realistic model with time-aware features.
        """
        print(f"🚀 Starting realistic model training...")
        
        # Create model
        input_size = X_train.shape[2]
        self.create_model(input_size)
        
        # Create datasets
        train_dataset = RealisticCryptoDataset(X_train, y_train, time_train)
        val_dataset = RealisticCryptoDataset(X_val, y_val, time_val)
        
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=self.config.BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.config.BATCH_SIZE, shuffle=False)
        
        # Create time feature loaders if available
        time_train_loader = None
        time_val_loader = None
        if time_train is not None:
            time_train_dataset = torch.utils.data.TensorDataset(torch.FloatTensor(time_train))
            time_val_dataset = torch.utils.data.TensorDataset(torch.FloatTensor(time_val))
            time_train_loader = DataLoader(time_train_dataset, batch_size=self.config.BATCH_SIZE, shuffle=True)
            time_val_loader = DataLoader(time_val_dataset, batch_size=self.config.BATCH_SIZE, shuffle=False)
        
        # Training loop
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(self.config.NUM_EPOCHS):
            print(f"\n📊 Epoch {epoch+1}/{self.config.NUM_EPOCHS}")
            
            # Train
            train_loss, train_metrics = self.train_epoch(train_loader, time_train_loader)
            
            # Validate
            val_loss, val_metrics = self.validate_epoch(val_loader, time_val_loader)
            
            # Update scheduler
            self.scheduler.step(val_loss)
            
            # Store history
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_metrics.append(train_metrics)
            self.val_metrics.append(val_metrics)
            
            # Print progress
            print(f"   Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
            print(f"   Train Metrics: {train_metrics}")
            print(f"   Val Metrics: {val_metrics}")
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                self.save_model(feature_cols)
                print(f"   💾 New best model saved!")
            else:
                patience_counter += 1
                if patience_counter >= self.config.EARLY_STOPPING_PATIENCE:
                    print(f"   ⏹️ Early stopping triggered after {patience_counter} epochs")
                    break
        
        print(f"✅ Training completed!")
        self.plot_training_history()
        self.save_training_history()
    
    def save_model(self, feature_cols: list = None):
        """
        Save the trained model and metadata.
        """
        # Create models directory
        os.makedirs(self.config.MODEL_SAVE_PATH, exist_ok=True)
        
        # Save model
        model_path = os.path.join(self.config.MODEL_SAVE_PATH, f'{self.crypto_symbol}_realistic_model.pth')
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config,
            'feature_cols': feature_cols,
            'feature_engineer': self.feature_engineer
        }, model_path)
        
        print(f"💾 Model saved to {model_path}")
    
    def plot_training_history(self):
        """
        Plot training history.
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Loss plot
        axes[0, 0].plot(self.train_losses, label='Train Loss')
        axes[0, 0].plot(self.val_losses, label='Val Loss')
        axes[0, 0].set_title('Training and Validation Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Volatility MSE
        train_vol_mse = [m['volatility_mse'] for m in self.train_metrics]
        val_vol_mse = [m['volatility_mse'] for m in self.val_metrics]
        axes[0, 1].plot(train_vol_mse, label='Train Volatility MSE')
        axes[0, 1].plot(val_vol_mse, label='Val Volatility MSE')
        axes[0, 1].set_title('Volatility MSE')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('MSE')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Skewness MSE
        train_skew_mse = [m['skewness_mse'] for m in self.train_metrics]
        val_skew_mse = [m['skewness_mse'] for m in self.val_metrics]
        axes[1, 0].plot(train_skew_mse, label='Train Skewness MSE')
        axes[1, 0].plot(val_skew_mse, label='Val Skewness MSE')
        axes[1, 0].set_title('Skewness MSE')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('MSE')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # Kurtosis MSE
        train_kurt_mse = [m['kurtosis_mse'] for m in self.train_metrics]
        val_kurt_mse = [m['kurtosis_mse'] for m in self.val_metrics]
        axes[1, 1].plot(train_kurt_mse, label='Train Kurtosis MSE')
        axes[1, 1].plot(val_kurt_mse, label='Val Kurtosis MSE')
        axes[1, 1].set_title('Kurtosis MSE')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('MSE')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        
        # Save plot
        os.makedirs(self.config.RESULTS_PATH, exist_ok=True)
        plot_path = os.path.join(self.config.RESULTS_PATH, f'{self.crypto_symbol}_training_history.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"📊 Training history plot saved to {plot_path}")
    
    def save_training_history(self):
        """
        Save training history to JSON.
        """
        history = {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_metrics': self.train_metrics,
            'val_metrics': self.val_metrics,
            'config': {
                'crypto_symbol': self.crypto_symbol,
                'num_epochs': len(self.train_losses),
                'best_val_loss': min(self.val_losses) if self.val_losses else None,
                'final_train_loss': self.train_losses[-1] if self.train_losses else None,
                'final_val_loss': self.val_losses[-1] if self.val_losses else None
            }
        }
        
        os.makedirs(self.config.RESULTS_PATH, exist_ok=True)
        history_path = os.path.join(self.config.RESULTS_PATH, f'{self.crypto_symbol}_training_history.json')
        
        with open(history_path, 'w') as f:
            json.dump(history, f, indent=2, default=str)
        
        print(f"📝 Training history saved to {history_path}")

def main():
    """
    Main training function.
    """
    # Configuration
    config = RealisticConfig()
    
    # Train for each supported crypto
    for crypto_symbol in config.SUPPORTED_CRYPTOS.keys():
        print(f"\n{'='*60}")
        print(f"🚀 Training Realistic Model for {crypto_symbol}")
        print(f"{'='*60}")
        
        try:
            # Initialize trainer
            trainer = RealisticModelTrainer(config, crypto_symbol)
            
            # Load and preprocess data
            df = trainer.load_and_preprocess_data()
            
            # Prepare training data
            X_train, X_val, y_train, y_val, time_train, time_val, feature_cols = trainer.prepare_training_data(df)
            
            # Train model
            trainer.train(X_train, X_val, y_train, y_val, time_train, time_val, feature_cols)
            
            print(f"✅ Training completed for {crypto_symbol}")
            
        except Exception as e:
            print(f"❌ Error training {crypto_symbol}: {str(e)}")
            continue

if __name__ == "__main__":
    main() 
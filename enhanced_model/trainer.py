#!/usr/bin/env python3
"""
Enhanced Trainer for Monte Carlo Simulation

This module provides training capabilities for the enhanced model architecture
specifically designed for Monte Carlo simulation with better statistical moment prediction.
"""

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

from config import EnhancedConfig
from enhanced_model import EnhancedVolatilityModel, EnhancedVolatilityLoss, create_enhanced_model
from feature_engineering import EnhancedFeatureEngineer, EnhancedCryptoDataset, create_train_val_split
from data_processor import EnhancedCryptoDataProcessor

class EnhancedCryptoVolatilityTrainer:
    """
    Enhanced training pipeline for cryptocurrency volatility prediction model.
    Optimized for Monte Carlo simulation with better statistical moment prediction.
    """
    
    def __init__(self, config: EnhancedConfig, crypto_symbol: str = 'BTC'):
        self.config = config
        self.crypto_symbol = crypto_symbol
        self.crypto_config = EnhancedConfig.SUPPORTED_CRYPTOS[crypto_symbol]
        self.device = config.DEVICE
        self.model = None
        self.feature_engineer = EnhancedFeatureEngineer()
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        
        # Create directories
        os.makedirs(config.MODEL_SAVE_PATH, exist_ok=True)
        os.makedirs(config.RESULTS_PATH, exist_ok=True)
        
        print(f"Enhanced training for {self.crypto_config['name']} ({crypto_symbol}) on {self.device}")
    
    def prepare_data(self, csv_path: str) -> Tuple[DataLoader, DataLoader, List[str], List[str]]:
        """
        Prepare training and validation data loaders.
        """
        # Load and preprocess data
        processor = EnhancedCryptoDataProcessor(csv_path, self.crypto_symbol)
        df = processor.preprocess_data(
            return_windows=self.config.RETURN_WINDOWS,
            prediction_horizon=self.config.PREDICTION_HORIZON
        )
        
        # Add additional features
        df = self.feature_engineer.engineer_features(df)
        
        # Remove any remaining NaN values
        df = df.dropna().reset_index(drop=True)
        
        # Get feature and target columns
        feature_cols = [col for col in df.columns if col not in 
                       ['timestamp', 'target_volatility', 'target_skewness', 'target_kurtosis']]
        target_cols = ['target_volatility', 'target_skewness', 'target_kurtosis']
        
        print(f"Enhanced dataset: {df.shape} | Features: {len(feature_cols)}")
        
        # Adaptive sequence length for limited data
        if len(df) < 200:
            # Very limited data - use shorter sequence
            adaptive_sequence_length = max(12, len(df) // 10)
            print(f"⚠️ Limited data ({len(df)} rows). Using adaptive sequence length: {adaptive_sequence_length}")
            self.config.SEQUENCE_LENGTH = adaptive_sequence_length
        elif len(df) < 500:
            # Limited data - use medium sequence
            adaptive_sequence_length = max(48, len(df) // 8)
            print(f"⚠️ Limited data ({len(df)} rows). Using adaptive sequence length: {adaptive_sequence_length}")
            self.config.SEQUENCE_LENGTH = adaptive_sequence_length
        
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
        
        # Train/validation split
        X_train, X_val, y_train, y_val = create_train_val_split(
            X, y, self.config.VALIDATION_SPLIT
        )
        
        # Create datasets and data loaders
        train_dataset = EnhancedCryptoDataset(X_train, y_train)
        val_dataset = EnhancedCryptoDataset(X_val, y_val)
        
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
        total_quantile_loss = 0.0
        num_batches = 0
        
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(self.device)
            batch_y = batch_y.to(self.device)
            
            optimizer.zero_grad()
            
            # Forward pass
            predictions = model(batch_x)
            loss, loss_components = criterion(predictions, batch_y)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            # Accumulate losses
            total_loss += loss.item()
            total_vol_loss += loss_components['volatility_loss'].item()
            total_skew_loss += loss_components['skewness_loss'].item()
            total_kurt_loss += loss_components['kurtosis_loss'].item()
            total_quantile_loss += loss_components['quantile_loss'].item()
            num_batches += 1
        
        return {
            'total_loss': total_loss / num_batches,
            'volatility_loss': total_vol_loss / num_batches,
            'skewness_loss': total_skew_loss / num_batches,
            'kurtosis_loss': total_kurt_loss / num_batches,
            'quantile_loss': total_quantile_loss / num_batches
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
        total_quantile_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                
                # Forward pass
                predictions = model(batch_x)
                loss, loss_components = criterion(predictions, batch_y)
                
                # Accumulate losses
                total_loss += loss.item()
                total_vol_loss += loss_components['volatility_loss'].item()
                total_skew_loss += loss_components['skewness_loss'].item()
                total_kurt_loss += loss_components['kurtosis_loss'].item()
                total_quantile_loss += loss_components['quantile_loss'].item()
                num_batches += 1
        
        return {
            'total_loss': total_loss / num_batches,
            'volatility_loss': total_vol_loss / num_batches,
            'skewness_loss': total_skew_loss / num_batches,
            'kurtosis_loss': total_kurt_loss / num_batches,
            'quantile_loss': total_quantile_loss / num_batches
        }
    
    def train(self, csv_path: str) -> Dict:
        """
        Train the enhanced model.
        
        Args:
            csv_path: Path to the CSV file containing OHLC data
            
        Returns:
            Dictionary containing training history
        """
        print(f"🚀 Starting enhanced training for {self.crypto_symbol}")
        
        # Prepare data
        train_loader, val_loader, feature_cols, target_cols = self.prepare_data(csv_path)
        
        # Initialize model
        self.model = create_enhanced_model(self.config)
        self.model.to(self.device)
        
        # Initialize optimizer and criterion
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.LEARNING_RATE)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5, verbose=True
        )
        criterion = EnhancedVolatilityLoss()
        
        # Training history
        history = {
            'train_losses': [],
            'val_losses': [],
            'train_vol_losses': [],
            'val_vol_losses': [],
            'train_skew_losses': [],
            'val_skew_losses': [],
            'train_kurt_losses': [],
            'val_kurt_losses': [],
            'train_quantile_losses': [],
            'val_quantile_losses': []
        }
        
        print(f"📊 Training on {len(train_loader.dataset)} samples, validating on {len(val_loader.dataset)} samples")
        
        # Training loop
        for epoch in range(self.config.NUM_EPOCHS):
            # Train
            train_metrics = self.train_epoch(self.model, train_loader, optimizer, criterion)
            
            # Validate
            val_metrics = self.validate_epoch(self.model, val_loader, criterion)
            
            # Update learning rate
            scheduler.step(val_metrics['total_loss'])
            
            # Store history
            history['train_losses'].append(train_metrics['total_loss'])
            history['val_losses'].append(val_metrics['total_loss'])
            history['train_vol_losses'].append(train_metrics['volatility_loss'])
            history['val_vol_losses'].append(val_metrics['volatility_loss'])
            history['train_skew_losses'].append(train_metrics['skewness_loss'])
            history['val_skew_losses'].append(val_metrics['skewness_loss'])
            history['train_kurt_losses'].append(train_metrics['kurtosis_loss'])
            history['val_kurt_losses'].append(val_metrics['kurtosis_loss'])
            history['train_quantile_losses'].append(train_metrics['quantile_loss'])
            history['val_quantile_losses'].append(val_metrics['quantile_loss'])
            
            # Print progress
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{self.config.NUM_EPOCHS}")
                print(f"  Train Loss: {train_metrics['total_loss']:.6f}")
                print(f"  Val Loss: {val_metrics['total_loss']:.6f}")
                print(f"  Vol Loss: {val_metrics['volatility_loss']:.6f}")
                print(f"  Skew Loss: {val_metrics['skewness_loss']:.6f}")
                print(f"  Kurt Loss: {val_metrics['kurtosis_loss']:.6f}")
                print(f"  Quantile Loss: {val_metrics['quantile_loss']:.6f}")
            
            # Early stopping
            if val_metrics['total_loss'] < self.best_val_loss:
                self.best_val_loss = val_metrics['total_loss']
                self.patience_counter = 0
                
                # Save best model
                self.save_model(epoch, train_metrics, val_metrics, feature_cols, target_cols)
            else:
                self.patience_counter += 1
                
            if self.patience_counter >= self.config.EARLY_STOPPING_PATIENCE:
                print(f"Early stopping at epoch {epoch+1}")
                break
        
        # Save training history
        self.save_training_history(history)
        
        # Plot training history
        self.plot_training_history(history)
        
        print(f"✅ Enhanced training completed for {self.crypto_symbol}")
        print(f"Best validation loss: {self.best_val_loss:.6f}")
        
        return {
            'final_train_loss': history['train_losses'][-1],
            'final_val_loss': history['val_losses'][-1],
            'best_val_loss': self.best_val_loss,
            'epochs_trained': len(history['train_losses'])
        }
    
    def save_model(self, epoch: int, train_metrics: Dict, val_metrics: Dict, 
                  feature_cols: List[str], target_cols: List[str]) -> None:
        """
        Save the enhanced model and feature engineer.
        """
        # Save model
        model_path = os.path.join(self.config.MODEL_SAVE_PATH, f"{self.crypto_symbol}_enhanced_model.pth")
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'epoch': epoch,
            'train_metrics': train_metrics,
            'val_metrics': val_metrics,
            'feature_cols': feature_cols,
            'target_cols': target_cols,
            'config': self.config.__dict__
        }, model_path)
        
        # Save feature engineer
        feature_engineer_path = os.path.join(self.config.MODEL_SAVE_PATH, f"{self.crypto_symbol}_enhanced_feature_engineer.pkl")
        with open(feature_engineer_path, 'wb') as f:
            pickle.dump(self.feature_engineer, f)
        
        # Save metadata
        metadata = {
            'crypto_symbol': self.crypto_symbol,
            'model_type': 'enhanced',
            'epoch': epoch,
            'train_loss': train_metrics['total_loss'],
            'val_loss': val_metrics['total_loss'],
            'feature_count': len(feature_cols),
            'target_count': len(target_cols),
            'last_updated': datetime.now().isoformat()
        }
        
        metadata_path = os.path.join(self.config.MODEL_SAVE_PATH, f"{self.crypto_symbol}_enhanced_metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"💾 Enhanced model saved: {model_path}")
    
    def save_training_history(self, history: Dict):
        """Save training history."""
        history_path = os.path.join(self.config.RESULTS_PATH, f"{self.crypto_symbol}_enhanced_training_history.json")
        with open(history_path, 'w') as f:
            json.dump(history, f, indent=2)
    
    def plot_training_history(self, history: Dict):
        """Plot training history."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Total loss
        axes[0, 0].plot(history['train_losses'], label='Train')
        axes[0, 0].plot(history['val_losses'], label='Validation')
        axes[0, 0].set_title('Total Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Volatility loss
        axes[0, 1].plot(history['train_vol_losses'], label='Train')
        axes[0, 1].plot(history['val_vol_losses'], label='Validation')
        axes[0, 1].set_title('Volatility Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Skewness loss
        axes[1, 0].plot(history['train_skew_losses'], label='Train')
        axes[1, 0].plot(history['val_skew_losses'], label='Validation')
        axes[1, 0].set_title('Skewness Loss')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # Kurtosis loss
        axes[1, 1].plot(history['train_kurt_losses'], label='Train')
        axes[1, 1].plot(history['val_kurt_losses'], label='Validation')
        axes[1, 1].set_title('Kurtosis Loss')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        plot_path = os.path.join(self.config.RESULTS_PATH, f"{self.crypto_symbol}_enhanced_training_history.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📈 Training plots saved: {plot_path}")

def main():
    """Main training function."""
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python trainer.py <crypto_symbol>")
        print("Supported cryptos: BTC, ETH, XAU, SOL")
        sys.exit(1)
    
    crypto_symbol = sys.argv[1].upper()
    
    if crypto_symbol not in EnhancedConfig.SUPPORTED_CRYPTOS:
        print(f"Unsupported crypto: {crypto_symbol}")
        print(f"Supported: {list(EnhancedConfig.SUPPORTED_CRYPTOS.keys())}")
        sys.exit(1)
    
    # Initialize config and trainer
    config = EnhancedConfig()
    trainer = EnhancedCryptoVolatilityTrainer(config, crypto_symbol)
    
    # Get data path
    data_file = EnhancedConfig.SUPPORTED_CRYPTOS[crypto_symbol]['data_file']
    csv_path = os.path.join(config.DATA_PATH, data_file)
    
    if not os.path.exists(csv_path):
        print(f"Data file not found: {csv_path}")
        sys.exit(1)
    
    # Train model
    try:
        results = trainer.train(csv_path)
        print(f"✅ Training completed successfully!")
        print(f"Final validation loss: {results['final_val_loss']:.6f}")
    except Exception as e:
        print(f"❌ Training failed: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main() 
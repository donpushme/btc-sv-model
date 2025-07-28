#!/usr/bin/env python3
"""
Enhanced Model Architecture for Monte Carlo Simulation

This module provides an improved neural network architecture specifically
designed for predicting statistical moments used in Monte Carlo simulation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, List
import numpy as np

class TransformerBlock(nn.Module):
    """Transformer block for capturing long-range dependencies."""
    
    def __init__(self, hidden_size: int, num_heads: int = 8, dropout: float = 0.1):
        super(TransformerBlock, self).__init__()
        self.attention = nn.MultiheadAttention(hidden_size, num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.feed_forward = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 4, hidden_size)
        )
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Self-attention
        attn_output, _ = self.attention(x, x, x)
        x = self.norm1(x + self.dropout(attn_output))
        
        # Feed-forward
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        
        return x

class EnhancedVolatilityModel(nn.Module):
    """
    Enhanced model architecture optimized for Monte Carlo simulation.
    
    Features:
    - Hybrid LSTM-Transformer architecture
    - Quantile regression for better tail prediction
    - Uncertainty quantification
    - Multi-scale feature processing
    """
    
    def __init__(self, input_size: int, hidden_size: int = 128, num_layers: int = 3,
                 num_heads: int = 8, dropout: float = 0.2, num_quantiles: int = 5):
        super(EnhancedVolatilityModel, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_quantiles = num_quantiles
        
        # Multi-scale feature processing
        self.feature_conv = nn.ModuleList([
            nn.Conv1d(input_size, hidden_size, kernel_size=k, padding=k//2)
            for k in [3, 5, 7]  # Different temporal scales
        ])
        
        # LSTM for temporal dependencies
        self.lstm = nn.LSTM(
            input_size=hidden_size * 3,  # Concatenated conv outputs
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
            bidirectional=True
        )
        
        # Transformer blocks for long-range dependencies
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(hidden_size * 2, num_heads, dropout)  # *2 for bidirectional
            for _ in range(2)
        ])
        
        # Uncertainty quantification
        self.uncertainty_head = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 3)  # Uncertainty for each moment
        )
        
        # Quantile regression heads
        self.quantile_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_size * 2, hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_size, 3)  # volatility, skewness, kurtosis
            ) for _ in range(num_quantiles)
        ])
        
        # Point prediction heads (mean)
        self.volatility_head = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1),
            nn.Softplus()  # Ensure positive volatility
        )
        
        self.skewness_head = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1),
            nn.Tanh()  # Bound skewness to [-1, 1]
        )
        
        self.kurtosis_head = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1),
            nn.Softplus()  # Ensure positive kurtosis
        )
        
        self.dropout = nn.Dropout(dropout)
        self.init_weights()
    
    def init_weights(self):
        """Initialize model weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LSTM):
                for name, param in module.named_parameters():
                    if 'weight_ih' in name:
                        nn.init.xavier_uniform_(param.data)
                    elif 'weight_hh' in name:
                        nn.init.orthogonal_(param.data)
                    elif 'bias' in name:
                        param.data.fill_(0)
                        n = param.size(0)
                        param.data[(n//4):(n//2)].fill_(1)
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass with enhanced features.
        
        Args:
            x: Input tensor (batch_size, sequence_length, input_size)
            
        Returns:
            Dictionary containing predictions and uncertainty
        """
        batch_size, seq_len, _ = x.size()
        
        # Multi-scale feature processing
        x_permuted = x.permute(0, 2, 1)  # (batch, features, seq_len)
        conv_outputs = []
        
        for conv in self.feature_conv:
            conv_out = conv(x_permuted)  # (batch, hidden_size, seq_len)
            conv_outputs.append(conv_out)
        
        # Concatenate multi-scale features
        multi_scale_features = torch.cat(conv_outputs, dim=1)  # (batch, hidden_size*3, seq_len)
        multi_scale_features = multi_scale_features.permute(0, 2, 1)  # (batch, seq_len, hidden_size*3)
        
        # LSTM processing
        lstm_out, _ = self.lstm(multi_scale_features)  # (batch, seq_len, hidden_size*2)
        
        # Transformer processing
        transformer_out = lstm_out
        for transformer in self.transformer_blocks:
            transformer_out = transformer(transformer_out)
        
        # Global average pooling
        pooled = torch.mean(transformer_out, dim=1)  # (batch, hidden_size*2)
        pooled = self.dropout(pooled)
        
        # Point predictions (mean)
        volatility = self.volatility_head(pooled)
        skewness = self.skewness_head(pooled) * 2.0  # Scale to [-2, 2]
        kurtosis = self.kurtosis_head(pooled) * 10.0  # Scale to [0, 10]
        
        # Quantile predictions
        quantile_predictions = []
        for head in self.quantile_heads:
            quantile_pred = head(pooled)
            quantile_pred[:, 0] = F.softplus(quantile_pred[:, 0])  # Volatility
            quantile_pred[:, 1] = torch.tanh(quantile_pred[:, 1]) * 2.0  # Skewness
            quantile_pred[:, 2] = F.softplus(quantile_pred[:, 2]) * 10.0  # Kurtosis
            quantile_predictions.append(quantile_pred)
        
        # Uncertainty quantification
        uncertainty = F.softplus(self.uncertainty_head(pooled))
        
        # Combine predictions
        point_predictions = torch.cat([volatility, skewness, kurtosis], dim=1)
        
        return {
            'point_predictions': point_predictions,
            'quantile_predictions': torch.stack(quantile_predictions, dim=1),
            'uncertainty': uncertainty,
            'pooled_features': pooled
        }

class QuantileLoss(nn.Module):
    """
    Quantile loss for better tail prediction in Monte Carlo simulation.
    """
    
    def __init__(self, quantiles: List[float] = [0.1, 0.25, 0.5, 0.75, 0.9]):
        super(QuantileLoss, self).__init__()
        self.register_buffer('quantiles', torch.tensor(quantiles))
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Calculate quantile loss.
        
        Args:
            predictions: (batch_size, num_quantiles, num_targets)
            targets: (batch_size, num_targets)
        """
        batch_size, num_quantiles, num_targets = predictions.size()
        
        # Ensure quantiles tensor is on the same device as predictions
        quantiles = self.quantiles.to(predictions.device)
        
        # Expand targets for broadcasting
        targets_expanded = targets.unsqueeze(1).expand(-1, num_quantiles, -1)
        
        # Calculate quantile loss
        diff = targets_expanded - predictions
        loss = torch.where(
            diff >= 0,
            quantiles.unsqueeze(0).unsqueeze(-1) * diff,
            (quantiles.unsqueeze(0).unsqueeze(-1) - 1) * diff
        )
        
        return loss.mean()

class EnhancedVolatilityLoss(nn.Module):
    """
    Enhanced loss function for Monte Carlo simulation.
    """
    
    def __init__(self, volatility_weight: float = 2.0, skewness_weight: float = 1.0,
                 kurtosis_weight: float = 1.5, quantile_weight: float = 0.5,
                 uncertainty_weight: float = 0.1):
        super(EnhancedVolatilityLoss, self).__init__()
        self.volatility_weight = volatility_weight
        self.skewness_weight = skewness_weight
        self.kurtosis_weight = kurtosis_weight
        self.quantile_weight = quantile_weight
        self.uncertainty_weight = uncertainty_weight
        self.quantile_loss = QuantileLoss()
    
    def forward(self, predictions: Dict[str, torch.Tensor], 
                targets: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Calculate enhanced loss.
        """
        point_pred = predictions['point_predictions']
        quantile_pred = predictions['quantile_predictions']
        uncertainty = predictions['uncertainty']
        
        # Split predictions and targets
        pred_vol, pred_skew, pred_kurt = point_pred[:, 0], point_pred[:, 1], point_pred[:, 2]
        target_vol, target_skew, target_kurt = targets[:, 0], targets[:, 1], targets[:, 2]
        
        # Point prediction losses
        vol_loss = F.mse_loss(pred_vol, target_vol)
        skew_loss = F.mse_loss(pred_skew, target_skew)
        kurt_loss = F.huber_loss(pred_kurt, target_kurt, delta=1.0)
        
        # Quantile loss
        quantile_loss = self.quantile_loss(quantile_pred, targets)
        
        # Uncertainty regularization (encourage reasonable uncertainty)
        uncertainty_reg = torch.mean(uncertainty**2)
        
        # Total loss
        total_loss = (
            self.volatility_weight * vol_loss +
            self.skewness_weight * skew_loss +
            self.kurtosis_weight * kurt_loss +
            self.quantile_weight * quantile_loss +
            self.uncertainty_weight * uncertainty_reg
        )
        
        loss_components = {
            'total_loss': total_loss,
            'volatility_loss': vol_loss,
            'skewness_loss': skew_loss,
            'kurtosis_loss': kurt_loss,
            'quantile_loss': quantile_loss,
            'uncertainty_reg': uncertainty_reg
        }
        
        return total_loss, loss_components

def create_enhanced_model(config) -> EnhancedVolatilityModel:
    """Create enhanced model for Monte Carlo simulation."""
    model = EnhancedVolatilityModel(
        input_size=config.INPUT_SIZE,
        hidden_size=config.HIDDEN_SIZE,
        num_layers=config.NUM_LAYERS,
        num_heads=8,
        dropout=config.DROPOUT,
        num_quantiles=5
    )
    
    print(f"Enhanced model created with {sum(p.numel() for p in model.parameters()):,} parameters")
    return model 
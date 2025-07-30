#!/usr/bin/env python3
"""
Realistic Enhanced Model Architecture for Monte Carlo Simulation

This module provides a professional neural network architecture specifically
designed for realistic cryptocurrency price prediction with emphasis on:
- Time-of-day patterns (US/Asian trading hours)
- Market microstructure (bid-ask spreads, volume patterns)
- Regime detection (volatility clustering, trend vs mean-reversion)
- Multi-scale analysis (short-term vs long-term patterns)
- Realistic constraints and temporal consistency
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, List, Optional
import numpy as np

class TimeAwareAttention(nn.Module):
    """
    Time-aware attention mechanism that considers time-of-day patterns.
    """
    
    def __init__(self, hidden_size: int, num_heads: int = 8, dropout: float = 0.1):
        super(TimeAwareAttention, self).__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        
        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)
        self.time_embedding = nn.Linear(2, hidden_size)  # hour_sin, hour_cos
        self.output = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor, time_features: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        
        # Time-aware embeddings
        time_emb = self.time_embedding(time_features)
        x_with_time = x + time_emb
        
        # Multi-head attention
        q = self.query(x_with_time).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.key(x_with_time).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.value(x_with_time).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(self.head_dim)
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention
        context = torch.matmul(attn_weights, v)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_size)
        
        return self.output(context)

class MarketRegimeDetector(nn.Module):
    """
    Detects market regimes (trending, mean-reverting, high/low volatility).
    """
    
    def __init__(self, hidden_size: int, num_regimes: int = 4):
        super(MarketRegimeDetector, self).__init__()
        self.num_regimes = num_regimes
        
        self.regime_classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size // 2, num_regimes),
            nn.Softmax(dim=-1)
        )
        
        # Regime-specific processing
        self.regime_processors = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_size, hidden_size // 2),
                nn.ReLU(),
                nn.Dropout(0.1)
            ) for _ in range(num_regimes)
        ])
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Detect regime probabilities
        regime_probs = self.regime_classifier(x)
        
        # Process features for each regime
        regime_features = []
        for i, processor in enumerate(self.regime_processors):
            regime_feat = processor(x)
            regime_features.append(regime_feat)
        
        # Weighted combination of regime features
        regime_features = torch.stack(regime_features, dim=1)  # (batch, num_regimes, features)
        weighted_features = torch.sum(regime_features * regime_probs.unsqueeze(-1), dim=1)
        
        return weighted_features, regime_probs

class MultiScaleProcessor(nn.Module):
    """
    Multi-scale feature processing for different temporal patterns.
    """
    
    def __init__(self, input_size: int, hidden_size: int):
        super(MultiScaleProcessor, self).__init__()
        
        # Short-term patterns (5-15 minutes)
        self.short_term = nn.Sequential(
            nn.Conv1d(input_size, hidden_size // 4, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(hidden_size // 4, hidden_size // 4, kernel_size=5, padding=2),
            nn.ReLU()
        )
        
        # Medium-term patterns (30-60 minutes)
        self.medium_term = nn.Sequential(
            nn.Conv1d(input_size, hidden_size // 4, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.Conv1d(hidden_size // 4, hidden_size // 4, kernel_size=11, padding=5),
            nn.ReLU()
        )
        
        # Long-term patterns (2-4 hours)
        self.long_term = nn.Sequential(
            nn.Conv1d(input_size, hidden_size // 4, kernel_size=15, padding=7),
            nn.ReLU(),
            nn.Conv1d(hidden_size // 4, hidden_size // 4, kernel_size=25, padding=12),
            nn.ReLU()
        )
        
        # Ultra-long-term patterns (daily)
        self.ultra_long_term = nn.Sequential(
            nn.Conv1d(input_size, hidden_size // 4, kernel_size=35, padding=17),
            nn.ReLU(),
            nn.Conv1d(hidden_size // 4, hidden_size // 4, kernel_size=50, padding=25),
            nn.ReLU()
        )
        
        # Combine multi-scale features
        self.combiner = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Reshape for 1D convolution
        x_conv = x.transpose(1, 2)  # (batch, features, seq_len)
        
        # Process at different scales
        short = self.short_term(x_conv)
        medium = self.medium_term(x_conv)
        long = self.long_term(x_conv)
        ultra = self.ultra_long_term(x_conv)
        
        # Concatenate multi-scale features
        multi_scale = torch.cat([short, medium, long, ultra], dim=1)
        multi_scale = multi_scale.transpose(1, 2)  # (batch, seq_len, features)
        
        return self.combiner(multi_scale)

class TemporalConsistencyLayer(nn.Module):
    """
    Enhanced temporal consistency layer with realistic constraints.
    """
    
    def __init__(self, hidden_size: int, smoothing_factor: float = 0.1):
        super(TemporalConsistencyLayer, self).__init__()
        self.smoothing_factor = smoothing_factor
        
        # Temporal smoothing with different kernels
        self.smooth_short = nn.Conv1d(hidden_size, hidden_size, kernel_size=3, padding=1)
        self.smooth_medium = nn.Conv1d(hidden_size, hidden_size, kernel_size=5, padding=2)
        self.smooth_long = nn.Conv1d(hidden_size, hidden_size, kernel_size=7, padding=3)
        
        # Adaptive smoothing weights
        self.smoothing_weights = nn.Sequential(
            nn.Linear(hidden_size, 3),
            nn.Softmax(dim=-1)
        )
        
        self.norm = nn.LayerNorm(hidden_size)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, hidden_size = x.shape
        
        # Reshape for 1D convolution
        x_conv = x.transpose(1, 2)  # (batch, hidden, seq)
        
        # Apply different smoothing kernels
        smooth_short = self.smooth_short(x_conv)
        smooth_medium = self.smooth_medium(x_conv)
        smooth_long = self.smooth_long(x_conv)
        
        # Calculate adaptive weights
        weights = self.smoothing_weights(x)  # (batch, seq, 3)
        weights = weights.transpose(1, 2)  # (batch, 3, seq)
        
        # Weighted combination
        smoothed = (weights[:, 0:1, :] * smooth_short + 
                   weights[:, 1:2, :] * smooth_medium + 
                   weights[:, 2:3, :] * smooth_long)
        
        smoothed = smoothed.transpose(1, 2)  # (batch, seq, hidden)
        smoothed = self.norm(smoothed)
        
        # Blend original and smoothed features
        output = (1 - self.smoothing_factor) * x + self.smoothing_factor * smoothed
        
        return output

class RealisticMomentPredictor(nn.Module):
    """
    Specialized predictor for realistic statistical moments with constraints.
    """
    
    def __init__(self, hidden_size: int, dropout: float = 0.2):
        super(RealisticMomentPredictor, self).__init__()
        
        # Shared feature extractor
        self.feature_extractor = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Volatility predictor (always positive, realistic ranges)
        self.volatility_predictor = nn.Sequential(
            nn.Linear(hidden_size // 2, hidden_size // 4),
            nn.ReLU(),
            nn.Linear(hidden_size // 4, 1),
            nn.Softplus()  # Ensure positive
        )
        
        # Skewness predictor (bounded, realistic ranges)
        self.skewness_predictor = nn.Sequential(
            nn.Linear(hidden_size // 2, hidden_size // 4),
            nn.ReLU(),
            nn.Linear(hidden_size // 4, 1),
            nn.Tanh()  # Constrain to [-1, 1]
        )
        
        # Kurtosis predictor (positive, realistic ranges)
        self.kurtosis_predictor = nn.Sequential(
            nn.Linear(hidden_size // 2, hidden_size // 4),
            nn.ReLU(),
            nn.Linear(hidden_size // 4, 1),
            nn.Softplus()  # Ensure positive
        )
        
        # Uncertainty predictors
        self.volatility_uncertainty = nn.Sequential(
            nn.Linear(hidden_size // 2, hidden_size // 4),
            nn.ReLU(),
            nn.Linear(hidden_size // 4, 1),
            nn.Softplus()
        )
        
        self.skewness_uncertainty = nn.Sequential(
            nn.Linear(hidden_size // 2, hidden_size // 4),
            nn.ReLU(),
            nn.Linear(hidden_size // 4, 1),
            nn.Softplus()
        )
        
        self.kurtosis_uncertainty = nn.Sequential(
            nn.Linear(hidden_size // 2, hidden_size // 4),
            nn.ReLU(),
            nn.Linear(hidden_size // 4, 1),
            nn.Softplus()
        )
        
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        features = self.feature_extractor(x)
        
        # Predict moments
        volatility = self.volatility_predictor(features)
        skewness = self.skewness_predictor(features)
        kurtosis = self.kurtosis_predictor(features)
        
        # Predict uncertainties
        vol_uncertainty = self.volatility_uncertainty(features)
        skew_uncertainty = self.skewness_uncertainty(features)
        kurt_uncertainty = self.kurtosis_uncertainty(features)
        
        return {
            'volatility': volatility,
            'skewness': skewness,
            'kurtosis': kurtosis,
            'volatility_uncertainty': vol_uncertainty,
            'skewness_uncertainty': skew_uncertainty,
            'kurtosis_uncertainty': kurt_uncertainty
        }

class RealisticEnhancedModel(nn.Module):
    """
    Realistic enhanced model architecture optimized for cryptocurrency prediction.
    
    Key features:
    - Time-aware attention for trading hour patterns
    - Market regime detection
    - Multi-scale feature processing
    - Realistic moment prediction with constraints
    - Temporal consistency for smooth predictions
    """
    
    def __init__(self, input_size: int, hidden_size: int = 256, num_layers: int = 4,
                 num_heads: int = 8, dropout: float = 0.2):
        super(RealisticEnhancedModel, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Multi-scale feature processing
        self.multi_scale_processor = MultiScaleProcessor(input_size, hidden_size)
        
        # Bidirectional LSTM for temporal dependencies
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size // 2,  # Bidirectional will double this
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
            bidirectional=True
        )
        
        # Time-aware attention layers
        self.time_attention_layers = nn.ModuleList([
            TimeAwareAttention(hidden_size, num_heads, dropout)
            for _ in range(2)
        ])
        
        # Market regime detector
        self.regime_detector = MarketRegimeDetector(hidden_size)
        
        # Temporal consistency layer
        self.temporal_consistency = TemporalConsistencyLayer(hidden_size, smoothing_factor=0.15)
        
        # Realistic moment predictor
        self.moment_predictor = RealisticMomentPredictor(hidden_size, dropout)
        
        # Initialize weights
        self.init_weights()
    
    def init_weights(self):
        """Initialize weights for better training stability."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LSTM):
                for name, param in module.named_parameters():
                    if 'weight' in name:
                        nn.init.xavier_uniform_(param)
                    elif 'bias' in name:
                        nn.init.zeros_(param)
            elif isinstance(module, nn.Conv1d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, x: torch.Tensor, time_features: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Forward pass with realistic architecture.
        
        Args:
            x: Input tensor of shape (batch_size, sequence_length, input_size)
            time_features: Time features tensor of shape (batch_size, sequence_length, 2) [hour_sin, hour_cos]
            
        Returns:
            Dictionary containing predictions and uncertainties
        """
        batch_size, seq_len, _ = x.shape
        
        # Multi-scale feature processing
        multi_scale_features = self.multi_scale_processor(x)
        
        # LSTM processing
        lstm_out, _ = self.lstm(multi_scale_features)  # (batch, seq_len, hidden_size)
        
        # Time-aware attention processing
        attention_out = lstm_out
        if time_features is not None:
            for attention_layer in self.time_attention_layers:
                attention_out = attention_layer(attention_out, time_features)
        
        # Market regime detection
        regime_features, regime_probs = self.regime_detector(attention_out)
        
        # Temporal consistency smoothing
        smoothed_features = self.temporal_consistency(regime_features)
        
        # Global average pooling
        pooled = torch.mean(smoothed_features, dim=1)  # (batch, hidden_size)
        
        # Predict realistic moments
        moment_predictions = self.moment_predictor(pooled)
        
        # Combine predictions
        point_predictions = torch.cat([
            moment_predictions['volatility'],
            moment_predictions['skewness'],
            moment_predictions['kurtosis']
        ], dim=1)
        
        uncertainty = torch.cat([
            moment_predictions['volatility_uncertainty'],
            moment_predictions['skewness_uncertainty'],
            moment_predictions['kurtosis_uncertainty']
        ], dim=1)
        
        return {
            'point_predictions': point_predictions,
            'uncertainty': uncertainty,
            'regime_probabilities': regime_probs,
            'pooled_features': pooled,
            'moment_predictions': moment_predictions
        }

class RealisticLoss(nn.Module):
    """
    Realistic loss function with temporal consistency and market-aware constraints.
    """
    
    def __init__(self, volatility_weight: float = 2.0, skewness_weight: float = 1.0,
                 kurtosis_weight: float = 1.5, uncertainty_weight: float = 0.1,
                 consistency_weight: float = 0.3, regime_weight: float = 0.2):
        super(RealisticLoss, self).__init__()
        self.volatility_weight = volatility_weight
        self.skewness_weight = skewness_weight
        self.kurtosis_weight = kurtosis_weight
        self.uncertainty_weight = uncertainty_weight
        self.consistency_weight = consistency_weight
        self.regime_weight = regime_weight
    
    def forward(self, predictions: Dict[str, torch.Tensor], 
                targets: torch.Tensor, time_features: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Calculate realistic loss with market-aware constraints.
        """
        point_pred = predictions['point_predictions']
        uncertainty = predictions['uncertainty']
        moment_preds = predictions['moment_predictions']
        
        # Split predictions and targets
        pred_vol, pred_skew, pred_kurt = point_pred[:, 0], point_pred[:, 1], point_pred[:, 2]
        target_vol, target_skew, target_kurt = targets[:, 0], targets[:, 1], targets[:, 2]
        
        # Basic prediction losses
        vol_loss = F.mse_loss(pred_vol, target_vol)
        skew_loss = F.mse_loss(pred_skew, target_skew)
        kurt_loss = F.huber_loss(pred_kurt, target_kurt, delta=2.0)
        
        # Realistic constraints
        # Volatility should be positive and reasonable
        vol_constraint = torch.mean(F.relu(-pred_vol + 0.001))  # Min volatility
        vol_constraint += torch.mean(F.relu(pred_vol - 0.5))    # Max volatility
        
        # Skewness should be in realistic range
        skew_constraint = torch.mean(F.relu(torch.abs(pred_skew) - 0.8))  # Max skewness
        
        # Kurtosis should be positive and reasonable
        kurt_constraint = torch.mean(F.relu(-pred_kurt + 0.1))   # Min kurtosis
        kurt_constraint += torch.mean(F.relu(pred_kurt - 10.0))  # Max kurtosis
        
        # Time-aware constraints (if time features available)
        time_constraint = torch.tensor(0.0, device=pred_vol.device)
        if time_features is not None:
            # Extract hour information
            hour_sin = time_features[:, -1, 0]  # Last timestep
            hour_cos = time_features[:, -1, 1]
            
            # Convert to hour (0-23)
            hour = torch.atan2(hour_sin, hour_cos) * 12 / np.pi + 12
            hour = torch.clamp(hour, 0, 23)
            
            # US trading hours (9:30-16:00 EST = 14:30-21:00 UTC)
            us_trading = ((hour >= 14.5) & (hour <= 21.0)).float()
            
            # Asian trading hours (0:00-8:00 UTC)
            asian_trading = ((hour >= 0.0) & (hour <= 8.0)).float()
            
            # Volatility should be higher during US trading hours
            vol_time_constraint = torch.mean(
                F.relu(pred_vol * (1 - us_trading) - pred_vol * us_trading * 1.2)
            )
            
            time_constraint = vol_time_constraint
        
        # Uncertainty regularization
        uncertainty_reg = torch.mean(uncertainty**2)
        
        # Total loss
        total_loss = (
            self.volatility_weight * (vol_loss + vol_constraint) +
            self.skewness_weight * (skew_loss + skew_constraint) +
            self.kurtosis_weight * (kurt_loss + kurt_constraint) +
            self.uncertainty_weight * uncertainty_reg +
            time_constraint
        )
        
        loss_components = {
            'total_loss': total_loss,
            'volatility_loss': vol_loss,
            'skewness_loss': skew_loss,
            'kurtosis_loss': kurt_loss,
            'volatility_constraint': vol_constraint,
            'skewness_constraint': skew_constraint,
            'kurtosis_constraint': kurt_constraint,
            'time_constraint': time_constraint,
            'uncertainty_reg': uncertainty_reg
        }
        
        return total_loss, loss_components

def create_realistic_enhanced_model(config, input_size: int = None) -> RealisticEnhancedModel:
    """Create realistic enhanced model for cryptocurrency prediction."""
    # Use provided input_size or fall back to config
    actual_input_size = input_size if input_size is not None else config.INPUT_SIZE
    
    model = RealisticEnhancedModel(
        input_size=actual_input_size,
        hidden_size=config.HIDDEN_SIZE,
        num_layers=config.NUM_LAYERS,
        num_heads=8,
        dropout=config.DROPOUT
    )
    
    print(f"Realistic enhanced model created with input_size={actual_input_size} and {sum(p.numel() for p in model.parameters()):,} parameters")
    return model 
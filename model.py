import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

class VolatilityLSTM(nn.Module):
    """
    LSTM-based neural network for predicting Bitcoin volatility, skewness, and kurtosis.
    """
    
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, 
                 output_size: int, dropout: float = 0.2):
        super(VolatilityLSTM, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
            bidirectional=False
        )
        
        # Attention mechanism
        self.attention = AttentionLayer(hidden_size)
        
        # Output layers
        self.dropout = nn.Dropout(dropout)
        
        # Separate heads for each target
        self.volatility_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1)
        )
        
        self.skewness_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1)
        )
        
        self.kurtosis_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1)
        )
        
        # Initialize weights
        self.init_weights()
    
    def init_weights(self):
        """Initialize model weights."""
        for name, param in self.lstm.named_parameters():
            if 'weight_ih' in name:
                torch.nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                torch.nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)
                # Set forget gate bias to 1
                n = param.size(0)
                param.data[(n//4):(n//2)].fill_(1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, sequence_length, input_size)
            
        Returns:
            Output tensor of shape (batch_size, output_size)
        """
        batch_size = x.size(0)
        
        # Initialize hidden state
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        
        # LSTM forward pass
        lstm_out, (hn, cn) = self.lstm(x, (h0, c0))
        
        # Apply attention
        attended_output = self.attention(lstm_out)
        
        # Apply dropout
        attended_output = self.dropout(attended_output)
        
        # Generate predictions for each target
        volatility = self.volatility_head(attended_output)
        skewness = self.skewness_head(attended_output)
        kurtosis = self.kurtosis_head(attended_output)
        
        # Concatenate outputs
        output = torch.cat([volatility, skewness, kurtosis], dim=1)
        
        # Apply activation functions
        output[:, 0] = F.softplus(output[:, 0])  # Volatility (always positive)
        # Skewness and kurtosis can be any real number
        
        return output


class AttentionLayer(nn.Module):
    """
    Attention mechanism for LSTM outputs.
    """
    
    def __init__(self, hidden_size: int):
        super(AttentionLayer, self).__init__()
        self.hidden_size = hidden_size
        self.attention_weights = nn.Linear(hidden_size, 1, bias=False)
        
    def forward(self, lstm_output: torch.Tensor) -> torch.Tensor:
        """
        Apply attention to LSTM outputs.
        
        Args:
            lstm_output: LSTM output of shape (batch_size, sequence_length, hidden_size)
            
        Returns:
            Attended output of shape (batch_size, hidden_size)
        """
        # Calculate attention scores
        attention_scores = self.attention_weights(lstm_output)  # (batch_size, seq_len, 1)
        attention_weights = F.softmax(attention_scores, dim=1)  # (batch_size, seq_len, 1)
        
        # Apply attention weights
        attended_output = torch.sum(lstm_output * attention_weights, dim=1)  # (batch_size, hidden_size)
        
        return attended_output


class EnsembleModel(nn.Module):
    """
    Ensemble of multiple LSTM models for improved predictions.
    """
    
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, 
                 output_size: int, num_models: int = 3, dropout: float = 0.2):
        super(EnsembleModel, self).__init__()
        
        self.num_models = num_models
        self.models = nn.ModuleList([
            VolatilityLSTM(input_size, hidden_size, num_layers, output_size, dropout)
            for _ in range(num_models)
        ])
        
        # Combine predictions
        self.combination_layer = nn.Linear(output_size * num_models, output_size)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through ensemble.
        """
        predictions = []
        for model in self.models:
            pred = model(x)
            predictions.append(pred)
        
        # Concatenate predictions
        combined = torch.cat(predictions, dim=1)
        
        # Final combination
        output = self.combination_layer(combined)
        
        # Apply activation functions
        output[:, 0] = F.softplus(output[:, 0])  # Volatility (always positive)
        
        return output


class VolatilityLoss(nn.Module):
    """
    Custom loss function for volatility prediction that considers the different scales
    and importance of volatility, skewness, and kurtosis.
    """
    
    def __init__(self, volatility_weight: float = 2.0, skewness_weight: float = 1.0, 
                 kurtosis_weight: float = 1.0):
        super(VolatilityLoss, self).__init__()
        self.volatility_weight = volatility_weight
        self.skewness_weight = skewness_weight
        self.kurtosis_weight = kurtosis_weight
        
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Calculate weighted loss for each target.
        """
        # Split predictions and targets
        pred_vol, pred_skew, pred_kurt = predictions[:, 0], predictions[:, 1], predictions[:, 2]
        target_vol, target_skew, target_kurt = targets[:, 0], targets[:, 1], targets[:, 2]
        
        # Calculate individual losses
        vol_loss = F.mse_loss(pred_vol, target_vol)
        skew_loss = F.mse_loss(pred_skew, target_skew)
        kurt_loss = F.mse_loss(pred_kurt, target_kurt)
        
        # Weighted combination
        total_loss = (self.volatility_weight * vol_loss + 
                     self.skewness_weight * skew_loss + 
                     self.kurtosis_weight * kurt_loss)
        
        return total_loss, vol_loss, skew_loss, kurt_loss


def count_parameters(model: nn.Module) -> int:
    """Count the number of trainable parameters in the model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def init_model(config) -> VolatilityLSTM:
    """Initialize model with configuration."""
    model = VolatilityLSTM(
        input_size=config.INPUT_SIZE,
        hidden_size=config.HIDDEN_SIZE,
        num_layers=config.NUM_LAYERS,
        output_size=config.OUTPUT_SIZE,
        dropout=config.DROPOUT
    )
    
    print(f"Model initialized with {count_parameters(model):,} parameters")
    return model 
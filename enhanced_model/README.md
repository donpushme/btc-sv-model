# Enhanced Model for Monte Carlo Simulation

This directory contains an enhanced model architecture specifically designed for Monte Carlo simulation with better statistical moment prediction.

## 🚀 Features

- **Hybrid LSTM-Transformer Architecture**: Combines LSTM for temporal dependencies with Transformer blocks for long-range relationships
- **Quantile Regression**: Better tail prediction for Monte Carlo simulation
- **Uncertainty Quantification**: Provides confidence intervals for predictions
- **Multi-scale Feature Processing**: Convolutional layers for different temporal scales
- **Enhanced Loss Function**: Optimized for statistical moment prediction
- **Advanced Feature Engineering**: Market microstructure and interaction features

## 📁 Directory Structure

```
enhanced_model/
├── enhanced_model.py          # Enhanced model architecture
├── config.py                  # Enhanced configuration
├── data_processor.py          # Enhanced data preprocessing
├── feature_engineering.py     # Advanced feature engineering
├── trainer.py                 # Enhanced training pipeline
├── predictor.py               # Enhanced prediction capabilities
├── train_all_cryptos.py       # Train all cryptocurrencies
├── models/                    # Enhanced models (saved in root models/)
├── results/                   # Training results and plots
└── README.md                  # This file
```

## 🎯 Key Improvements Over Standard Model

### 1. **Model Architecture**
- **Standard**: Single LSTM + Attention
- **Enhanced**: LSTM + Transformer + Multi-scale Convolutions

### 2. **Loss Function**
- **Standard**: Weighted MSE + Huber
- **Enhanced**: Quantile regression + Uncertainty regularization

### 3. **Feature Engineering**
- **Standard**: 80+ technical indicators
- **Enhanced**: + Market microstructure + Interaction features

### 4. **Prediction Output**
- **Standard**: Point predictions only
- **Enhanced**: Point predictions + Quantiles + Uncertainty

## 🚀 Quick Start

### 1. **Train Enhanced Models**

```bash
# Train all cryptocurrencies
cd enhanced_model
python train_all_cryptos.py

# Train specific cryptocurrency
python trainer.py BTC
python trainer.py ETH
python trainer.py XAU
python trainer.py SOL
```

### 2. **Make Predictions**

```bash
# Test prediction
python predictor.py BTC
```

### 3. **Model Files**

Enhanced models are saved in the root `models/` directory with the following naming convention:
- `{CRYPTO}_enhanced_model.pth` - Model weights
- `{CRYPTO}_enhanced_feature_engineer.pkl` - Feature engineering pipeline
- `{CRYPTO}_enhanced_metadata.json` - Model metadata

## 📊 Model Output

The enhanced model provides:

### **Point Predictions**
- `predicted_volatility`: Volatility (0.001-0.1 scale)
- `predicted_skewness`: Skewness (-2.0 to +2.0 scale)
- `predicted_kurtosis`: Excess kurtosis (-1.0 to +10.0 scale)

### **Uncertainty Quantification**
- `uncertainty_volatility`: Confidence in volatility prediction
- `uncertainty_skewness`: Confidence in skewness prediction
- `uncertainty_kurtosis`: Confidence in kurtosis prediction

### **Risk Assessment**
- `risk_level`: LOW/MEDIUM/HIGH based on predicted moments

## 🔧 Configuration

The enhanced model uses `EnhancedConfig` with optimized parameters:

```python
# Enhanced model parameters
HIDDEN_SIZE = 128
NUM_LAYERS = 3
NUM_HEADS = 8  # Transformer attention heads
NUM_QUANTILES = 5  # Quantile regression

# Enhanced loss weights
VOLATILITY_WEIGHT = 2.0
SKEWNESS_WEIGHT = 1.0
KURTOSIS_WEIGHT = 1.5
QUANTILE_WEIGHT = 0.5
UNCERTAINTY_WEIGHT = 0.1
```

## 📈 Training Process

1. **Data Preprocessing**: Enhanced data processor with better NaN handling
2. **Feature Engineering**: Advanced technical indicators + microstructure features
3. **Model Training**: Hybrid architecture with quantile regression
4. **Validation**: Early stopping with patience
5. **Model Saving**: Complete pipeline with feature engineer

## 🎲 Monte Carlo Simulation

The enhanced model is specifically designed for Monte Carlo simulation:

- **Better Tail Prediction**: Quantile regression captures extreme events
- **Uncertainty Propagation**: Confidence intervals for risk assessment
- **Realistic Bounds**: Proper scaling for financial data
- **Time-varying Parameters**: Captures intraday patterns

## 📋 Requirements

- PyTorch >= 1.9.0
- pandas >= 1.3.0
- numpy >= 1.21.0
- scikit-learn >= 1.0.0
- matplotlib >= 3.4.0
- seaborn >= 0.11.0

## 🔄 Integration

The enhanced model can be used alongside the standard model:

- **Standard Model**: For basic volatility prediction
- **Enhanced Model**: For advanced Monte Carlo simulation

Both models use the same training data from the root `training_data/` directory.

## 📊 Performance Comparison

| Metric | Standard Model | Enhanced Model |
|--------|---------------|----------------|
| Architecture | LSTM + Attention | LSTM + Transformer + Conv |
| Parameters | ~500K | ~800K |
| Training Time | 1x | 1.5x |
| Prediction Quality | Good | Excellent |
| Uncertainty | No | Yes |
| Monte Carlo Ready | Basic | Advanced |

## 🎯 Use Cases

### **Enhanced Model Recommended For:**
- High-frequency trading strategies
- Options pricing with precise tail modeling
- Portfolio optimization with uncertainty
- Regulatory compliance requiring confidence intervals
- Advanced risk management systems

### **Standard Model Sufficient For:**
- Basic volatility prediction
- Simple risk assessment
- Educational purposes
- Prototyping

## 🚨 Important Notes

1. **Training Data**: Uses the same training data as the standard model
2. **Model Storage**: Enhanced models are saved in the root `models/` directory
3. **Compatibility**: Enhanced models are not compatible with standard predictors
4. **Performance**: Enhanced models require more computational resources
5. **Retraining**: Enhanced models should be retrained from scratch (not compatible with standard models)

## 📞 Support

For issues or questions about the enhanced model:
1. Check the training logs in `results/` directory
2. Verify training data exists in root `training_data/` directory
3. Ensure sufficient computational resources for enhanced training 
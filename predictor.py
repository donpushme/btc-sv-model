import torch
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
import os
from datetime import datetime, timedelta
import pickle

from config import Config
from model import VolatilityLSTM
from feature_engineering import FeatureEngineer
from data_processor import CryptoDataProcessor

class RealTimeVolatilityPredictor:
    """
    Real-time predictor for cryptocurrency volatility, skewness, and kurtosis.
    Supports multiple cryptocurrencies: BTC, ETH, XAU, SOL
    """
    
    def __init__(self, crypto_symbol: str = 'BTC', model_path: str = None):
        """
        Initialize predictor with trained model for a specific cryptocurrency.
        
        Args:
            crypto_symbol: Cryptocurrency symbol (BTC, ETH, XAU, SOL)
            model_path: Path to saved model checkpoint
        """
        # Validate crypto symbol
        if crypto_symbol not in Config.SUPPORTED_CRYPTOS:
            raise ValueError(f"Unsupported crypto symbol: {crypto_symbol}. Supported: {list(Config.SUPPORTED_CRYPTOS.keys())}")
        
        self.crypto_symbol = crypto_symbol
        self.crypto_config = Config.SUPPORTED_CRYPTOS[crypto_symbol]
        self.config = Config()
        self.device = self.config.DEVICE
        self.model = None
        self.feature_engineer = FeatureEngineer()
        self.feature_cols = None
        self.target_cols = None
        self.processor = CryptoDataProcessor('', self.crypto_symbol)
        
        if model_path:
            self.load_model(model_path)
        else:
            # Try to load default model for this crypto
            default_path = os.path.join(self.config.MODEL_SAVE_PATH, f'{self.crypto_symbol}_best_model.pth')
            if os.path.exists(default_path):
                self.load_model(default_path)
            else:
                print(f"No model found for {self.crypto_config['name']} ({crypto_symbol}). Please provide model_path or train a model first.")
    
    def load_model(self, model_path: str):
        """
        Load trained model and preprocessing components.
        """
        try:
            # Use safe loading utility to handle PyTorch 2.6+ weights_only changes
            from utils import safe_torch_load
            checkpoint = safe_torch_load(model_path, map_location=self.device)
            
            # Load model configuration
            model_config = checkpoint['config']
            
            # Initialize model
            self.model = VolatilityLSTM(
                input_size=model_config['INPUT_SIZE'],
                hidden_size=model_config['HIDDEN_SIZE'],
                num_layers=model_config['NUM_LAYERS'],
                output_size=model_config['OUTPUT_SIZE'],
                dropout=model_config['DROPOUT']
            ).to(self.device)
            
            # Load model weights
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.eval()
            
            # Load preprocessing components
            self.feature_cols = checkpoint['feature_cols']
            self.target_cols = checkpoint['target_cols']
            self.feature_engineer.scalers['features'] = checkpoint['feature_scaler']
            self.feature_engineer.scalers['targets'] = checkpoint['target_scaler']
            self.feature_engineer.feature_names = self.feature_cols
            
            # Update config
            self.config.INPUT_SIZE = model_config['INPUT_SIZE']
            self.config.SEQUENCE_LENGTH = model_config['SEQUENCE_LENGTH']
            
            print(f"Model loaded successfully from {model_path}")
            print(f"Model trained on {len(self.feature_cols)} features for {self.crypto_config['name']} ({self.crypto_symbol})")
            print(f"Validation metrics from training:")
            val_metrics = checkpoint.get('val_metrics', {})
            for key, value in val_metrics.items():
                print(f"  {key}: {value:.4f}")
                
        except Exception as e:
            raise Exception(f"Failed to load model: {str(e)}")
    
    def load_latest_model(self) -> None:
        """
        Load the most recent trained model automatically for this cryptocurrency.
        """
        if not os.path.exists(self.config.MODEL_SAVE_PATH):
            raise ValueError(f"Model directory not found: {self.config.MODEL_SAVE_PATH}")
        
        # Find all model files for this crypto
        model_files = [f for f in os.listdir(self.config.MODEL_SAVE_PATH) 
                      if f.startswith(f'{self.crypto_symbol}_model_') and f.endswith('.pth')]
        
        if not model_files:
            raise ValueError(f"No trained models found for {self.crypto_config['name']} ({self.crypto_symbol}). Please train a model first.")
        
        # Sort by timestamp (newest first)
        model_files.sort(reverse=True)
        latest_model_file = model_files[0]
        
        # Extract model version from filename
        model_version = latest_model_file.replace(f'{self.crypto_symbol}_model_', '').replace('.pth', '')
        
        # Load the latest model
        model_path = os.path.join(self.config.MODEL_SAVE_PATH, latest_model_file)
        self.load_model(model_path)
        
        print(f"✅ Loaded latest model for {self.crypto_config['name']} ({self.crypto_symbol}): {model_version}")
    
    def preprocess_input_data(self, price_data: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess raw price data for prediction.
        
        Args:
            price_data: DataFrame with columns [timestamp, open, close, high, low]
            
        Returns:
            Preprocessed DataFrame ready for prediction
        """
        if len(price_data) < self.config.SEQUENCE_LENGTH + max(self.config.RETURN_WINDOWS):
            raise ValueError(f"Need at least {self.config.SEQUENCE_LENGTH + max(self.config.RETURN_WINDOWS)} "
                           f"data points for prediction, got {len(price_data)}")
        
        # Set the processor data and ensure timestamp is datetime
        df = price_data.copy()
        if 'timestamp' in df.columns:
            try:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
            except Exception as e:
                raise ValueError(f"Cannot convert timestamp column to datetime: {str(e)}")
        else:
            raise ValueError("Required column 'timestamp' not found in input data")
        
        self.processor.data = df
        
        # Calculate returns and basic features
        df = self.processor.calculate_returns()
        df = self.processor.add_time_features(df)
        df = self.processor.calculate_rolling_statistics(df, self.config.RETURN_WINDOWS)
        
        # Add engineered features
        df = self.feature_engineer.engineer_features(df)
        
        # Handle volume feature naming mismatch
        # During training, the model expects 'volume' but data_processor creates 'volume_proxy'
        if 'volume_proxy' in df.columns and 'volume' not in df.columns:
            df['volume'] = df['volume_proxy']
        
        # Remove NaN values
        df = df.dropna().reset_index(drop=True)
        
        return df
    
    def predict_next_period(self, price_data: pd.DataFrame, 
                           current_price: Optional[float] = None) -> Dict[str, float]:
        """
        Predict volatility, skewness, and kurtosis for the next 24-hour period.
        
        Args:
            price_data: Historical price data DataFrame
            current_price: Current Bitcoin price (optional, will use last close if not provided)
            
        Returns:
            Dictionary with predictions
        """
        if self.model is None:
            raise ValueError("Model not loaded. Please load a model first.")
        
        # Preprocess the data
        df = self.preprocess_input_data(price_data)
        
        # Ensure we have the required features
        missing_features = [col for col in self.feature_cols if col not in df.columns]
        if missing_features:
            raise ValueError(f"Missing required features: {missing_features}")
        
        # Scale the features
        df_scaled = df.copy()
        df_scaled[self.feature_cols] = self.feature_engineer.scalers['features'].transform(
            df[self.feature_cols]
        )
        
        # Prepare input sequence
        input_sequence = df_scaled[self.feature_cols].iloc[-self.config.SEQUENCE_LENGTH:].values
        input_tensor = torch.FloatTensor(input_sequence).unsqueeze(0).to(self.device)
        
        # Make prediction
        with torch.no_grad():
            prediction = self.model(input_tensor)
            prediction_np = prediction.cpu().numpy()
        
        # Inverse transform to original scale
        prediction_original = self.feature_engineer.inverse_transform_targets(prediction_np)
        
        # Extract individual predictions
        volatility = float(prediction_original[0, 0])
        skewness = float(prediction_original[0, 1])
        kurtosis = float(prediction_original[0, 2])
        
        # Apply validation bounds to prevent extreme predictions
        # Volatility should be positive and reasonable (0.001 to 0.1 = 0.1% to 10%)
        volatility = max(min(volatility, 0.1), 0.001)
        
        # Skewness can be negative or positive but should be reasonable (-2 to +2)
        skewness = max(min(skewness, 2.0), -2.0)
        
        # Kurtosis (excess kurtosis) should be reasonable (-1 to +27, corresponding to absolute kurtosis 2-30)
        kurtosis = max(min(kurtosis, 27.0), -1.0)
        
        # Current price for context
        if current_price is None:
            current_price = float(price_data['close'].iloc[-1])
        
        # Calculate additional metrics
        current_time = pd.to_datetime(price_data['timestamp'].iloc[-1])
        
        result = {
            'timestamp': current_time.isoformat(),
            'current_price': current_price,
            'predicted_volatility': volatility,
            'predicted_skewness': skewness,
            'predicted_kurtosis': kurtosis,
            'volatility_annualized': volatility * np.sqrt(365 * 24 * 12),  # Annualized volatility
            'prediction_period': '24_hours',
            'confidence_interval_lower': current_price * (1 - 2 * volatility),
            'confidence_interval_upper': current_price * (1 + 2 * volatility),
            'market_regime': self._classify_market_regime(volatility, skewness, kurtosis),
            'risk_assessment': self._assess_risk_level(volatility, skewness, kurtosis)
        }
        
        return result
    
    def predict_intraday_pattern(self, price_data: pd.DataFrame, 
                                intervals: int = 288) -> pd.DataFrame:
        """
        Predict volatility pattern for the next 24 hours at 5-minute intervals.
        
        Args:
            price_data: Historical price data
            intervals: Number of 5-minute intervals to predict (288 = 24 hours)
            
        Returns:
            DataFrame with time-series predictions
        """
        if self.model is None:
            raise ValueError("Model not loaded. Please load a model first.")
        
        base_prediction = self.predict_next_period(price_data)
        
        # Create time series for next 24 hours
        start_time = pd.to_datetime(price_data['timestamp'].iloc[-1])
        time_points = [start_time + timedelta(minutes=5*i) for i in range(1, intervals + 1)]
        
        # Generate intraday volatility pattern based on US trading hours
        intraday_pattern = []
        for t in time_points:
            hour_utc = t.hour
            
            # US trading hours effect (14:30 - 21:00 UTC = 9:30 AM - 4:00 PM EST)
            if 14 <= hour_utc <= 21:
                volatility_multiplier = 1.3  # Higher volatility during US hours
            elif 22 <= hour_utc <= 2:  # Late US/early Asian
                volatility_multiplier = 1.1
            elif 3 <= hour_utc <= 9:  # Asian trading hours
                volatility_multiplier = 0.9
            else:  # Low activity hours
                volatility_multiplier = 0.7
            
            # Weekend effect
            if t.weekday() >= 5:  # Saturday, Sunday
                volatility_multiplier *= 0.6
            
            # Add some noise for realism
            noise = np.random.normal(1.0, 0.1)
            volatility_multiplier *= noise
            
            intraday_pattern.append(volatility_multiplier)
        
        # Create predictions DataFrame
        predictions_df = pd.DataFrame({
            'timestamp': time_points,
            'predicted_volatility': [base_prediction['predicted_volatility'] * mult 
                                   for mult in intraday_pattern],
            'predicted_skewness': [base_prediction['predicted_skewness']] * len(time_points),
            'predicted_kurtosis': [base_prediction['predicted_kurtosis']] * len(time_points),
            'volatility_multiplier': intraday_pattern
        })
        
        return predictions_df
    
    def predict_for_timestamp(self, price_data: pd.DataFrame, target_timestamp: str,
                             current_price: Optional[float] = None) -> Dict[str, float]:
        """
        Predict volatility, skewness, and kurtosis for a specific timestamp.
        
        Args:
            price_data: Historical price data DataFrame
            target_timestamp: Target timestamp to predict for (format: 'YYYY-MM-DD HH:MM:SS')
            current_price: Current Bitcoin price (optional, will use close price at target time if not provided)
            
        Returns:
            Dictionary with predictions
        """
        if self.model is None:
            raise ValueError("Model not loaded. Please load a model first.")
        
        # Convert timestamp to datetime
        target_dt = pd.to_datetime(target_timestamp)
        
        # Find the index of the target timestamp in the data
        price_data['timestamp'] = pd.to_datetime(price_data['timestamp'])
        target_idx = price_data[price_data['timestamp'] == target_dt].index
        
        if len(target_idx) == 0:
            raise ValueError(f"Target timestamp {target_timestamp} not found in data")
        
        target_idx = target_idx[0]
        
        # Ensure we have enough historical data for the sequence
        if target_idx < self.config.SEQUENCE_LENGTH:
            raise ValueError(f"Not enough historical data. Need at least {self.config.SEQUENCE_LENGTH} periods before {target_timestamp}")
        
        # Get the data up to the target timestamp (exclusive)
        historical_data = price_data.iloc[:target_idx]
        
        # Preprocess the historical data
        df = self.preprocess_input_data(historical_data)
        
        # Ensure we have the required features
        missing_features = [col for col in self.feature_cols if col not in df.columns]
        if missing_features:
            raise ValueError(f"Missing required features: {missing_features}")
        
        # Scale the features
        df_scaled = df.copy()
        df_scaled[self.feature_cols] = self.feature_engineer.scalers['features'].transform(
            df[self.feature_cols]
        )
        
        # Prepare input sequence (last SEQUENCE_LENGTH periods before target)
        input_sequence = df_scaled[self.feature_cols].iloc[-self.config.SEQUENCE_LENGTH:].values
        input_tensor = torch.FloatTensor(input_sequence).unsqueeze(0).to(self.device)
        
        # Make prediction
        with torch.no_grad():
            prediction = self.model(input_tensor)
            prediction_np = prediction.cpu().numpy()
        
        # Inverse transform to original scale
        prediction_original = self.feature_engineer.inverse_transform_targets(prediction_np)
        
        # Extract individual predictions
        volatility = float(prediction_original[0, 0])
        skewness = float(prediction_original[0, 1])
        kurtosis = float(prediction_original[0, 2])
        
        # Apply validation bounds to prevent extreme predictions
        # Volatility should be positive and reasonable (0.001 to 0.1 = 0.1% to 10%)
        volatility = max(min(volatility, 0.1), 0.001)
        
        # Skewness can be negative or positive but should be reasonable (-2 to +2)
        skewness = max(min(skewness, 2.0), -2.0)
        
        # Kurtosis (excess kurtosis) should be reasonable (-1 to +27, corresponding to absolute kurtosis 2-30)
        kurtosis = max(min(kurtosis, 27.0), -1.0)
        
        # Get the actual price at target timestamp
        if current_price is None:
            current_price = float(price_data.iloc[target_idx]['close'])
        
        # Calculate annualized volatility
        volatility_annualized = volatility * np.sqrt(288 * 365)  # Annualized from 5-min data
        
        # Calculate confidence intervals
        confidence_interval_lower = current_price * (1 - 2 * volatility)
        confidence_interval_upper = current_price * (1 + 2 * volatility)
        
        # Determine market regime
        if volatility < 0.01:
            market_regime = 'low_volatility_stable'
        elif volatility < 0.025:
            market_regime = 'moderate_volatility'
        else:
            market_regime = 'high_volatility_stress'
        
        # Risk assessment
        if volatility > 0.05 or abs(skewness) > 1.5 or kurtosis > 15:
            risk_assessment = 'high'
        elif volatility > 0.025 or abs(skewness) > 0.8 or kurtosis > 8:
            risk_assessment = 'medium'
        else:
            risk_assessment = 'low'
        
        # Generate prediction ID
        prediction_id = f"{target_dt.strftime('%Y%m%d_%H%M%S')}_{int(pd.Timestamp.now().timestamp())}"
        
        return {
            'timestamp': target_timestamp,
            'current_price': current_price,
            'predicted_volatility': volatility,
            'predicted_skewness': skewness,
            'predicted_kurtosis': kurtosis,
            'volatility_annualized': volatility_annualized,
            'prediction_period': '24_hours',
            'confidence_interval_lower': confidence_interval_lower,
            'confidence_interval_upper': confidence_interval_upper,
            'market_regime': market_regime,
            'risk_assessment': risk_assessment,
            'model_version': self.model_version,
            'prediction_id': prediction_id,
            'source': 'Historical Analysis',
            'prediction_type': 'historical_timestamp'
        }
    
    def _classify_market_regime(self, volatility: float, skewness: float, 
                               kurtosis: float) -> str:
        """
        Classify market regime based on predicted statistics.
        """
        if volatility > 0.05:  # High volatility threshold
            if abs(skewness) > 0.5:
                return "high_volatility_skewed"
            else:
                return "high_volatility_normal"
        elif volatility < 0.02:  # Low volatility threshold
            return "low_volatility_stable"
        else:
            if kurtosis > 3:
                return "medium_volatility_fat_tails"
            else:
                return "medium_volatility_normal"
    
    def _assess_risk_level(self, volatility: float, skewness: float, 
                          kurtosis: float) -> str:
        """
        Assess risk level based on predicted statistics.
        """
        risk_score = 0
        
        # Volatility contribution
        if volatility > 0.06:
            risk_score += 3
        elif volatility > 0.03:
            risk_score += 2
        elif volatility > 0.015:
            risk_score += 1
        
        # Skewness contribution (negative skew is riskier)
        if skewness < -0.5:
            risk_score += 2
        elif abs(skewness) > 0.3:
            risk_score += 1
        
        # Kurtosis contribution (fat tails)
        if kurtosis > 5:
            risk_score += 2
        elif kurtosis > 3:
            risk_score += 1
        
        if risk_score >= 5:
            return "very_high"
        elif risk_score >= 3:
            return "high"
        elif risk_score >= 2:
            return "medium"
        else:
            return "low"
    
    def batch_predict(self, price_data_list: List[pd.DataFrame]) -> List[Dict]:
        """
        Make predictions for multiple datasets.
        """
        results = []
        for price_data in price_data_list:
            try:
                prediction = self.predict_next_period(price_data)
                results.append(prediction)
            except Exception as e:
                results.append({'error': str(e)})
        return results
    
    def get_model_info(self) -> Dict:
        """
        Get information about the loaded model.
        """
        if self.model is None:
            return {'error': 'No model loaded'}
        
        return {
            'crypto_symbol': self.crypto_symbol,
            'model_type': 'VolatilityLSTM',
            'input_size': self.config.INPUT_SIZE,
            'sequence_length': self.config.SEQUENCE_LENGTH,
            'num_features': len(self.feature_cols),
            'target_variables': self.target_cols,
            'device': str(self.device)
        }


def demo_prediction():
    """
    Demo function showing how to use the predictor.
    """
    # Initialize predictor
    predictor = RealTimeVolatilityPredictor()
    
    # Check if model is loaded
    if predictor.model is None:
        print("No trained model found. Please train a model first using trainer.py")
        return
    
    # Create sample data (in real use, this would be your actual Bitcoin price data)
    print("Creating sample data for demonstration...")
    
    # Generate sample Bitcoin price data
    dates = pd.date_range(start='2024-01-01', end='2024-01-02', freq='5min')
    n_points = len(dates)
    
    # Simulate realistic Bitcoin price movements
    np.random.seed(42)
    price_base = 45000
    returns = np.random.normal(0, 0.02, n_points)  # 2% daily volatility
    prices = [price_base]
    
    for ret in returns[1:]:
        prices.append(prices[-1] * (1 + ret))
    
    sample_data = pd.DataFrame({
        'timestamp': dates,
        'open': prices,
        'close': prices,
        'high': [p * (1 + abs(np.random.normal(0, 0.01))) for p in prices],
        'low': [p * (1 - abs(np.random.normal(0, 0.01))) for p in prices]
    })
    
    try:
        # Make prediction
        print("Making prediction...")
        prediction = predictor.predict_next_period(sample_data)
        
        print("\nPrediction Results:")
        print(f"Current Price: ${prediction['current_price']:,.2f}")
        print(f"Predicted Volatility: {prediction['predicted_volatility']:.4f}")
        print(f"Predicted Skewness: {prediction['predicted_skewness']:.4f}")
        print(f"Predicted Kurtosis: {prediction['predicted_kurtosis']:.4f}")
        print(f"Annualized Volatility: {prediction['volatility_annualized']:.2%}")
        print(f"Market Regime: {prediction['market_regime']}")
        print(f"Risk Level: {prediction['risk_assessment']}")
        
        # Intraday pattern prediction
        print("\nGenerating intraday pattern...")
        pattern = predictor.predict_intraday_pattern(sample_data, intervals=48)  # Next 4 hours
        print(f"Generated {len(pattern)} intraday predictions")
        
    except Exception as e:
        print(f"Prediction failed: {str(e)}")


if __name__ == "__main__":
    demo_prediction() 
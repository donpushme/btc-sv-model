#!/usr/bin/env python3
"""
Kurtosis Smoothing Utilities for Enhanced Model

This module provides utilities to smooth kurtosis predictions and prevent
zigzag patterns that are unrealistic for Monte Carlo simulation.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
from scipy import signal
from scipy.ndimage import gaussian_filter1d
import warnings
warnings.filterwarnings('ignore')

class KurtosisSmoother:
    """
    Utility class for smoothing kurtosis predictions to prevent zigzag patterns.
    """
    
    def __init__(self, method: str = 'gaussian', window_size: int = 5, sigma: float = 1.0):
        """
        Initialize kurtosis smoother.
        
        Args:
            method: Smoothing method ('gaussian', 'savgol', 'ewm', 'median')
            window_size: Window size for smoothing
            sigma: Standard deviation for Gaussian smoothing
        """
        self.method = method.lower()
        self.window_size = window_size
        self.sigma = sigma
        
        if self.method not in ['gaussian', 'savgol', 'ewm', 'median', 'kalman']:
            raise ValueError("Method must be 'gaussian', 'savgol', 'ewm', 'median', or 'kalman'")
    
    def smooth_kurtosis(self, kurtosis_values: np.ndarray) -> np.ndarray:
        """
        Smooth kurtosis values to prevent zigzag patterns.
        
        Args:
            kurtosis_values: Array of kurtosis predictions
            
        Returns:
            Smoothed kurtosis values
        """
        if len(kurtosis_values) < 3:
            return kurtosis_values
        
        if self.method == 'gaussian':
            return self._gaussian_smooth(kurtosis_values)
        elif self.method == 'savgol':
            return self._savgol_smooth(kurtosis_values)
        elif self.method == 'ewm':
            return self._ewm_smooth(kurtosis_values)
        elif self.method == 'median':
            return self._median_smooth(kurtosis_values)
        elif self.method == 'kalman':
            return self._kalman_smooth(kurtosis_values)
    
    def _gaussian_smooth(self, values: np.ndarray) -> np.ndarray:
        """Apply Gaussian smoothing."""
        return gaussian_filter1d(values, sigma=self.sigma)
    
    def _savgol_smooth(self, values: np.ndarray) -> np.ndarray:
        """Apply Savitzky-Golay smoothing."""
        window = min(self.window_size, len(values) // 2 * 2 - 1)  # Must be odd
        if window < 3:
            window = 3
        return signal.savgol_filter(values, window, 2)
    
    def _ewm_smooth(self, values: np.ndarray) -> np.ndarray:
        """Apply exponential weighted moving average."""
        span = max(2, self.window_size)
        return pd.Series(values).ewm(span=span).mean().values
    
    def _median_smooth(self, values: np.ndarray) -> np.ndarray:
        """Apply median smoothing."""
        return signal.medfilt(values, kernel_size=self.window_size)
    
    def _kalman_smooth(self, values: np.ndarray) -> np.ndarray:
        """Apply Kalman filter smoothing."""
        # Simple Kalman filter implementation
        n = len(values)
        smoothed = np.zeros(n)
        
        # Initial state
        x = values[0]
        P = 1.0
        
        # Process noise and measurement noise
        Q = 0.01  # Process noise
        R = 0.1   # Measurement noise
        
        # Forward pass
        for i in range(n):
            # Predict
            x_pred = x
            P_pred = P + Q
            
            # Update
            K = P_pred / (P_pred + R)  # Kalman gain
            x = x_pred + K * (values[i] - x_pred)
            P = (1 - K) * P_pred
            
            smoothed[i] = x
        
        # Backward pass (Rauch-Tung-Striebel smoothing)
        for i in range(n-2, -1, -1):
            smoothed[i] = smoothed[i] + 0.5 * (smoothed[i+1] - smoothed[i])
        
        return smoothed

class PredictionPostProcessor:
    """
    Post-processor for enhanced model predictions to ensure realistic patterns.
    """
    
    def __init__(self, kurtosis_smoother: Optional[KurtosisSmoother] = None):
        """
        Initialize post-processor.
        
        Args:
            kurtosis_smoother: Optional kurtosis smoother instance
        """
        if kurtosis_smoother is None:
            self.kurtosis_smoother = KurtosisSmoother(method='gaussian', sigma=1.5)
        else:
            self.kurtosis_smoother = kurtosis_smoother
    
    def process_predictions(self, predictions: List[Dict]) -> List[Dict]:
        """
        Process predictions to ensure realistic patterns.
        
        Args:
            predictions: List of prediction dictionaries
            
        Returns:
            Processed predictions
        """
        if not predictions:
            return predictions
        
        # Extract kurtosis values
        kurtosis_values = np.array([p.get('predicted_kurtosis', 3.0) for p in predictions])
        
        # Smooth kurtosis
        smoothed_kurtosis = self.kurtosis_smoother.smooth_kurtosis(kurtosis_values)
        
        # Apply constraints
        smoothed_kurtosis = self._apply_kurtosis_constraints(smoothed_kurtosis)
        
        # Update predictions
        processed_predictions = []
        for i, pred in enumerate(predictions):
            processed_pred = pred.copy()
            processed_pred['predicted_kurtosis'] = float(smoothed_kurtosis[i])
            processed_pred['kurtosis_smoothed'] = True
            processed_predictions.append(processed_pred)
        
        return processed_predictions
    
    def _apply_kurtosis_constraints(self, kurtosis_values: np.ndarray) -> np.ndarray:
        """
        Apply realistic constraints to kurtosis values.
        
        Args:
            kurtosis_values: Array of kurtosis values
            
        Returns:
            Constrained kurtosis values
        """
        # Ensure minimum kurtosis (excess kurtosis should be > -1)
        kurtosis_values = np.maximum(kurtosis_values, 0.1)
        
        # Cap maximum kurtosis (excess kurtosis should be reasonable)
        kurtosis_values = np.minimum(kurtosis_values, 10.0)
        
        # Ensure smooth transitions (no sudden jumps)
        for i in range(1, len(kurtosis_values)):
            max_change = 0.5  # Maximum allowed change per step
            diff = kurtosis_values[i] - kurtosis_values[i-1]
            if abs(diff) > max_change:
                if diff > 0:
                    kurtosis_values[i] = kurtosis_values[i-1] + max_change
                else:
                    kurtosis_values[i] = kurtosis_values[i-1] - max_change
        
        return kurtosis_values
    
    def validate_predictions(self, predictions: List[Dict]) -> Dict[str, bool]:
        """
        Validate predictions for realistic patterns.
        
        Args:
            predictions: List of prediction dictionaries
            
        Returns:
            Validation results
        """
        if not predictions:
            return {'valid': False, 'reason': 'No predictions provided'}
        
        # Extract values
        volatilities = [p.get('predicted_volatility', 0.001) for p in predictions]
        skewnesses = [p.get('predicted_skewness', 0.0) for p in predictions]
        kurtoses = [p.get('predicted_kurtosis', 3.0) for p in predictions]
        
        # Check ranges
        vol_valid = all(0.0001 <= v <= 0.1 for v in volatilities)
        skew_valid = all(-2.0 <= s <= 2.0 for s in skewnesses)
        kurt_valid = all(0.1 <= k <= 10.0 for k in kurtoses)
        
        # Check for zigzag patterns in kurtosis
        kurt_changes = np.diff(kurtoses)
        zigzag_score = np.mean(np.abs(kurt_changes))
        zigzag_valid = zigzag_score < 1.0  # Threshold for acceptable zigzag
        
        # Check temporal consistency
        temporal_valid = self._check_temporal_consistency(predictions)
        
        return {
            'valid': vol_valid and skew_valid and kurt_valid and zigzag_valid and temporal_valid,
            'volatility_valid': vol_valid,
            'skewness_valid': skew_valid,
            'kurtosis_valid': kurt_valid,
            'zigzag_valid': zigzag_valid,
            'temporal_valid': temporal_valid,
            'zigzag_score': float(zigzag_score)
        }
    
    def _check_temporal_consistency(self, predictions: List[Dict]) -> bool:
        """
        Check temporal consistency of predictions.
        
        Args:
            predictions: List of prediction dictionaries
            
        Returns:
            True if temporally consistent
        """
        if len(predictions) < 2:
            return True
        
        # Check if timestamps are in order
        timestamps = []
        for pred in predictions:
            if 'timestamp' in pred:
                if isinstance(pred['timestamp'], str):
                    timestamps.append(pd.to_datetime(pred['timestamp']))
                else:
                    timestamps.append(pred['timestamp'])
        
        if len(timestamps) > 1:
            # Check if timestamps are monotonically increasing
            return all(timestamps[i] <= timestamps[i+1] for i in range(len(timestamps)-1))
        
        return True

def smooth_predictions_batch(predictions_list: List[List[Dict]], 
                           method: str = 'gaussian') -> List[List[Dict]]:
    """
    Smooth a batch of prediction sequences.
    
    Args:
        predictions_list: List of prediction sequences
        method: Smoothing method
        
    Returns:
        List of smoothed prediction sequences
    """
    smoother = KurtosisSmoother(method=method)
    post_processor = PredictionPostProcessor(smoother)
    
    smoothed_predictions = []
    for predictions in predictions_list:
        smoothed = post_processor.process_predictions(predictions)
        smoothed_predictions.append(smoothed)
    
    return smoothed_predictions

def analyze_prediction_quality(predictions: List[Dict]) -> Dict:
    """
    Analyze the quality of predictions.
    
    Args:
        predictions: List of prediction dictionaries
        
    Returns:
        Quality analysis results
    """
    if not predictions:
        return {'error': 'No predictions provided'}
    
    # Extract values
    volatilities = np.array([p.get('predicted_volatility', 0.001) for p in predictions])
    skewnesses = np.array([p.get('predicted_skewness', 0.0) for p in predictions])
    kurtoses = np.array([p.get('predicted_kurtosis', 3.0) for p in predictions])
    
    # Calculate statistics
    analysis = {
        'volatility': {
            'mean': float(np.mean(volatilities)),
            'std': float(np.std(volatilities)),
            'min': float(np.min(volatilities)),
            'max': float(np.max(volatilities)),
            'range': float(np.max(volatilities) - np.min(volatilities))
        },
        'skewness': {
            'mean': float(np.mean(skewnesses)),
            'std': float(np.std(skewnesses)),
            'min': float(np.min(skewnesses)),
            'max': float(np.max(skewnesses)),
            'range': float(np.max(skewnesses) - np.min(skewnesses))
        },
        'kurtosis': {
            'mean': float(np.mean(kurtoses)),
            'std': float(np.std(kurtoses)),
            'min': float(np.min(kurtoses)),
            'max': float(np.max(kurtoses)),
            'range': float(np.max(kurtoses) - np.min(kurtoses))
        }
    }
    
    # Calculate zigzag scores
    vol_changes = np.diff(volatilities)
    skew_changes = np.diff(skewnesses)
    kurt_changes = np.diff(kurtoses)
    
    analysis['zigzag_scores'] = {
        'volatility': float(np.mean(np.abs(vol_changes))),
        'skewness': float(np.mean(np.abs(skew_changes))),
        'kurtosis': float(np.mean(np.abs(kurt_changes)))
    }
    
    # Quality indicators
    analysis['quality_indicators'] = {
        'volatility_stable': analysis['zigzag_scores']['volatility'] < 0.001,
        'skewness_stable': analysis['zigzag_scores']['skewness'] < 0.1,
        'kurtosis_stable': analysis['zigzag_scores']['kurtosis'] < 0.5,
        'kurtosis_realistic': 0.1 <= analysis['kurtosis']['mean'] <= 10.0
    }
    
    return analysis

# Example usage
if __name__ == "__main__":
    # Create sample predictions with zigzag kurtosis
    sample_predictions = []
    for i in range(100):
        # Simulate zigzag kurtosis pattern
        kurtosis = 4.0 + 2.0 * np.sin(i * 0.5) + 0.5 * np.random.normal(0, 1)
        sample_predictions.append({
            'predicted_volatility': 0.001 + 0.0001 * np.random.normal(0, 1),
            'predicted_skewness': -0.1 + 0.05 * np.random.normal(0, 1),
            'predicted_kurtosis': max(0.1, kurtosis)
        })
    
    # Analyze original predictions
    print("Original predictions analysis:")
    analysis = analyze_prediction_quality(sample_predictions)
    print(f"Kurtosis zigzag score: {analysis['zigzag_scores']['kurtosis']:.3f}")
    
    # Smooth predictions
    smoother = KurtosisSmoother(method='gaussian', sigma=2.0)
    post_processor = PredictionPostProcessor(smoother)
    smoothed_predictions = post_processor.process_predictions(sample_predictions)
    
    # Analyze smoothed predictions
    print("\nSmoothed predictions analysis:")
    smoothed_analysis = analyze_prediction_quality(smoothed_predictions)
    print(f"Kurtosis zigzag score: {smoothed_analysis['zigzag_scores']['kurtosis']:.3f}")
    
    # Validate predictions
    validation = post_processor.validate_predictions(smoothed_predictions)
    print(f"\nValidation results: {validation}") 
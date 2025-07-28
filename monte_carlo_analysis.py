#!/usr/bin/env python3
"""
Monte Carlo Analysis and Best Practices

This script analyzes the prediction data to understand kurtosis scales and variation,
and provides best practices for Monte Carlo simulation with the model predictions.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

def analyze_prediction_data(predictions: List[Dict]) -> Dict:
    """
    Analyze prediction data to understand scales and variation.
    
    Args:
        predictions: List of 288 prediction dictionaries
        
    Returns:
        Dictionary with analysis results
    """
    print("🔍 Analyzing Prediction Data")
    print("=" * 50)
    
    if not predictions:
        return {'error': 'No predictions provided'}
    
    # Extract parameters
    volatilities = [p['predicted_volatility'] for p in predictions]
    skewnesses = [p['predicted_skewness'] for p in predictions]
    kurtoses = [p['predicted_kurtosis'] for p in predictions]
    
    # Basic statistics
    analysis = {
        'total_predictions': len(predictions),
        'volatility': {
            'min': min(volatilities),
            'max': max(volatilities),
            'mean': np.mean(volatilities),
            'std': np.std(volatilities),
            'range': max(volatilities) - min(volatilities),
            'unique_values': len(set(volatilities))
        },
        'skewness': {
            'min': min(skewnesses),
            'max': max(skewnesses),
            'mean': np.mean(skewnesses),
            'std': np.std(skewnesses),
            'range': max(skewnesses) - min(skewnesses),
            'unique_values': len(set(skewnesses))
        },
        'kurtosis': {
            'min': min(kurtoses),
            'max': max(kurtoses),
            'mean': np.mean(kurtoses),
            'std': np.std(kurtoses),
            'range': max(kurtoses) - min(kurtoses),
            'unique_values': len(set(kurtoses))
        }
    }
    
    # Print analysis
    print(f"📊 Total predictions: {analysis['total_predictions']}")
    
    print(f"\n📈 Volatility Analysis:")
    print(f"   Range: {analysis['volatility']['min']:.6f} to {analysis['volatility']['max']:.6f}")
    print(f"   Mean: {analysis['volatility']['mean']:.6f} ± {analysis['volatility']['std']:.6f}")
    print(f"   Unique values: {analysis['volatility']['unique_values']}")
    print(f"   Variation: {analysis['volatility']['range']:.6f}")
    
    print(f"\n📈 Skewness Analysis:")
    print(f"   Range: {analysis['skewness']['min']:.6f} to {analysis['skewness']['max']:.6f}")
    print(f"   Mean: {analysis['skewness']['mean']:.6f} ± {analysis['skewness']['std']:.6f}")
    print(f"   Unique values: {analysis['skewness']['unique_values']}")
    print(f"   Variation: {analysis['skewness']['range']:.6f}")
    
    print(f"\n📈 Kurtosis Analysis:")
    print(f"   Range: {analysis['kurtosis']['min']:.6f} to {analysis['kurtosis']['max']:.6f}")
    print(f"   Mean: {analysis['kurtosis']['mean']:.6f} ± {analysis['kurtosis']['std']:.6f}")
    print(f"   Unique values: {analysis['kurtosis']['unique_values']}")
    print(f"   Variation: {analysis['kurtosis']['range']:.6f}")
    
    # Check for issues
    issues = []
    
    if analysis['kurtosis']['unique_values'] == 1:
        issues.append("⚠️  CRITICAL: Kurtosis is constant across all 288 predictions")
        issues.append("   This indicates the model is not predicting time-varying kurtosis")
    
    if analysis['skewness']['unique_values'] == 1:
        issues.append("⚠️  CRITICAL: Skewness is constant across all 288 predictions")
        issues.append("   This indicates the model is not predicting time-varying skewness")
    
    if analysis['volatility']['unique_values'] < 10:
        issues.append("⚠️  WARNING: Very low volatility variation")
        issues.append("   Expected more variation based on time-of-day patterns")
    
    if analysis['kurtosis']['std'] < 0.1:
        issues.append("⚠️  WARNING: Very low kurtosis variation")
        issues.append("   Expected more variation in tail behavior")
    
    # Kurtosis scale analysis
    print(f"\n🎯 Kurtosis Scale Analysis:")
    print(f"   Current scale: Excess kurtosis (kurtosis - 3)")
    print(f"   Absolute kurtosis range: {analysis['kurtosis']['min'] + 3:.2f} to {analysis['kurtosis']['max'] + 3:.2f}")
    print(f"   Normal distribution: kurtosis = 3 (excess = 0)")
    print(f"   Fat tails: excess kurtosis > 0")
    print(f"   Thin tails: excess kurtosis < 0")
    
    if issues:
        print(f"\n🚨 Issues Found:")
        for issue in issues:
            print(f"   {issue}")
    
    return analysis

def create_enhanced_monte_carlo_simulator():
    """
    Create an enhanced Monte Carlo simulator with best practices.
    """
    
    class EnhancedMonteCarloSimulator:
        """
        Enhanced Monte Carlo simulator with best practices for cryptocurrency predictions.
        """
        
        def __init__(self, method: str = 'cornish_fisher', 
                     kurtosis_scale: str = 'excess',
                     time_varying_mode: str = 'adaptive'):
            """
            Initialize enhanced Monte Carlo simulator.
            
            Args:
                method: 'cornish_fisher' or 'normal'
                kurtosis_scale: 'excess' (kurtosis - 3) or 'absolute' (kurtosis)
                time_varying_mode: 'adaptive', 'constant', or 'realistic'
            """
            self.method = method.lower()
            self.kurtosis_scale = kurtosis_scale.lower()
            self.time_varying_mode = time_varying_mode.lower()
            
            if self.method not in ['cornish_fisher', 'normal']:
                raise ValueError("Method must be 'cornish_fisher' or 'normal'")
            
            if self.kurtosis_scale not in ['excess', 'absolute']:
                raise ValueError("Kurtosis scale must be 'excess' or 'absolute'")
            
            if self.time_varying_mode not in ['adaptive', 'constant', 'realistic']:
                raise ValueError("Time varying mode must be 'adaptive', 'constant', or 'realistic'")
        
        def analyze_predictions(self, predictions: List[Dict]) -> Dict:
            """
            Analyze predictions and provide recommendations.
            """
            analysis = analyze_prediction_data(predictions)
            
            if 'error' in analysis:
                return analysis
            
            # Recommendations based on analysis
            recommendations = []
            
            if analysis['kurtosis']['unique_values'] == 1:
                recommendations.append({
                    'issue': 'Constant kurtosis',
                    'problem': 'Model predicts same kurtosis for all time points',
                    'solution': 'Use constant parameter simulation or realistic variation',
                    'mode': 'constant'
                })
            
            if analysis['skewness']['unique_values'] == 1:
                recommendations.append({
                    'issue': 'Constant skewness', 
                    'problem': 'Model predicts same skewness for all time points',
                    'solution': 'Use constant parameter simulation or realistic variation',
                    'mode': 'constant'
                })
            
            if analysis['volatility']['unique_values'] < 10:
                recommendations.append({
                    'issue': 'Low volatility variation',
                    'problem': 'Very little time-of-day variation in volatility',
                    'solution': 'Use realistic time patterns or constant parameters',
                    'mode': 'realistic'
                })
            
            if not recommendations:
                recommendations.append({
                    'issue': 'Good variation',
                    'problem': 'None',
                    'solution': 'Use time-varying simulation as intended',
                    'mode': 'adaptive'
                })
            
            analysis['recommendations'] = recommendations
            return analysis
        
        def generate_realistic_variation(self, base_value: float, 
                                       variation_type: str = 'kurtosis') -> List[float]:
            """
            Generate realistic time-varying patterns when model predictions are constant.
            
            Args:
                base_value: Base value from model prediction
                variation_type: 'kurtosis', 'skewness', or 'volatility'
                
            Returns:
                List of 288 varied values
            """
            variations = []
            
            for i in range(288):
                hour = (i // 12) % 24
                day_of_week = (i // (24 * 12)) % 7
                
                if variation_type == 'volatility':
                    # Time-of-day patterns for volatility
                    if 14 <= hour <= 21:  # US trading hours
                        multiplier = 1.3
                    elif 22 <= hour <= 2:  # Late US/early Asian
                        multiplier = 1.1
                    elif 3 <= hour <= 9:  # Asian trading hours
                        multiplier = 0.9
                    else:  # Low activity hours
                        multiplier = 0.7
                    
                    # Weekend effect
                    if day_of_week >= 5:
                        multiplier *= 0.6
                    
                    # Add realistic noise
                    noise = np.random.normal(1.0, 0.05)
                    final_multiplier = multiplier * noise
                    
                    varied_value = base_value * final_multiplier
                    
                elif variation_type == 'kurtosis':
                    # Kurtosis tends to be higher during high volatility periods
                    if 14 <= hour <= 21:  # US trading hours
                        base_multiplier = 1.2
                    else:
                        base_multiplier = 0.9
                    
                    # Add realistic variation
                    noise = np.random.normal(0, 0.1)
                    varied_value = base_value + noise
                    
                    # Ensure reasonable bounds
                    if self.kurtosis_scale == 'excess':
                        varied_value = max(min(varied_value, 27.0), -1.0)
                    else:  # absolute
                        varied_value = max(min(varied_value, 30.0), 2.0)
                    
                elif variation_type == 'skewness':
                    # Skewness can vary based on market sentiment
                    # Add realistic variation
                    noise = np.random.normal(0, 0.05)
                    varied_value = base_value + noise
                    
                    # Ensure reasonable bounds
                    varied_value = max(min(varied_value, 2.0), -2.0)
                
                variations.append(varied_value)
            
            return variations
        
        def simulate_with_best_practices(self, predictions: List[Dict], 
                                       initial_price: float,
                                       num_simulations: int = 1000) -> Tuple[pd.DataFrame, Dict]:
            """
            Simulate using best practices based on prediction analysis.
            """
            # Analyze predictions first
            analysis = self.analyze_predictions(predictions)
            
            if 'error' in analysis:
                raise ValueError(f"Prediction analysis failed: {analysis['error']}")
            
            # Determine simulation mode based on recommendations
            if analysis['recommendations']:
                recommended_mode = analysis['recommendations'][0]['mode']
                print(f"🎯 Using recommended mode: {recommended_mode}")
                print(f"   Reason: {analysis['recommendations'][0]['issue']}")
                print(f"   Solution: {analysis['recommendations'][0]['solution']}")
            else:
                recommended_mode = 'adaptive'
            
            # Extract and potentially enhance parameters
            volatilities = [p['predicted_volatility'] for p in predictions]
            skewnesses = [p['predicted_skewness'] for p in predictions]
            kurtoses = [p['predicted_kurtosis'] for p in predictions]
            
            # Apply enhancements based on mode
            if recommended_mode == 'constant':
                # Use average values for constant simulation
                avg_volatility = np.mean(volatilities)
                avg_skewness = np.mean(skewnesses)
                avg_kurtosis = np.mean(kurtoses)
                
                print(f"📊 Using constant parameters:")
                print(f"   Volatility: {avg_volatility:.6f}")
                print(f"   Skewness: {avg_skewness:.6f}")
                print(f"   Kurtosis: {avg_kurtosis:.6f}")
                
                return self._simulate_constant(avg_volatility, avg_skewness, avg_kurtosis,
                                             initial_price, num_simulations)
            
            elif recommended_mode == 'realistic':
                # Generate realistic variation for constant predictions
                print(f"📊 Generating realistic variation for constant predictions...")
                
                enhanced_volatilities = self.generate_realistic_variation(
                    np.mean(volatilities), 'volatility')
                enhanced_skewnesses = self.generate_realistic_variation(
                    np.mean(skewnesses), 'skewness')
                enhanced_kurtoses = self.generate_realistic_variation(
                    np.mean(kurtoses), 'kurtosis')
                
                return self._simulate_time_varying(enhanced_volatilities, enhanced_skewnesses, 
                                                 enhanced_kurtoses, initial_price, num_simulations)
            
            else:  # adaptive
                # Use original predictions as-is
                print(f"📊 Using original time-varying predictions...")
                return self._simulate_time_varying(volatilities, skewnesses, kurtoses,
                                                 initial_price, num_simulations)
        
        def _simulate_constant(self, volatility: float, skewness: float, kurtosis: float,
                             initial_price: float, num_simulations: int) -> Tuple[pd.DataFrame, Dict]:
            """
            Simulate with constant parameters.
            """
            intervals = 288
            dt = 1/288
            
            # Generate all returns at once
            if self.method == 'cornish_fisher':
                returns = self._generate_returns_cornish_fisher(
                    size=(num_simulations, intervals),
                    volatility=volatility,
                    skewness=skewness,
                    kurtosis=kurtosis,
                    dt=dt
                )
            else:
                returns = np.random.normal(0, volatility * np.sqrt(dt), (num_simulations, intervals))
            
            # Convert to price paths
            price_paths = np.zeros((num_simulations, intervals + 1))
            price_paths[:, 0] = initial_price
            
            for i in range(intervals):
                price_paths[:, i + 1] = price_paths[:, i] * (1 + returns[:, i])
            
            # Create results
            simulation_results = pd.DataFrame(price_paths.T)
            simulation_results.index = range(len(simulation_results))
            
            # Calculate summary stats
            summary_stats = self._calculate_summary_stats(price_paths, initial_price)
            summary_stats['simulation_type'] = 'constant_parameters'
            summary_stats['method'] = self.method
            summary_stats['kurtosis_scale'] = self.kurtosis_scale
            
            return simulation_results, summary_stats
        
        def _simulate_time_varying(self, volatilities: List[float], skewnesses: List[float],
                                 kurtoses: List[float], initial_price: float,
                                 num_simulations: int) -> Tuple[pd.DataFrame, Dict]:
            """
            Simulate with time-varying parameters.
            """
            intervals = 288
            dt = 1/288
            
            # Generate price paths
            price_paths = np.zeros((num_simulations, intervals + 1))
            price_paths[:, 0] = initial_price
            
            for i in range(intervals):
                if self.method == 'cornish_fisher':
                    returns = self._generate_returns_cornish_fisher(
                        size=(num_simulations, 1),
                        volatility=volatilities[i],
                        skewness=skewnesses[i],
                        kurtosis=kurtoses[i],
                        dt=dt
                    ).flatten()
                else:
                    returns = np.random.normal(0, volatilities[i] * np.sqrt(dt), num_simulations)
                
                price_paths[:, i + 1] = price_paths[:, i] * (1 + returns)
            
            # Create results
            simulation_results = pd.DataFrame(price_paths.T)
            simulation_results.index = range(len(simulation_results))
            
            # Calculate summary stats
            summary_stats = self._calculate_summary_stats(price_paths, initial_price)
            summary_stats['simulation_type'] = 'time_varying'
            summary_stats['method'] = self.method
            summary_stats['kurtosis_scale'] = self.kurtosis_scale
            
            return simulation_results, summary_stats
        
        def _generate_returns_cornish_fisher(self, size: Tuple[int, int], 
                                          volatility: float, skewness: float, 
                                          kurtosis: float, dt: float) -> np.ndarray:
            """
            Generate returns using Cornish-Fisher expansion.
            """
            # Generate standard normal returns
            normal_returns = np.random.normal(0, 1, size)
            
            # Apply Cornish-Fisher transformation
            if self.kurtosis_scale == 'excess':
                # kurtosis is already excess kurtosis
                cf_returns = normal_returns + \
                            (skewness / 6) * (normal_returns**2 - 1) + \
                            (kurtosis) / 24 * (normal_returns**3 - 3 * normal_returns) - \
                            (skewness**2) / 36 * (2 * normal_returns**3 - 5 * normal_returns)
            else:
                # kurtosis is absolute kurtosis, convert to excess
                excess_kurtosis = kurtosis - 3
                cf_returns = normal_returns + \
                            (skewness / 6) * (normal_returns**2 - 1) + \
                            (excess_kurtosis) / 24 * (normal_returns**3 - 3 * normal_returns) - \
                            (skewness**2) / 36 * (2 * normal_returns**3 - 5 * normal_returns)
            
            # Scale by volatility and time step
            returns = cf_returns * volatility * np.sqrt(dt)
            
            return returns
        
        def _calculate_summary_stats(self, price_paths: np.ndarray, initial_price: float) -> Dict:
            """
            Calculate comprehensive summary statistics.
            """
            final_prices = price_paths[:, -1]
            returns_all = np.diff(np.log(price_paths), axis=1)
            
            summary_stats = {
                'mean_final_price': final_prices.mean(),
                'std_final_price': final_prices.std(),
                'min_final_price': final_prices.min(),
                'max_final_price': final_prices.max(),
                'percentile_5': np.percentile(final_prices, 5),
                'percentile_25': np.percentile(final_prices, 25),
                'percentile_50': np.percentile(final_prices, 50),
                'percentile_75': np.percentile(final_prices, 75),
                'percentile_95': np.percentile(final_prices, 95),
                'probability_profit': (final_prices > initial_price).mean(),
                'probability_loss': (final_prices < initial_price).mean(),
                'max_drawdown': self._calculate_max_drawdown(price_paths),
                'volatility_realized': np.std(returns_all) * np.sqrt(1/(1/288)),
                'skewness_realized': float(stats.skew(returns_all.flatten())),
                'kurtosis_realized': float(stats.kurtosis(returns_all.flatten())),
                'var_95': np.percentile(final_prices - initial_price, 5),
                'cvar_95': np.mean(final_prices[final_prices <= np.percentile(final_prices, 5)] - initial_price),
                'expected_return': (final_prices.mean() - initial_price) / initial_price,
                'sharpe_ratio': ((final_prices.mean() - initial_price) / initial_price) / (final_prices.std() / initial_price),
                'num_simulations': price_paths.shape[0],
                'intervals': price_paths.shape[1] - 1,
                'initial_price': initial_price
            }
            
            return summary_stats
        
        def _calculate_max_drawdown(self, price_paths: np.ndarray) -> float:
            """
            Calculate maximum drawdown across all simulation paths.
            """
            max_drawdowns = []
            
            for path in price_paths:
                running_max = np.maximum.accumulate(path)
                drawdown = (path - running_max) / running_max
                max_drawdowns.append(drawdown.min())
            
            return np.mean(max_drawdowns)
    
    return EnhancedMonteCarloSimulator

def best_practices_guide():
    """
    Print best practices guide for Monte Carlo simulation.
    """
    print("\n" + "="*60)
    print("🎯 MONTE CARLO SIMULATION BEST PRACTICES")
    print("="*60)
    
    print("\n📊 Understanding Kurtosis Scale:")
    print("   • Model predicts EXCESS kurtosis (kurtosis - 3)")
    print("   • Normal distribution: excess kurtosis = 0")
    print("   • Fat tails: excess kurtosis > 0")
    print("   • Thin tails: excess kurtosis < 0")
    print("   • Valid range: -1 to +27 (absolute kurtosis: 2 to 30)")
    
    print("\n🔍 Common Issues and Solutions:")
    print("   1. Constant Kurtosis Across 288 Points:")
    print("      → Problem: Model predicts same kurtosis for all time points")
    print("      → Solution: Use constant parameter simulation")
    print("      → Alternative: Generate realistic time-varying patterns")
    
    print("\n   2. Low Parameter Variation:")
    print("      → Problem: Very little time-of-day variation")
    print("      → Solution: Use realistic market patterns")
    print("      → Alternative: Use constant parameters")
    
    print("\n   3. Scale Confusion:")
    print("      → Problem: Uncertainty about kurtosis scale")
    print("      → Solution: Always use excess kurtosis for Cornish-Fisher")
    print("      → Note: Model outputs are already in excess kurtosis scale")
    
    print("\n🚀 Recommended Workflow:")
    print("   1. Analyze predictions with EnhancedMonteCarloSimulator")
    print("   2. Follow automatic recommendations")
    print("   3. Choose appropriate simulation mode:")
    print("      • 'constant': Use average parameters")
    print("      • 'realistic': Generate realistic variation")
    print("      • 'adaptive': Use original predictions")
    
    print("\n💡 Implementation Tips:")
    print("   • Always validate parameter bounds before simulation")
    print("   • Use Cornish-Fisher for non-normal distributions")
    print("   • Consider time-of-day patterns for realistic variation")
    print("   • Monitor realized vs predicted statistics")
    print("   • Use sufficient simulations (1000+) for stable results")

def example_usage():
    """
    Example usage of the enhanced Monte Carlo simulator.
    """
    print("\n🚀 Enhanced Monte Carlo Simulator Example")
    print("=" * 50)
    
    # Create enhanced simulator
    EnhancedSimulator = create_enhanced_monte_carlo_simulator()
    simulator = EnhancedSimulator(
        method='cornish_fisher',
        kurtosis_scale='excess',
        time_varying_mode='adaptive'
    )
    
    # Create sample predictions (simulating your database data)
    print("📊 Creating sample predictions...")
    predictions = []
    
    # Simulate the issue: constant kurtosis and skewness
    base_volatility = 0.025
    base_skewness = -0.1
    base_kurtosis = 4.0  # This will be constant across all 288 points
    
    for i in range(288):
        hour = (i // 12) % 24
        
        # Only volatility varies (as in your current system)
        if 14 <= hour <= 21:
            vol_multiplier = 1.3
        elif 22 <= hour <= 2:
            vol_multiplier = 1.1
        elif 3 <= hour <= 9:
            vol_multiplier = 0.9
        else:
            vol_multiplier = 0.7
        
        # Weekend effect
        day_of_week = (i // (24 * 12)) % 7
        if day_of_week >= 5:
            vol_multiplier *= 0.6
        
        noise = np.random.normal(1.0, 0.05)
        final_vol_multiplier = vol_multiplier * noise
        
        prediction = {
            'sequence_number': i + 1,
            'predicted_volatility': base_volatility * final_vol_multiplier,
            'predicted_skewness': base_skewness,  # Constant!
            'predicted_kurtosis': base_kurtosis,  # Constant!
            'current_price': 45000.0
        }
        predictions.append(prediction)
    
    print(f"✅ Generated {len(predictions)} sample predictions")
    
    # Analyze predictions
    print("\n🔍 Analyzing predictions...")
    analysis = simulator.analyze_predictions(predictions)
    
    # Run simulation with best practices
    print("\n🔮 Running simulation with best practices...")
    initial_price = 45000.0
    num_simulations = 1000
    
    simulation_results, summary_stats = simulator.simulate_with_best_practices(
        predictions=predictions,
        initial_price=initial_price,
        num_simulations=num_simulations
    )
    
    print(f"\n📊 Simulation Results:")
    print(f"   Mean final price: ${summary_stats['mean_final_price']:,.2f}")
    print(f"   Probability of profit: {summary_stats['probability_profit']:.1%}")
    print(f"   Expected return: {summary_stats['expected_return']:.2%}")
    print(f"   Sharpe ratio: {summary_stats['sharpe_ratio']:.3f}")
    print(f"   VaR (95%): ${summary_stats['var_95']:,.2f}")
    print(f"   Max drawdown: {summary_stats['max_drawdown']:.2%}")
    print(f"   Simulation type: {summary_stats['simulation_type']}")
    print(f"   Method: {summary_stats['method']}")
    print(f"   Kurtosis scale: {summary_stats['kurtosis_scale']}")
    
    print("\n✅ Example completed!")
    print("\n💡 Next Steps:")
    print("   1. Use this enhanced simulator with your real database predictions")
    print("   2. Follow the automatic recommendations")
    print("   3. Choose the appropriate simulation mode based on your data")

if __name__ == "__main__":
    best_practices_guide()
    example_usage() 
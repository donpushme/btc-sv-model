#!/usr/bin/env python3
"""
Enhanced Monte Carlo Simulator with Best Practices

This simulator handles the issues with constant kurtosis/skewness predictions
and provides intelligent simulation modes based on prediction analysis.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

class EnhancedMonteCarloSimulator:
    """
    Enhanced Monte Carlo simulator that handles constant predictions intelligently.
    """
    
    def __init__(self, method: str = 'cornish_fisher'):
        """
        Initialize enhanced simulator.
        
        Args:
            method: 'cornish_fisher' or 'normal'
        """
        self.method = method.lower()
        if self.method not in ['cornish_fisher', 'normal']:
            raise ValueError("Method must be 'cornish_fisher' or 'normal'")
    
    def analyze_predictions(self, predictions: List[Dict]) -> Dict:
        """
        Analyze predictions and determine best simulation approach.
        """
        if not predictions:
            return {'error': 'No predictions provided'}
        
        # Extract parameters
        volatilities = [p['predicted_volatility'] for p in predictions]
        skewnesses = [p['predicted_skewness'] for p in predictions]
        kurtoses = [p['predicted_kurtosis'] for p in predictions]
        
        # Check for constant values
        vol_unique = len(set(volatilities))
        skew_unique = len(set(skewnesses))
        kurt_unique = len(set(kurtoses))
        
        analysis = {
            'total_predictions': len(predictions),
            'volatility_variation': vol_unique,
            'skewness_variation': skew_unique,
            'kurtosis_variation': kurt_unique,
            'volatility_stats': {
                'min': min(volatilities), 'max': max(volatilities),
                'mean': np.mean(volatilities), 'std': np.std(volatilities)
            },
            'skewness_stats': {
                'min': min(skewnesses), 'max': max(skewnesses),
                'mean': np.mean(skewnesses), 'std': np.std(skewnesses)
            },
            'kurtosis_stats': {
                'min': min(kurtoses), 'max': max(kurtoses),
                'mean': np.mean(kurtoses), 'std': np.std(kurtoses)
            }
        }
        
        # Determine simulation mode
        if kurt_unique == 1 and skew_unique == 1:
            analysis['recommended_mode'] = 'constant_parameters'
            analysis['reason'] = 'Kurtosis and skewness are constant across all predictions'
        elif vol_unique < 10:
            analysis['recommended_mode'] = 'realistic_variation'
            analysis['reason'] = 'Low volatility variation, generating realistic patterns'
        else:
            analysis['recommended_mode'] = 'time_varying'
            analysis['reason'] = 'Good variation in all parameters'
        
        return analysis
    
    def generate_realistic_variation(self, base_value: float, 
                                   variation_type: str = 'kurtosis') -> List[float]:
        """
        Generate realistic time-varying patterns for constant predictions.
        """
        variations = []
        
        for i in range(288):
            hour = (i // 12) % 24
            day_of_week = (i // (24 * 12)) % 7
            
            if variation_type == 'kurtosis':
                # Kurtosis tends to be higher during high volatility periods
                if 14 <= hour <= 21:  # US trading hours
                    multiplier = 1.2
                else:
                    multiplier = 0.9
                
                # Add realistic variation
                noise = np.random.normal(0, 0.1)
                varied_value = base_value * multiplier + noise
                
                # Ensure reasonable bounds (excess kurtosis)
                varied_value = max(min(varied_value, 10.0), -1.0)
                
            elif variation_type == 'skewness':
                # Skewness can vary based on market sentiment
                noise = np.random.normal(0, 0.05)
                varied_value = base_value + noise
                
                # Ensure reasonable bounds
                varied_value = max(min(varied_value, 2.0), -2.0)
            
            variations.append(varied_value)
        
        return variations
    
    def simulate(self, predictions: List[Dict], initial_price: float,
                num_simulations: int = 1000) -> Tuple[pd.DataFrame, Dict]:
        """
        Intelligent simulation based on prediction analysis.
        """
        # Analyze predictions
        analysis = self.analyze_predictions(predictions)
        
        if 'error' in analysis:
            raise ValueError(f"Analysis failed: {analysis['error']}")
        
        print(f"🎯 Analysis Results:")
        print(f"   Total predictions: {analysis['total_predictions']}")
        print(f"   Volatility variation: {analysis['volatility_variation']} unique values")
        print(f"   Skewness variation: {analysis['skewness_variation']} unique values")
        print(f"   Kurtosis variation: {analysis['kurtosis_variation']} unique values")
        print(f"   Recommended mode: {analysis['recommended_mode']}")
        print(f"   Reason: {analysis['reason']}")
        
        # Extract parameters
        volatilities = [p['predicted_volatility'] for p in predictions]
        skewnesses = [p['predicted_skewness'] for p in predictions]
        kurtoses = [p['predicted_kurtosis'] for p in predictions]
        
        # Apply recommended mode
        if analysis['recommended_mode'] == 'constant_parameters':
            # Use average values
            avg_vol = np.mean(volatilities)
            avg_skew = np.mean(skewnesses)
            avg_kurt = np.mean(kurtoses)
            
            print(f"📊 Using constant parameters:")
            print(f"   Volatility: {avg_vol:.6f}")
            print(f"   Skewness: {avg_skew:.6f}")
            print(f"   Kurtosis: {avg_kurt:.6f}")
            
            return self._simulate_constant(avg_vol, avg_skew, avg_kurt,
                                         initial_price, num_simulations)
        
        elif analysis['recommended_mode'] == 'realistic_variation':
            # Generate realistic variation for constant parameters
            print(f"📊 Generating realistic variation...")
            
            enhanced_vols = volatilities  # Already varying
            enhanced_skews = self.generate_realistic_variation(
                np.mean(skewnesses), 'skewness')
            enhanced_kurts = self.generate_realistic_variation(
                np.mean(kurtoses), 'kurtosis')
            
            return self._simulate_time_varying(enhanced_vols, enhanced_skews, 
                                             enhanced_kurts, initial_price, num_simulations)
        
        else:  # time_varying
            # Use original predictions
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
        
        return simulation_results, summary_stats
    
    def _generate_returns_cornish_fisher(self, size: Tuple[int, int], 
                                      volatility: float, skewness: float, 
                                      kurtosis: float, dt: float) -> np.ndarray:
        """
        Generate returns using Cornish-Fisher expansion.
        Note: kurtosis is already excess kurtosis from the model.
        """
        # Generate standard normal returns
        normal_returns = np.random.normal(0, 1, size)
        
        # Apply Cornish-Fisher transformation
        cf_returns = normal_returns + \
                    (skewness / 6) * (normal_returns**2 - 1) + \
                    (kurtosis) / 24 * (normal_returns**3 - 3 * normal_returns) - \
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
    
    def plot_results(self, simulation_results: pd.DataFrame, summary_stats: Dict,
                    initial_price: float, save_path: str = None) -> None:
        """
        Create visualization of simulation results.
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'Enhanced Monte Carlo Simulation Results ({self.method.upper()})', 
                    fontsize=16, fontweight='bold')
        
        # 1. Price path evolution
        num_paths_to_plot = min(100, simulation_results.shape[1])
        selected_paths = np.random.choice(simulation_results.columns, num_paths_to_plot, replace=False)
        
        for path in selected_paths:
            axes[0, 0].plot(simulation_results.index, simulation_results[path], 
                           alpha=0.1, color='blue', linewidth=0.5)
        
        # Add percentiles
        percentiles = [5, 25, 50, 75, 95]
        colors = ['red', 'orange', 'green', 'orange', 'red']
        
        for p, color in zip(percentiles, colors):
            values = simulation_results.quantile(p/100, axis=1)
            axes[0, 0].plot(simulation_results.index, values, 
                           color=color, linewidth=2, label=f'{p}th percentile')
        
        axes[0, 0].axhline(y=initial_price, color='black', linestyle='--', label='Initial Price')
        axes[0, 0].set_title('Monte Carlo Price Paths')
        axes[0, 0].set_xlabel('Time Steps (5-min intervals)')
        axes[0, 0].set_ylabel('Price ($)')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Final price distribution
        final_prices = simulation_results.iloc[-1]
        axes[0, 1].hist(final_prices, bins=50, alpha=0.7, density=True, color='skyblue')
        axes[0, 1].axvline(initial_price, color='black', linestyle='--', label='Initial Price')
        axes[0, 1].axvline(summary_stats['mean_final_price'], color='red', linestyle='-', label='Mean Final Price')
        axes[0, 1].axvline(summary_stats['percentile_5'], color='orange', linestyle=':', label='5th Percentile')
        axes[0, 1].axvline(summary_stats['percentile_95'], color='orange', linestyle=':', label='95th Percentile')
        axes[0, 1].set_title('Final Price Distribution')
        axes[0, 1].set_xlabel('Final Price ($)')
        axes[0, 1].set_ylabel('Density')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Risk metrics
        risk_metrics = {
            'Probability of Profit': f"{summary_stats['probability_profit']:.1%}",
            'Probability of Loss': f"{summary_stats['probability_loss']:.1%}",
            'Max Drawdown': f"{summary_stats['max_drawdown']:.2%}",
            'VaR (95%)': f"${summary_stats['var_95']:,.2f}",
            'CVaR (95%)': f"${summary_stats['cvar_95']:,.2f}",
            'Expected Return': f"{summary_stats['expected_return']:.2%}",
            'Sharpe Ratio': f"{summary_stats['sharpe_ratio']:.3f}"
        }
        
        y_pos = np.arange(len(risk_metrics))
        metric_values = []
        for v in risk_metrics.values():
            if '%' in v:
                metric_values.append(float(v.replace('%', '')) / 100)
            elif '$' in v:
                metric_values.append(float(v.replace('$', '').replace(',', '')))
            else:
                metric_values.append(float(v))
        
        axes[1, 0].barh(y_pos, metric_values)
        axes[1, 0].set_yticks(y_pos)
        axes[1, 0].set_yticklabels(risk_metrics.keys())
        axes[1, 0].set_title('Risk Metrics')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Summary statistics
        summary_text = f"""
        Simulation Summary:
        
        Initial Price: ${initial_price:,.2f}
        Mean Final Price: ${summary_stats['mean_final_price']:,.2f}
        Std Final Price: ${summary_stats['std_final_price']:,.2f}
        
        Min Price: ${summary_stats['min_final_price']:,.2f}
        Max Price: ${summary_stats['max_final_price']:,.2f}
        
        5th Percentile: ${summary_stats['percentile_5']:,.2f}
        95th Percentile: ${summary_stats['percentile_95']:,.2f}
        
        Probability of Profit: {summary_stats['probability_profit']:.1%}
        Expected Return: {summary_stats['expected_return']:.2%}
        Sharpe Ratio: {summary_stats['sharpe_ratio']:.3f}
        
        Simulation Type: {summary_stats['simulation_type']}
        Method: {summary_stats['method']}
        Number of Simulations: {summary_stats['num_simulations']:,}
        """
        
        axes[1, 1].text(0.1, 0.9, summary_text, transform=axes[1, 1].transAxes, 
                       fontsize=10, verticalalignment='top', fontfamily='monospace')
        axes[1, 1].set_title('Simulation Summary')
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 Enhanced Monte Carlo plot saved to: {save_path}")
        
        plt.show()


def load_predictions_from_database(db_manager, crypto_symbol: str, 
                                 hours_back: int = 24) -> List[Dict]:
    """
    Load 288 predictions from database for Monte Carlo simulation.
    """
    try:
        # Get recent predictions from database
        recent_predictions = db_manager.get_recent_predictions(hours=hours_back)
        
        if not recent_predictions.empty:
            # Get the most recent batch of 288 predictions
            latest_batch = recent_predictions.iloc[-1]
            
            if 'predictions' in latest_batch and len(latest_batch['predictions']) == 288:
                print(f"✅ Loaded 288 predictions from database for {crypto_symbol}")
                return latest_batch['predictions']
            else:
                print(f"⚠️ Latest prediction batch doesn't contain 288 predictions")
                return None
        else:
            print(f"⚠️ No recent predictions found in database for {crypto_symbol}")
            return None
            
    except Exception as e:
        print(f"❌ Error loading predictions from database: {str(e)}")
        return None


def example_usage():
    """
    Example usage of the enhanced Monte Carlo simulator.
    """
    print("🚀 Enhanced Monte Carlo Simulator Example")
    print("=" * 50)
    
    # Initialize enhanced simulator
    simulator = EnhancedMonteCarloSimulator(method='cornish_fisher')
    
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
    
    # Run intelligent simulation
    print("\n🔮 Running intelligent simulation...")
    initial_price = 45000.0
    num_simulations = 1000
    
    simulation_results, summary_stats = simulator.simulate(
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
    
    # Create visualization
    print("\n📊 Creating visualization...")
    simulator.plot_results(
        simulation_results=simulation_results,
        summary_stats=summary_stats,
        initial_price=initial_price,
        save_path='results/enhanced_monte_carlo_example.png'
    )
    
    print("\n✅ Example completed!")
    print("\n💡 Usage with real database:")
    print("   1. Initialize DatabaseManager")
    print("   2. Load predictions using load_predictions_from_database()")
    print("   3. Use simulator.simulate() for intelligent simulation")
    print("   4. The simulator will automatically choose the best approach")


if __name__ == "__main__":
    example_usage() 
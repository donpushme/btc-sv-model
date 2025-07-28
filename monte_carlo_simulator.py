#!/usr/bin/env python3
"""
Monte Carlo Simulator for Cryptocurrency Volatility Predictions

This module provides Monte Carlo simulation capabilities using predictions from the 
real-time volatility prediction model. It supports both Cornish-Fisher expansion
and time-varying parameters for realistic price path generation.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from typing import Dict, List, Tuple, Optional, Union
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class MonteCarloSimulator:
    """
    Monte Carlo simulator for cryptocurrency price paths using predicted statistical moments.
    """
    
    def __init__(self, method: str = 'cornish_fisher'):
        """
        Initialize the Monte Carlo simulator.
        
        Args:
            method: Simulation method ('cornish_fisher' or 'normal')
        """
        self.method = method.lower()
        if self.method not in ['cornish_fisher', 'normal']:
            raise ValueError("Method must be 'cornish_fisher' or 'normal'")
    
    def generate_returns_cornish_fisher(self, size: Tuple[int, int], 
                                      volatility: float, skewness: float, 
                                      kurtosis: float, dt: float) -> np.ndarray:
        """
        Generate returns using Cornish-Fisher expansion for non-normal distributions.
        
        Args:
            size: (num_simulations, num_intervals)
            volatility: Predicted volatility (0.001 to 0.1 scale)
            skewness: Predicted skewness (-2 to +2 scale)
            kurtosis: Predicted excess kurtosis (-1 to +10 scale)
            dt: Time step (1/288 for 5-minute intervals)
            
        Returns:
            Array of returns with specified moments
        """
        # Generate standard normal returns
        normal_returns = np.random.normal(0, 1, size)
        
        # Apply Cornish-Fisher transformation to match target moments
        # Note: kurtosis is already excess kurtosis from the model
        cf_returns = normal_returns + \
                    (skewness / 6) * (normal_returns**2 - 1) + \
                    (kurtosis) / 24 * (normal_returns**3 - 3 * normal_returns) - \
                    (skewness**2) / 36 * (2 * normal_returns**3 - 5 * normal_returns)
        
        # Scale by volatility and time step
        # Model volatility is already in the correct scale (0.001 to 0.1)
        returns = cf_returns * volatility * np.sqrt(dt)
        
        return returns
    
    def generate_returns_normal(self, size: Tuple[int, int], 
                              volatility: float, dt: float) -> np.ndarray:
        """
        Generate returns using normal distribution (for comparison).
        
        Args:
            size: (num_simulations, num_intervals)
            volatility: Predicted volatility
            dt: Time step
            
        Returns:
            Array of normally distributed returns
        """
        return np.random.normal(0, volatility * np.sqrt(dt), size)
    
    def simulate_time_varying(self, predictions: List[Dict], initial_price: float,
                            num_simulations: int = 1000) -> Tuple[pd.DataFrame, Dict]:
        """
        Simulate price paths with time-varying parameters.
        
        Args:
            predictions: List of 288 prediction dictionaries from continuous predictor
            initial_price: Starting price
            num_simulations: Number of simulation paths
            
        Returns:
            Tuple of (simulation_results, summary_stats)
        """
        if not predictions or len(predictions) != 288:
            raise ValueError(f"Expected 288 predictions, got {len(predictions) if predictions else 0}")
        
        intervals = 288  # 24 hours × 12 five-minute intervals per hour
        dt = 1/288  # 5-minute time step
        
        # Extract time-varying parameters
        volatilities = [p['predicted_volatility'] for p in predictions]
        skewnesses = [p['predicted_skewness'] for p in predictions]
        kurtoses = [p['predicted_kurtosis'] for p in predictions]
        
        # Validate parameter bounds
        for i, (vol, skew, kurt) in enumerate(zip(volatilities, skewnesses, kurtoses)):
            if not (0.001 <= vol <= 0.1):
                print(f"Warning: Volatility {vol:.6f} at interval {i} outside bounds [0.001, 0.1]")
            if not (-2.0 <= skew <= 2.0):
                print(f"Warning: Skewness {skew:.6f} at interval {i} outside bounds [-2.0, 2.0]")
            if not (-1.0 <= kurt <= 10.0):
                print(f"Warning: Kurtosis {kurt:.6f} at interval {i} outside bounds [-1.0, 10.0]")
        
        # Generate price paths with time-varying parameters
        price_paths = np.zeros((num_simulations, intervals + 1))
        price_paths[:, 0] = initial_price
        
        for i in range(intervals):
            if self.method == 'cornish_fisher':
                # Generate returns for this specific time interval
                returns = self.generate_returns_cornish_fisher(
                    size=(num_simulations, 1),
                    volatility=volatilities[i],
                    skewness=skewnesses[i],
                    kurtosis=kurtoses[i],
                    dt=dt
                ).flatten()
            else:
                # Use normal distribution
                returns = self.generate_returns_normal(
                    size=(num_simulations, 1),
                    volatility=volatilities[i],
                    dt=dt
                ).flatten()
            
            # Update prices
            price_paths[:, i + 1] = price_paths[:, i] * (1 + returns)
        
        # Create results DataFrame
        simulation_results = pd.DataFrame(price_paths.T)
        simulation_results.index = range(len(simulation_results))
        
        # Calculate summary statistics
        summary_stats = self._calculate_summary_stats(price_paths, initial_price, predictions)
        summary_stats['simulation_type'] = 'time_varying'
        summary_stats['method'] = self.method
        
        return simulation_results, summary_stats
    
    def simulate_constant_parameters(self, predictions: List[Dict], initial_price: float,
                                   num_simulations: int = 1000) -> Tuple[pd.DataFrame, Dict]:
        """
        Simulate price paths with constant (average) parameters.
        
        Args:
            predictions: List of 288 prediction dictionaries
            initial_price: Starting price
            num_simulations: Number of simulation paths
            
        Returns:
            Tuple of (simulation_results, summary_stats)
        """
        if not predictions or len(predictions) != 288:
            raise ValueError(f"Expected 288 predictions, got {len(predictions) if predictions else 0}")
        
        intervals = 288
        dt = 1/288
        
        # Use average values for constant parameters
        avg_volatility = np.mean([p['predicted_volatility'] for p in predictions])
        avg_skewness = np.mean([p['predicted_skewness'] for p in predictions])
        avg_kurtosis = np.mean([p['predicted_kurtosis'] for p in predictions])
        
        print(f"📊 Using constant parameters:")
        print(f"   Average Volatility: {avg_volatility:.6f}")
        print(f"   Average Skewness: {avg_skewness:.6f}")
        print(f"   Average Kurtosis: {avg_kurtosis:.6f}")
        
        if self.method == 'cornish_fisher':
            # Generate all returns at once using average parameters
            returns = self.generate_returns_cornish_fisher(
                size=(num_simulations, intervals),
                volatility=avg_volatility,
                skewness=avg_skewness,
                kurtosis=avg_kurtosis,
                dt=dt
            )
        else:
            # Use normal distribution
            returns = self.generate_returns_normal(
                size=(num_simulations, intervals),
                volatility=avg_volatility,
                dt=dt
            )
        
        # Convert returns to price paths
        price_paths = np.zeros((num_simulations, intervals + 1))
        price_paths[:, 0] = initial_price
        
        for i in range(intervals):
            price_paths[:, i + 1] = price_paths[:, i] * (1 + returns[:, i])
        
        # Create results DataFrame
        simulation_results = pd.DataFrame(price_paths.T)
        simulation_results.index = range(len(simulation_results))
        
        # Calculate summary statistics
        summary_stats = self._calculate_summary_stats(price_paths, initial_price, predictions)
        summary_stats['simulation_type'] = 'constant_parameters'
        summary_stats['method'] = self.method
        
        return simulation_results, summary_stats
    
    def _calculate_summary_stats(self, price_paths: np.ndarray, initial_price: float,
                               predictions: List[Dict]) -> Dict:
        """
        Calculate comprehensive summary statistics for simulation results.
        """
        final_prices = price_paths[:, -1]
        returns_all = np.diff(np.log(price_paths), axis=1)
        
        # Basic price statistics
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
        
        # Add prediction statistics
        if predictions:
            pred_volatilities = [p['predicted_volatility'] for p in predictions]
            pred_skewnesses = [p['predicted_skewness'] for p in predictions]
            pred_kurtoses = [p['predicted_kurtosis'] for p in predictions]
            
            summary_stats.update({
                'pred_volatility_min': min(pred_volatilities),
                'pred_volatility_max': max(pred_volatilities),
                'pred_volatility_mean': np.mean(pred_volatilities),
                'pred_skewness_min': min(pred_skewnesses),
                'pred_skewness_max': max(pred_skewnesses),
                'pred_skewness_mean': np.mean(pred_skewnesses),
                'pred_kurtosis_min': min(pred_kurtoses),
                'pred_kurtosis_max': max(pred_kurtoses),
                'pred_kurtosis_mean': np.mean(pred_kurtoses)
            })
        
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
                    initial_price: float, predictions: List[Dict] = None,
                    save_path: str = None) -> None:
        """
        Create comprehensive visualization of Monte Carlo simulation results.
        """
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(f'Monte Carlo Simulation Results ({self.method.upper()})', 
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
        
        axes[0, 2].barh(y_pos, metric_values)
        axes[0, 2].set_yticks(y_pos)
        axes[0, 2].set_yticklabels(risk_metrics.keys())
        axes[0, 2].set_title('Risk Metrics')
        axes[0, 2].grid(True, alpha=0.3)
        
        # 4. Prediction analysis (if predictions provided)
        if predictions:
            pred_volatilities = [p['predicted_volatility'] for p in predictions]
            pred_skewnesses = [p['predicted_skewness'] for p in predictions]
            pred_kurtoses = [p['predicted_kurtosis'] for p in predictions]
            
            time_points = range(len(predictions))
            
            axes[1, 0].plot(time_points, pred_volatilities, color='red', linewidth=2, label='Predicted Volatility')
            axes[1, 0].axhline(y=summary_stats.get('pred_volatility_mean', 0), color='red', 
                              linestyle='--', alpha=0.7, label='Mean Volatility')
            axes[1, 0].set_title('Predicted Volatility Over Time')
            axes[1, 0].set_xlabel('Time Steps (5-min intervals)')
            axes[1, 0].set_ylabel('Volatility')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
            
            axes[1, 1].plot(time_points, pred_skewnesses, color='green', linewidth=2, label='Predicted Skewness')
            axes[1, 1].axhline(y=summary_stats.get('pred_skewness_mean', 0), color='green', 
                              linestyle='--', alpha=0.7, label='Mean Skewness')
            axes[1, 1].set_title('Predicted Skewness Over Time')
            axes[1, 1].set_xlabel('Time Steps (5-min intervals)')
            axes[1, 1].set_ylabel('Skewness')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
            
            axes[1, 2].plot(time_points, pred_kurtoses, color='purple', linewidth=2, label='Predicted Kurtosis')
            axes[1, 2].axhline(y=summary_stats.get('pred_kurtosis_mean', 0), color='purple', 
                              linestyle='--', alpha=0.7, label='Mean Kurtosis')
            axes[1, 2].set_title('Predicted Kurtosis Over Time')
            axes[1, 2].set_xlabel('Time Steps (5-min intervals)')
            axes[1, 2].set_ylabel('Kurtosis')
            axes[1, 2].legend()
            axes[1, 2].grid(True, alpha=0.3)
        else:
            # Show summary statistics
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
            
            axes[1, 0].text(0.1, 0.9, summary_text, transform=axes[1, 0].transAxes, 
                           fontsize=10, verticalalignment='top', fontfamily='monospace')
            axes[1, 0].set_title('Simulation Summary')
            axes[1, 0].axis('off')
            
            # Hide other subplots
            axes[1, 1].axis('off')
            axes[1, 2].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 Monte Carlo plot saved to: {save_path}")
        
        plt.show()


def load_predictions_from_database(db_manager, crypto_symbol: str, 
                                 hours_back: int = 24) -> List[Dict]:
    """
    Load 288 predictions from database for Monte Carlo simulation.
    
    Args:
        db_manager: Database manager instance
        crypto_symbol: Cryptocurrency symbol
        hours_back: How many hours back to look for predictions
        
    Returns:
        List of 288 prediction dictionaries
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
    Example usage of the Monte Carlo simulator with database predictions.
    """
    print("🚀 Monte Carlo Simulator Example")
    print("=" * 50)
    
    # Initialize simulator with Cornish-Fisher method
    simulator = MonteCarloSimulator(method='cornish_fisher')
    
    # Example: Load predictions from database (replace with your actual database connection)
    # from database_manager import DatabaseManager
    # db_manager = DatabaseManager(crypto_symbol='BTC')
    # predictions = load_predictions_from_database(db_manager, 'BTC', hours_back=24)
    
    # For demonstration, create sample predictions
    print("📊 Creating sample predictions for demonstration...")
    predictions = []
    base_volatility = 0.025  # 2.5% base volatility
    base_skewness = -0.1     # Slight negative skewness
    base_kurtosis = 4.0      # Fat tails
    
    for i in range(288):
        hour = (i // 12) % 24
        
        # US trading hours effect
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
            'predicted_skewness': base_skewness + np.random.normal(0, 0.05),
            'predicted_kurtosis': base_kurtosis + np.random.normal(0, 0.2),
            'current_price': 45000.0
        }
        predictions.append(prediction)
    
    print(f"✅ Generated {len(predictions)} sample predictions")
    
    # Run time-varying simulation
    print("\n🔮 Running time-varying Monte Carlo simulation...")
    initial_price = 45000.0
    num_simulations = 1000
    
    simulation_results, summary_stats = simulator.simulate_time_varying(
        predictions=predictions,
        initial_price=initial_price,
        num_simulations=num_simulations
    )
    
    print(f"📊 Time-varying simulation results:")
    print(f"   Mean final price: ${summary_stats['mean_final_price']:,.2f}")
    print(f"   Probability of profit: {summary_stats['probability_profit']:.1%}")
    print(f"   Expected return: {summary_stats['expected_return']:.2%}")
    print(f"   Sharpe ratio: {summary_stats['sharpe_ratio']:.3f}")
    print(f"   VaR (95%): ${summary_stats['var_95']:,.2f}")
    print(f"   Max drawdown: {summary_stats['max_drawdown']:.2%}")
    
    # Run constant parameter simulation for comparison
    print("\n🔮 Running constant parameter simulation...")
    simulation_results_const, summary_stats_const = simulator.simulate_constant_parameters(
        predictions=predictions,
        initial_price=initial_price,
        num_simulations=num_simulations
    )
    
    print(f"📊 Constant parameter simulation results:")
    print(f"   Mean final price: ${summary_stats_const['mean_final_price']:,.2f}")
    print(f"   Probability of profit: {summary_stats_const['probability_profit']:.1%}")
    print(f"   Expected return: {summary_stats_const['expected_return']:.2%}")
    
    # Create visualization
    print("\n📊 Creating visualization...")
    simulator.plot_results(
        simulation_results=simulation_results,
        summary_stats=summary_stats,
        initial_price=initial_price,
        predictions=predictions,
        save_path='results/monte_carlo_example.png'
    )
    
    print("\n✅ Example completed!")
    print("\n💡 Usage with real database:")
    print("   1. Initialize DatabaseManager")
    print("   2. Load predictions using load_predictions_from_database()")
    print("   3. Run simulator.simulate_time_varying() or simulate_constant_parameters()")
    print("   4. Use simulator.plot_results() for visualization")


if __name__ == "__main__":
    example_usage() 
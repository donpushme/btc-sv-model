#!/usr/bin/env python3
"""
Enhanced Monte Carlo Simulator for Cryptocurrency Volatility Predictions

This module provides advanced Monte Carlo simulation capabilities using predictions 
from the enhanced real-time volatility prediction model. It supports:
- Database integration with MongoDB
- Time-varying parameters from continuous predictions
- Multiple simulation methods (Cornish-Fisher, Normal, Student-t)
- Advanced risk metrics and visualization
- Real-time simulation with live data
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import norm, t
from typing import Dict, List, Tuple, Optional, Union, Any
from datetime import datetime, timedelta
import warnings
import json
import os
from pathlib import Path

# Import enhanced model components
from database_manager import DatabaseManager
from config import EnhancedConfig
from utils import format_prediction_output

warnings.filterwarnings('ignore')

class EnhancedMonteCarloSimulator:
    """
    Enhanced Monte Carlo simulator for cryptocurrency price paths using predicted statistical moments.
    Designed to work with the enhanced model's database predictions.
    """
    
    def __init__(self, crypto_symbol: str = "BTC", method: str = 'cornish_fisher', 
                 db_manager: Optional[DatabaseManager] = None):
        """
        Initialize the enhanced Monte Carlo simulator.
        
        Args:
            crypto_symbol: Cryptocurrency symbol (BTC, ETH, XAU, SOL)
            method: Simulation method ('cornish_fisher', 'normal', 'student_t', 'mixed')
            db_manager: Optional database manager instance
        """
        self.crypto_symbol = crypto_symbol
        self.method = method.lower()
        
        if self.method not in ['cornish_fisher', 'normal', 'student_t', 'mixed']:
            raise ValueError("Method must be 'cornish_fisher', 'normal', 'student_t', or 'mixed'")
        
        # Initialize database manager if not provided
        if db_manager is None:
            try:
                self.db_manager = DatabaseManager(crypto_symbol=crypto_symbol)
            except Exception as e:
                print(f"Warning: Database connection failed: {str(e)}")
                self.db_manager = None
        else:
            self.db_manager = db_manager
        
        # Configuration
        self.config = EnhancedConfig()
        self.intervals = 288  # 24 hours × 12 five-minute intervals per hour
        self.dt = 1/288  # 5-minute time step
        
        # Results storage
        self.simulation_results = None
        self.summary_stats = None
        
    def load_predictions_from_database(self, hours_back: int = 24, 
                                     limit: int = 100) -> List[Dict[str, Any]]:
        """
        Load predictions from the enhanced model database.
        
        Args:
            hours_back: Hours of historical data to load
            limit: Maximum number of prediction batches to load
            
        Returns:
            List of prediction dictionaries
        """
        if self.db_manager is None:
            raise ValueError("Database manager not available")
        
        try:
            # Get recent predictions from database
            predictions = self.db_manager.get_recent_predictions(hours=hours_back, limit=limit)
            
            if not predictions:
                raise ValueError(f"No predictions found for {self.crypto_symbol} in the last {hours_back} hours")
            
            # Extract the most recent prediction batch
            latest_prediction = predictions[0]
            
            # Validate the prediction structure
            if 'predictions' not in latest_prediction:
                raise ValueError("Invalid prediction format: missing 'predictions' field")
            
            # Return the predictions array from the latest batch
            return latest_prediction['predictions']
            
        except Exception as e:
            raise ValueError(f"Failed to load predictions from database: {str(e)}")
    
    def load_predictions_from_file(self, file_path: str) -> List[Dict[str, Any]]:
        """
        Load predictions from a JSON file (for testing/offline use).
        
        Args:
            file_path: Path to JSON file containing predictions
            
        Returns:
            List of prediction dictionaries
        """
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            if 'predictions' not in data:
                raise ValueError("Invalid file format: missing 'predictions' field")
            
            return data['predictions']
            
        except Exception as e:
            raise ValueError(f"Failed to load predictions from file: {str(e)}")
    
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
    
    def generate_returns_student_t(self, size: Tuple[int, int], 
                                 volatility: float, kurtosis: float, dt: float) -> np.ndarray:
        """
        Generate returns using Student's t-distribution for fat tails.
        
        Args:
            size: (num_simulations, num_intervals)
            volatility: Predicted volatility
            kurtosis: Predicted excess kurtosis
            dt: Time step
            
        Returns:
            Array of Student's t-distributed returns
        """
        # Calculate degrees of freedom from kurtosis
        # For Student's t: kurtosis = 6/(df-4) for df > 4
        if kurtosis <= 0:
            df = 10  # Default for low kurtosis
        else:
            df = max(5, 6/kurtosis + 4)
        
        # Generate Student's t returns
        t_returns = t.rvs(df=df, size=size)
        
        # Scale by volatility and time step
        returns = t_returns * volatility * np.sqrt(dt)
        
        return returns
    
    def generate_returns_mixed(self, size: Tuple[int, int], 
                             volatility: float, skewness: float, 
                             kurtosis: float, dt: float) -> np.ndarray:
        """
        Generate returns using a mixed distribution approach.
        
        Args:
            size: (num_simulations, num_intervals)
            volatility: Predicted volatility
            skewness: Predicted skewness
            kurtosis: Predicted excess kurtosis
            dt: Time step
            
        Returns:
            Array of mixed distribution returns
        """
        # Use Cornish-Fisher for moderate non-normality
        if abs(skewness) < 1.0 and kurtosis < 3.0:
            return self.generate_returns_cornish_fisher(size, volatility, skewness, kurtosis, dt)
        
        # Use Student's t for high kurtosis
        elif kurtosis >= 3.0:
            return self.generate_returns_student_t(size, volatility, kurtosis, dt)
        
        # Use normal for near-normal distributions
        else:
            return self.generate_returns_normal(size, volatility, dt)
    
    def simulate_time_varying(self, predictions: List[Dict], initial_price: float,
                            num_simulations: int = 1000) -> Tuple[pd.DataFrame, Dict]:
        """
        Simulate price paths with time-varying parameters from enhanced model predictions.
        
        Args:
            predictions: List of 288 prediction dictionaries from enhanced model
            initial_price: Starting price
            num_simulations: Number of simulation paths
            
        Returns:
            Tuple of (simulation_results, summary_stats)
        """
        if not predictions or len(predictions) != 288:
            raise ValueError(f"Expected 288 predictions, got {len(predictions) if predictions else 0}")
        
        # Initialize price paths
        price_paths = np.zeros((num_simulations, self.intervals + 1))
        price_paths[:, 0] = initial_price
        
        # Extract parameters for each interval
        volatilities = []
        skewnesses = []
        kurtoses = []
        
        for pred in predictions:
            volatilities.append(pred.get('predicted_volatility', 0.001))
            skewnesses.append(pred.get('predicted_skewness', 0.0))
            kurtoses.append(pred.get('predicted_kurtosis', 3.0))
        
        # Generate returns for each interval
        for i in range(self.intervals):
            vol = volatilities[i]
            skew = skewnesses[i]
            kurt = kurtoses[i]
            
            # Generate returns based on method
            if self.method == 'cornish_fisher':
                returns = self.generate_returns_cornish_fisher(
                    (num_simulations, 1), vol, skew, kurt, self.dt
                )
            elif self.method == 'normal':
                returns = self.generate_returns_normal(
                    (num_simulations, 1), vol, self.dt
                )
            elif self.method == 'student_t':
                returns = self.generate_returns_student_t(
                    (num_simulations, 1), vol, kurt, self.dt
                )
            elif self.method == 'mixed':
                returns = self.generate_returns_mixed(
                    (num_simulations, 1), vol, skew, kurt, self.dt
                )
            
            # Update prices
            price_paths[:, i + 1] = price_paths[:, i] * (1 + returns.flatten())
        
        # Create results DataFrame
        simulation_results = pd.DataFrame(price_paths)
        simulation_results.columns = [f't_{i}' for i in range(self.intervals + 1)]
        
        # Calculate summary statistics
        summary_stats = self._calculate_enhanced_summary_stats(
            price_paths, initial_price, predictions
        )
        
        self.simulation_results = simulation_results
        self.summary_stats = summary_stats
        
        return simulation_results, summary_stats
    
    def simulate_constant_parameters(self, predictions: List[Dict], initial_price: float,
                                   num_simulations: int = 1000) -> Tuple[pd.DataFrame, Dict]:
        """
        Simulate price paths with constant parameters (average of predictions).
        
        Args:
            predictions: List of prediction dictionaries
            initial_price: Starting price
            num_simulations: Number of simulation paths
            
        Returns:
            Tuple of (simulation_results, summary_stats)
        """
        if not predictions:
            raise ValueError("No predictions provided")
        
        # Calculate average parameters
        avg_volatility = np.mean([p.get('predicted_volatility', 0.001) for p in predictions])
        avg_skewness = np.mean([p.get('predicted_skewness', 0.0) for p in predictions])
        avg_kurtosis = np.mean([p.get('predicted_kurtosis', 3.0) for p in predictions])
        
        # Initialize price paths
        price_paths = np.zeros((num_simulations, self.intervals + 1))
        price_paths[:, 0] = initial_price
        
        # Generate returns for all intervals at once
        if self.method == 'cornish_fisher':
            returns = self.generate_returns_cornish_fisher(
                (num_simulations, self.intervals), avg_volatility, avg_skewness, avg_kurtosis, self.dt
            )
        elif self.method == 'normal':
            returns = self.generate_returns_normal(
                (num_simulations, self.intervals), avg_volatility, self.dt
            )
        elif self.method == 'student_t':
            returns = self.generate_returns_student_t(
                (num_simulations, self.intervals), avg_volatility, avg_kurtosis, self.dt
            )
        elif self.method == 'mixed':
            returns = self.generate_returns_mixed(
                (num_simulations, self.intervals), avg_volatility, avg_skewness, avg_kurtosis, self.dt
            )
        
        # Update prices
        for i in range(self.intervals):
            price_paths[:, i + 1] = price_paths[:, i] * (1 + returns[:, i])
        
        # Create results DataFrame
        simulation_results = pd.DataFrame(price_paths)
        simulation_results.columns = [f't_{i}' for i in range(self.intervals + 1)]
        
        # Calculate summary statistics
        summary_stats = self._calculate_enhanced_summary_stats(
            price_paths, initial_price, predictions
        )
        
        self.simulation_results = simulation_results
        self.summary_stats = summary_stats
        
        return simulation_results, summary_stats 
    
    def _calculate_enhanced_summary_stats(self, price_paths: np.ndarray, initial_price: float,
                                        predictions: List[Dict]) -> Dict:
        """
        Calculate comprehensive summary statistics for simulation results.
        
        Args:
            price_paths: Array of simulated price paths
            initial_price: Initial price
            predictions: Original predictions used for simulation
            
        Returns:
            Dictionary of summary statistics
        """
        # Basic statistics
        final_prices = price_paths[:, -1]
        returns = (final_prices - initial_price) / initial_price
        
        # Risk metrics
        var_95 = np.percentile(returns, 5)
        var_99 = np.percentile(returns, 1)
        cvar_95 = np.mean(returns[returns <= var_95])
        cvar_99 = np.mean(returns[returns <= var_99])
        
        # Maximum drawdown
        max_drawdown = self._calculate_max_drawdown(price_paths)
        
        # Volatility clustering
        path_volatilities = []
        for path in price_paths:
            path_returns = np.diff(path) / path[:-1]
            path_volatilities.append(np.std(path_returns))
        
        # Confidence intervals
        price_ci_95 = np.percentile(final_prices, [2.5, 97.5])
        price_ci_99 = np.percentile(final_prices, [0.5, 99.5])
        
        # Extract prediction statistics
        pred_volatilities = [p.get('predicted_volatility', 0.001) for p in predictions]
        pred_skewnesses = [p.get('predicted_skewness', 0.0) for p in predictions]
        pred_kurtoses = [p.get('predicted_kurtosis', 3.0) for p in predictions]
        
        summary_stats = {
            'simulation_method': self.method,
            'num_simulations': price_paths.shape[0],
            'initial_price': initial_price,
            'final_price_stats': {
                'mean': np.mean(final_prices),
                'median': np.median(final_prices),
                'std': np.std(final_prices),
                'min': np.min(final_prices),
                'max': np.max(final_prices),
                'ci_95_lower': price_ci_95[0],
                'ci_95_upper': price_ci_95[1],
                'ci_99_lower': price_ci_99[0],
                'ci_99_upper': price_ci_99[1]
            },
            'return_stats': {
                'mean': np.mean(returns),
                'median': np.median(returns),
                'std': np.std(returns),
                'skewness': stats.skew(returns),
                'kurtosis': stats.kurtosis(returns),
                'var_95': var_95,
                'var_99': var_99,
                'cvar_95': cvar_95,
                'cvar_99': cvar_99
            },
            'risk_metrics': {
                'max_drawdown': max_drawdown,
                'volatility_of_volatility': np.std(path_volatilities),
                'mean_volatility': np.mean(path_volatilities)
            },
            'prediction_stats': {
                'mean_volatility': np.mean(pred_volatilities),
                'mean_skewness': np.mean(pred_skewnesses),
                'mean_kurtosis': np.mean(pred_kurtoses),
                'volatility_range': [np.min(pred_volatilities), np.max(pred_volatilities)],
                'skewness_range': [np.min(pred_skewnesses), np.max(pred_skewnesses)],
                'kurtosis_range': [np.min(pred_kurtoses), np.max(pred_kurtoses)]
            },
            'simulation_timestamp': datetime.now().isoformat()
        }
        
        return summary_stats
    
    def _calculate_max_drawdown(self, price_paths: np.ndarray) -> float:
        """
        Calculate maximum drawdown across all simulation paths.
        
        Args:
            price_paths: Array of simulated price paths
            
        Returns:
            Maximum drawdown as a percentage
        """
        max_drawdown = 0
        
        for path in price_paths:
            peak = path[0]
            drawdown = 0
            
            for price in path:
                if price > peak:
                    peak = price
                else:
                    current_drawdown = (peak - price) / peak
                    drawdown = max(drawdown, current_drawdown)
            
            max_drawdown = max(max_drawdown, drawdown)
        
        return max_drawdown
    
    def plot_enhanced_results(self, simulation_results: pd.DataFrame, summary_stats: Dict,
                            initial_price: float, predictions: List[Dict] = None,
                            save_path: str = None, show_plots: bool = True) -> None:
        """
        Create comprehensive visualization of simulation results.
        
        Args:
            simulation_results: DataFrame of simulation results
            summary_stats: Summary statistics dictionary
            initial_price: Initial price
            predictions: Original predictions used for simulation
            save_path: Path to save plots
            show_plots: Whether to display plots
        """
        # Set up the plotting style
        plt.style.use('seaborn-v0_8')
        fig = plt.figure(figsize=(20, 16))
        
        # 1. Price paths plot
        ax1 = plt.subplot(3, 3, 1)
        for i in range(min(100, len(simulation_results))):  # Plot first 100 paths
            plt.plot(simulation_results.iloc[i], alpha=0.1, color='blue')
        
        # Plot confidence intervals
        final_prices = simulation_results.iloc[:, -1]
        ci_95 = summary_stats['final_price_stats']['ci_95_lower'], summary_stats['final_price_stats']['ci_95_upper']
        ci_99 = summary_stats['final_price_stats']['ci_99_lower'], summary_stats['final_price_stats']['ci_99_upper']
        
        plt.axhline(y=ci_95[0], color='orange', linestyle='--', alpha=0.7, label='95% CI')
        plt.axhline(y=ci_95[1], color='orange', linestyle='--', alpha=0.7)
        plt.axhline(y=ci_99[0], color='red', linestyle='--', alpha=0.7, label='99% CI')
        plt.axhline(y=ci_99[1], color='red', linestyle='--', alpha=0.7)
        plt.axhline(y=initial_price, color='black', linestyle='-', linewidth=2, label='Initial Price')
        
        plt.title(f'{self.crypto_symbol} Price Paths Simulation\n({self.method.upper()} Method)')
        plt.xlabel('Time Steps (5-min intervals)')
        plt.ylabel('Price')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 2. Final price distribution
        ax2 = plt.subplot(3, 3, 2)
        plt.hist(final_prices, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        plt.axvline(x=initial_price, color='red', linestyle='--', linewidth=2, label='Initial Price')
        plt.axvline(x=np.mean(final_prices), color='green', linestyle='--', linewidth=2, label='Mean Final Price')
        plt.title('Final Price Distribution')
        plt.xlabel('Final Price')
        plt.ylabel('Frequency')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 3. Return distribution
        ax3 = plt.subplot(3, 3, 3)
        returns = (final_prices - initial_price) / initial_price
        plt.hist(returns, bins=50, alpha=0.7, color='lightgreen', edgecolor='black')
        plt.axvline(x=0, color='red', linestyle='--', linewidth=2, label='No Change')
        plt.axvline(x=np.mean(returns), color='green', linestyle='--', linewidth=2, label='Mean Return')
        plt.title('Return Distribution')
        plt.xlabel('Return')
        plt.ylabel('Frequency')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 4. Risk metrics
        ax4 = plt.subplot(3, 3, 4)
        risk_metrics = [
            summary_stats['return_stats']['var_95'],
            summary_stats['return_stats']['var_99'],
            summary_stats['return_stats']['cvar_95'],
            summary_stats['return_stats']['cvar_99'],
            summary_stats['risk_metrics']['max_drawdown']
        ]
        risk_labels = ['VaR 95%', 'VaR 99%', 'CVaR 95%', 'CVaR 99%', 'Max DD']
        colors = ['red', 'darkred', 'orange', 'darkorange', 'purple']
        
        bars = plt.bar(risk_labels, risk_metrics, color=colors, alpha=0.7)
        plt.title('Risk Metrics')
        plt.ylabel('Value')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, value in zip(bars, risk_metrics):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                    f'{value:.3f}', ha='center', va='bottom', fontsize=8)
        
        # 5. Prediction parameters over time
        if predictions:
            ax5 = plt.subplot(3, 3, 5)
            times = range(len(predictions))
            volatilities = [p.get('predicted_volatility', 0.001) for p in predictions]
            skewnesses = [p.get('predicted_skewness', 0.0) for p in predictions]
            kurtoses = [p.get('predicted_kurtosis', 3.0) for p in predictions]
            
            plt.plot(times, volatilities, label='Volatility', color='blue', alpha=0.7)
            plt.plot(times, np.abs(skewnesses), label='|Skewness|', color='red', alpha=0.7)
            plt.plot(times, kurtoses, label='Kurtosis', color='green', alpha=0.7)
            plt.title('Prediction Parameters Over Time')
            plt.xlabel('Time Steps')
            plt.ylabel('Value')
            plt.legend()
            plt.grid(True, alpha=0.3)
        
        # 6. QQ plot for normality test
        ax6 = plt.subplot(3, 3, 6)
        stats.probplot(returns, dist="norm", plot=plt)
        plt.title('Q-Q Plot (Normality Test)')
        plt.grid(True, alpha=0.3)
        
        # 7. Summary statistics table
        ax7 = plt.subplot(3, 3, 7)
        ax7.axis('tight')
        ax7.axis('off')
        
        # Create summary table
        table_data = [
            ['Metric', 'Value'],
            ['Initial Price', f"${initial_price:,.2f}"],
            ['Mean Final Price', f"${summary_stats['final_price_stats']['mean']:,.2f}"],
            ['Mean Return', f"{summary_stats['return_stats']['mean']:.2%}"],
            ['Volatility', f"{summary_stats['return_stats']['std']:.2%}"],
            ['Skewness', f"{summary_stats['return_stats']['skewness']:.3f}"],
            ['Kurtosis', f"{summary_stats['return_stats']['kurtosis']:.3f}"],
            ['Max Drawdown', f"{summary_stats['risk_metrics']['max_drawdown']:.2%}"],
            ['VaR 95%', f"{summary_stats['return_stats']['var_95']:.2%}"],
            ['CVaR 95%', f"{summary_stats['return_stats']['cvar_95']:.2%}"]
        ]
        
        table = ax7.table(cellText=table_data[1:], colLabels=table_data[0],
                         cellLoc='center', loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 2)
        ax7.set_title('Summary Statistics', fontsize=12, fontweight='bold')
        
        # 8. Volatility clustering
        ax8 = plt.subplot(3, 3, 8)
        path_volatilities = []
        for i in range(min(50, len(simulation_results))):  # Sample 50 paths
            path = simulation_results.iloc[i]
            path_returns = np.diff(path) / path[:-1]
            rolling_vol = pd.Series(path_returns).rolling(window=12).std()
            path_volatilities.append(rolling_vol.values)
        
        path_volatilities = np.array(path_volatilities)
        mean_vol = np.nanmean(path_volatilities, axis=0)
        std_vol = np.nanstd(path_volatilities, axis=0)
        
        times = range(len(mean_vol))
        plt.plot(times, mean_vol, color='blue', alpha=0.7, label='Mean Volatility')
        plt.fill_between(times, mean_vol - std_vol, mean_vol + std_vol, 
                        alpha=0.3, color='blue', label='±1 Std Dev')
        plt.title('Volatility Clustering')
        plt.xlabel('Time Steps')
        plt.ylabel('Rolling Volatility')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 9. Simulation metadata
        ax9 = plt.subplot(3, 3, 9)
        ax9.axis('tight')
        ax9.axis('off')
        
        metadata = [
            ['Parameter', 'Value'],
            ['Crypto Symbol', self.crypto_symbol],
            ['Simulation Method', self.method.upper()],
            ['Number of Paths', f"{summary_stats['num_simulations']:,}"],
            ['Time Horizon', '24 hours'],
            ['Time Step', '5 minutes'],
            ['Simulation Date', datetime.now().strftime('%Y-%m-%d %H:%M:%S')]
        ]
        
        meta_table = ax9.table(cellText=metadata[1:], colLabels=metadata[0],
                              cellLoc='center', loc='center')
        meta_table.auto_set_font_size(False)
        meta_table.set_fontsize(9)
        meta_table.scale(1, 2)
        ax9.set_title('Simulation Metadata', fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 Enhanced simulation plots saved to: {save_path}")
        
        if show_plots:
            plt.show()
        else:
            plt.close()
    
    def save_simulation_results(self, file_path: str) -> None:
        """
        Save simulation results to JSON file.
        
        Args:
            file_path: Path to save the results
        """
        if self.simulation_results is None or self.summary_stats is None:
            raise ValueError("No simulation results to save. Run simulation first.")
        
        results_data = {
            'crypto_symbol': self.crypto_symbol,
            'simulation_method': self.method,
            'simulation_timestamp': datetime.now().isoformat(),
            'summary_stats': self.summary_stats,
            'price_paths': self.simulation_results.to_dict('records'),
            'metadata': {
                'num_simulations': len(self.simulation_results),
                'num_intervals': self.intervals,
                'time_step_minutes': 5,
                'total_hours': 24
            }
        }
        
        with open(file_path, 'w') as f:
            json.dump(results_data, f, indent=2, default=str)
        
        print(f"💾 Simulation results saved to: {file_path}")
    
    def run_simulation_from_database(self, initial_price: float, num_simulations: int = 1000,
                                   hours_back: int = 24, use_time_varying: bool = True,
                                   save_results: bool = True, show_plots: bool = True) -> Tuple[pd.DataFrame, Dict]:
        """
        Run complete simulation using data from database.
        
        Args:
            initial_price: Starting price for simulation
            num_simulations: Number of simulation paths
            hours_back: Hours of historical data to load
            use_time_varying: Whether to use time-varying parameters
            save_results: Whether to save results to file
            show_plots: Whether to display plots
            
        Returns:
            Tuple of (simulation_results, summary_stats)
        """
        print(f"🚀 Starting enhanced Monte Carlo simulation for {self.crypto_symbol}")
        print(f"📊 Method: {self.method.upper()}")
        print(f"🎯 Initial Price: ${initial_price:,.2f}")
        print(f"🔄 Number of Simulations: {num_simulations:,}")
        
        # Load predictions from database
        print("📥 Loading predictions from database...")
        predictions = self.load_predictions_from_database(hours_back=hours_back)
        print(f"✅ Loaded {len(predictions)} predictions")
        
        # Run simulation
        print("⚡ Running simulation...")
        if use_time_varying:
            simulation_results, summary_stats = self.simulate_time_varying(
                predictions, initial_price, num_simulations
            )
        else:
            simulation_results, summary_stats = self.simulate_constant_parameters(
                predictions, initial_price, num_simulations
            )
        
        print("✅ Simulation completed!")
        
        # Save results
        if save_results:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            results_dir = Path(self.config.RESULTS_PATH)
            results_dir.mkdir(exist_ok=True)
            
            results_file = results_dir / f"{self.crypto_symbol}_simulation_{timestamp}.json"
            plots_file = results_dir / f"{self.crypto_symbol}_simulation_{timestamp}.png"
            
            self.save_simulation_results(str(results_file))
            
            # Create plots
            print("📊 Generating visualization...")
            self.plot_enhanced_results(
                simulation_results, summary_stats, initial_price, predictions,
                save_path=str(plots_file), show_plots=show_plots
            )
        
        return simulation_results, summary_stats


def example_usage():
    """
    Example usage of the enhanced Monte Carlo simulator.
    """
    # Initialize simulator
    simulator = EnhancedMonteCarloSimulator(
        crypto_symbol="BTC",
        method="cornish_fisher"
    )
    
    # Run simulation with current price
    current_price = 117760.88  # Example price from your data
    
    try:
        simulation_results, summary_stats = simulator.run_simulation_from_database(
            initial_price=current_price,
            num_simulations=1000,
            hours_back=24,
            use_time_varying=True,
            save_results=True,
            show_plots=True
        )
        
        print("\n📈 Simulation Summary:")
        print(f"Mean Final Price: ${summary_stats['final_price_stats']['mean']:,.2f}")
        print(f"Mean Return: {summary_stats['return_stats']['mean']:.2%}")
        print(f"Volatility: {summary_stats['return_stats']['std']:.2%}")
        print(f"VaR 95%: {summary_stats['return_stats']['var_95']:.2%}")
        print(f"Max Drawdown: {summary_stats['risk_metrics']['max_drawdown']:.2%}")
        
    except Exception as e:
        print(f"❌ Simulation failed: {str(e)}")


if __name__ == "__main__":
    example_usage() 
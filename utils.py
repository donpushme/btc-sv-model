import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

def validate_crypto_data(df: pd.DataFrame) -> Dict[str, any]:
    """
    Validate cryptocurrency price data for completeness and quality.
    
    Args:
        df: DataFrame with cryptocurrency price data
        
    Returns:
        Dictionary with validation results
    """
    validation_results = {
        'is_valid': True,
        'errors': [],
        'warnings': [],
        'statistics': {}
    }
    
    # Check required columns
    required_cols = ['timestamp', 'open', 'close', 'high', 'low']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        validation_results['is_valid'] = False
        validation_results['errors'].append(f"Missing required columns: {missing_cols}")
    
    if not validation_results['is_valid']:
        return validation_results
    
    # Check data types
    try:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
    except:
        validation_results['errors'].append("Timestamp column cannot be converted to datetime")
        validation_results['is_valid'] = False
    
    for col in ['open', 'close', 'high', 'low']:
        if not pd.api.types.is_numeric_dtype(df[col]):
            validation_results['errors'].append(f"Column {col} is not numeric")
            validation_results['is_valid'] = False
    
    if not validation_results['is_valid']:
        return validation_results
    
    # Check for missing values
    missing_count = df.isnull().sum().sum()
    if missing_count > 0:
        validation_results['warnings'].append(f"Found {missing_count} missing values")
    
    # Check for negative prices
    price_cols = ['open', 'close', 'high', 'low']
    negative_prices = (df[price_cols] <= 0).any().any()
    if negative_prices:
        validation_results['errors'].append("Found negative or zero prices")
        validation_results['is_valid'] = False
    
    # Check OHLC logic
    invalid_ohlc = (
        (df['high'] < df['low']) |
        (df['high'] < df['open']) |
        (df['high'] < df['close']) |
        (df['low'] > df['open']) |
        (df['low'] > df['close'])
    ).sum()
    
    if invalid_ohlc > 0:
        validation_results['warnings'].append(f"Found {invalid_ohlc} rows with invalid OHLC relationships")
    
    # Check for duplicated timestamps
    duplicated_timestamps = df['timestamp'].duplicated().sum()
    if duplicated_timestamps > 0:
        validation_results['warnings'].append(f"Found {duplicated_timestamps} duplicated timestamps")
    
    # Check time intervals
    df_sorted = df.sort_values('timestamp')
    time_diffs = df_sorted['timestamp'].diff().dropna()
    
    if len(time_diffs) > 0:
        most_common_interval = time_diffs.mode().iloc[0] if len(time_diffs.mode()) > 0 else None
        irregular_intervals = (time_diffs != most_common_interval).sum()
        
        validation_results['statistics']['most_common_interval'] = str(most_common_interval)
        validation_results['statistics']['irregular_intervals'] = int(irregular_intervals)
        
        if irregular_intervals > len(df) * 0.05:  # More than 5% irregular
            validation_results['warnings'].append(f"High number of irregular intervals: {irregular_intervals}")
    
    # Data quality statistics
    validation_results['statistics'].update({
        'total_rows': len(df),
        'date_range_start': str(df['timestamp'].min()),
        'date_range_end': str(df['timestamp'].max()),
        'missing_values': int(missing_count),
        'price_range_min': float(df[price_cols].min().min()),
        'price_range_max': float(df[price_cols].max().max()),
        'average_price': float(df['close'].mean()),
        'price_volatility': float(df['close'].pct_change().std())
    })
    
    return validation_results


def monte_carlo_simulation(volatility: float, skewness: float, kurtosis: float,
                          initial_price: float, intervals: int = 288,
                          num_simulations: int = 1000, dt: float = 1/288) -> pd.DataFrame:
    """
    Perform Monte Carlo simulation using predicted volatility, skewness, and kurtosis.
    
    Args:
        volatility: Predicted volatility
        skewness: Predicted skewness
        kurtosis: Predicted kurtosis
        initial_price: Starting Bitcoin price
        intervals: Number of time intervals (288 = 24 hours at 5-min intervals)
        num_simulations: Number of simulation paths
        dt: Time step (1/288 for 5-minute intervals in a day)
        
    Returns:
        DataFrame with simulation results
    """
    # Generate random returns with specified moments
    returns = generate_skewed_kurtotic_returns(
        size=(num_simulations, intervals),
        volatility=volatility,
        skewness=skewness,
        kurtosis=kurtosis,
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
    
    # Add summary statistics
    summary_stats = {
        'mean_final_price': price_paths[:, -1].mean(),
        'std_final_price': price_paths[:, -1].std(),
        'min_final_price': price_paths[:, -1].min(),
        'max_final_price': price_paths[:, -1].max(),
        'percentile_5': np.percentile(price_paths[:, -1], 5),
        'percentile_95': np.percentile(price_paths[:, -1], 95),
        'probability_profit': (price_paths[:, -1] > initial_price).mean(),
        'max_drawdown': calculate_max_drawdown(price_paths),
        'volatility_realized': np.std(np.diff(np.log(price_paths), axis=1)) * np.sqrt(1/dt)
    }
    
    return simulation_results, summary_stats


def generate_monte_carlo_from_predictions(predictions: List[Dict], initial_price: float,
                                        num_simulations: int = 1000, 
                                        time_varying: bool = True) -> Tuple[pd.DataFrame, Dict]:
    """
    Generate Monte Carlo simulation using 288 predictions from continuous predictor.
    
    This function can handle both time-varying predictions (each 5-minute interval has different
    volatility, skewness, kurtosis) and constant predictions (using average values).
    
    Args:
        predictions: List of 288 prediction dictionaries from continuous predictor
        initial_price: Starting Bitcoin price
        num_simulations: Number of simulation paths
        time_varying: If True, use time-varying predictions; if False, use average values
        
    Returns:
        Tuple of (simulation_results, summary_stats)
    """
    if not predictions or len(predictions) != 288:
        raise ValueError(f"Expected 288 predictions, got {len(predictions) if predictions else 0}")
    
    intervals = 288  # 24 hours × 12 five-minute intervals per hour
    dt = 1/288  # 5-minute time step
    
    if time_varying:
        # Extract time-varying parameters
        volatilities = [p['predicted_volatility'] for p in predictions]
        skewnesses = [p['predicted_skewness'] for p in predictions]
        kurtoses = [p['predicted_kurtosis'] for p in predictions]
        
        # Generate price paths with time-varying parameters
        price_paths = np.zeros((num_simulations, intervals + 1))
        price_paths[:, 0] = initial_price
        
        for i in range(intervals):
            # Generate returns for this specific time interval
            returns = generate_skewed_kurtotic_returns(
                size=(num_simulations, 1),
                volatility=volatilities[i],
                skewness=skewnesses[i],
                kurtosis=kurtoses[i],
                dt=dt
            ).flatten()
            
            # Update prices
            price_paths[:, i + 1] = price_paths[:, i] * (1 + returns)
            
    else:
        # Use average values for constant parameters
        avg_volatility = np.mean([p['predicted_volatility'] for p in predictions])
        avg_skewness = np.mean([p['predicted_skewness'] for p in predictions])
        avg_kurtosis = np.mean([p['predicted_kurtosis'] for p in predictions])
        
        # Generate all returns at once using average parameters
        returns = generate_skewed_kurtotic_returns(
            size=(num_simulations, intervals),
            volatility=avg_volatility,
            skewness=avg_skewness,
            kurtosis=avg_kurtosis,
            dt=dt
        )
        
        # Convert returns to price paths
        price_paths = np.zeros((num_simulations, intervals + 1))
        price_paths[:, 0] = initial_price
        
        for i in range(intervals):
            price_paths[:, i + 1] = price_paths[:, i] * (1 + returns[:, i])
    
    # Create results DataFrame
    simulation_results = pd.DataFrame(price_paths)
    simulation_results.index = range(len(simulation_results))
    
    # Calculate comprehensive summary statistics
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
        'max_drawdown': calculate_max_drawdown(price_paths),
        'volatility_realized': np.std(returns_all) * np.sqrt(1/dt),
        'skewness_realized': float(stats.skew(returns_all.flatten())),
        'kurtosis_realized': float(stats.kurtosis(returns_all.flatten())),
        'var_95': np.percentile(final_prices - initial_price, 5),
        'cvar_95': np.mean(final_prices[final_prices <= np.percentile(final_prices, 5)] - initial_price),
        'expected_return': (final_prices.mean() - initial_price) / initial_price,
        'sharpe_ratio': ((final_prices.mean() - initial_price) / initial_price) / (final_prices.std() / initial_price),
        'simulation_type': 'time_varying' if time_varying else 'constant_parameters',
        'num_simulations': num_simulations,
        'intervals': intervals,
        'initial_price': initial_price
    }
    
    # Add prediction statistics if available
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
    
    return simulation_results, summary_stats


def plot_enhanced_monte_carlo_results(simulation_results: pd.DataFrame, summary_stats: Dict,
                                    initial_price: float, predictions: List[Dict] = None,
                                    save_path: str = None) -> None:
    """
    Create enhanced plots for Monte Carlo simulation results with prediction analysis.
    
    Args:
        simulation_results: DataFrame with simulation results
        summary_stats: Dictionary with summary statistics
        initial_price: Starting price
        predictions: List of 288 predictions (optional, for prediction analysis)
        save_path: Path to save the plot (optional)
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Enhanced Monte Carlo Simulation Results', fontsize=16, fontweight='bold')
    
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
    axes[0, 2].barh(y_pos, [float(v.replace('%', '').replace('$', '').replace(',', '')) 
                           if '%' in v or '$' in v else float(v) for v in risk_metrics.values()])
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
        # If no predictions, show summary statistics
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
        print(f"📊 Enhanced Monte Carlo plot saved to: {save_path}")
    
    plt.show()


def example_monte_carlo_with_predictions():
    """
    Example function showing how to use the new Monte Carlo simulation with 288 predictions.
    This demonstrates both time-varying and constant parameter approaches.
    """
    print("🚀 Example: Monte Carlo Simulation with 288 Predictions")
    print("=" * 60)
    
    # Simulate 288 predictions (in real use, these would come from continuous_predictor.py)
    print("📊 Generating sample 288 predictions...")
    
    # Create sample predictions with realistic patterns
    predictions = []
    base_volatility = 0.025  # 2.5% base volatility
    base_skewness = -0.1     # Slight negative skewness
    base_kurtosis = 4.0      # Fat tails
    
    for i in range(288):
        # Add time-varying patterns (US trading hours, weekends, etc.)
        hour = (i // 12) % 24  # Hour of day (0-23)
        
        # US trading hours effect (14:30-21:00 UTC = 9:30 AM-4:00 PM EST)
        if 14 <= hour <= 21:
            vol_multiplier = 1.3
        elif 22 <= hour <= 2:
            vol_multiplier = 1.1
        elif 3 <= hour <= 9:
            vol_multiplier = 0.9
        else:
            vol_multiplier = 0.7
        
        # Weekend effect
        day_of_week = (i // (24 * 12)) % 7  # Day of week (0=Monday, 6=Sunday)
        if day_of_week >= 5:  # Weekend
            vol_multiplier *= 0.6
        
        # Add some realistic variation
        noise = np.random.normal(1.0, 0.05)
        final_vol_multiplier = vol_multiplier * noise
        
        prediction = {
            'sequence_number': i + 1,
            'timestamp': f"2024-01-15T{(hour):02d}:{(i % 12) * 5:02d}:00",
            'minutes_ahead': (i + 1) * 5,
            'predicted_volatility': base_volatility * final_vol_multiplier,
            'predicted_skewness': base_skewness + np.random.normal(0, 0.05),
            'predicted_kurtosis': base_kurtosis + np.random.normal(0, 0.2),
            'current_price': 45000.0,
            'is_us_trading_hours': 14 <= hour <= 21,
            'is_weekend': day_of_week >= 5
        }
        predictions.append(prediction)
    
    print(f"✅ Generated {len(predictions)} sample predictions")
    
    # Example 1: Time-varying Monte Carlo simulation
    print("\n🔮 Example 1: Time-varying Monte Carlo Simulation")
    print("-" * 50)
    
    initial_price = 45000.0
    num_simulations = 1000
    
    simulation_results, summary_stats = generate_monte_carlo_from_predictions(
        predictions=predictions,
        initial_price=initial_price,
        num_simulations=num_simulations,
        time_varying=True  # Use time-varying parameters
    )
    
    print(f"📊 Time-varying simulation results:")
    print(f"   Mean final price: ${summary_stats['mean_final_price']:,.2f}")
    print(f"   Probability of profit: {summary_stats['probability_profit']:.1%}")
    print(f"   Expected return: {summary_stats['expected_return']:.2%}")
    print(f"   Sharpe ratio: {summary_stats['sharpe_ratio']:.3f}")
    print(f"   VaR (95%): ${summary_stats['var_95']:,.2f}")
    print(f"   Max drawdown: {summary_stats['max_drawdown']:.2%}")
    
    # Example 2: Constant parameter Monte Carlo simulation
    print("\n🔮 Example 2: Constant Parameter Monte Carlo Simulation")
    print("-" * 50)
    
    simulation_results_const, summary_stats_const = generate_monte_carlo_from_predictions(
        predictions=predictions,
        initial_price=initial_price,
        num_simulations=num_simulations,
        time_varying=False  # Use average parameters
    )
    
    print(f"📊 Constant parameter simulation results:")
    print(f"   Mean final price: ${summary_stats_const['mean_final_price']:,.2f}")
    print(f"   Probability of profit: {summary_stats_const['probability_profit']:.1%}")
    print(f"   Expected return: {summary_stats_const['expected_return']:.2%}")
    print(f"   Sharpe ratio: {summary_stats_const['sharpe_ratio']:.3f}")
    print(f"   VaR (95%): ${summary_stats_const['var_95']:,.2f}")
    print(f"   Max drawdown: {summary_stats_const['max_drawdown']:.2%}")
    
    # Compare the two approaches
    print("\n📈 Comparison: Time-varying vs Constant Parameters")
    print("-" * 50)
    print(f"   Mean Final Price:")
    print(f"     Time-varying: ${summary_stats['mean_final_price']:,.2f}")
    print(f"     Constant:     ${summary_stats_const['mean_final_price']:,.2f}")
    print(f"     Difference:   ${summary_stats['mean_final_price'] - summary_stats_const['mean_final_price']:,.2f}")
    
    print(f"\n   Probability of Profit:")
    print(f"     Time-varying: {summary_stats['probability_profit']:.1%}")
    print(f"     Constant:     {summary_stats_const['probability_profit']:.1%}")
    print(f"     Difference:   {summary_stats['probability_profit'] - summary_stats_const['probability_profit']:.1%}")
    
    print(f"\n   Expected Return:")
    print(f"     Time-varying: {summary_stats['expected_return']:.2%}")
    print(f"     Constant:     {summary_stats_const['expected_return']:.2%}")
    print(f"     Difference:   {summary_stats['expected_return'] - summary_stats_const['expected_return']:.2%}")
    
    # Create enhanced plots
    print("\n📊 Creating enhanced visualization...")
    plot_enhanced_monte_carlo_results(
        simulation_results=simulation_results,
        summary_stats=summary_stats,
        initial_price=initial_price,
        predictions=predictions,
        save_path='results/enhanced_monte_carlo_example.png'
    )
    
    print("\n✅ Example completed!")
    print("\n💡 Usage with continuous predictor:")
    print("   1. Get 288 predictions from continuous_predictor.py")
    print("   2. Pass predictions to generate_monte_carlo_from_predictions()")
    print("   3. Use plot_enhanced_monte_carlo_results() for visualization")
    print("   4. Choose time_varying=True for more realistic simulations")


def generate_skewed_kurtotic_returns(size: Tuple[int, int], volatility: float,
                                   skewness: float, kurtosis: float, dt: float) -> np.ndarray:
    """
    Generate returns with specified volatility, skewness, and kurtosis using Cornish-Fisher expansion.
    """
    # Generate standard normal returns
    normal_returns = np.random.normal(0, 1, size)
    
    # Apply Cornish-Fisher transformation to match target moments
    cf_returns = normal_returns + (skewness / 6) * (normal_returns**2 - 1) + \
                 (kurtosis - 3) / 24 * (normal_returns**3 - 3 * normal_returns) - \
                 (skewness**2) / 36 * (2 * normal_returns**3 - 5 * normal_returns)
    
    # Scale by volatility and time step
    returns = cf_returns * volatility * np.sqrt(dt)
    
    return returns


def calculate_max_drawdown(price_paths: np.ndarray) -> float:
    """
    Calculate maximum drawdown across all simulation paths.
    """
    max_drawdowns = []
    
    for path in price_paths:
        running_max = np.maximum.accumulate(path)
        drawdown = (path - running_max) / running_max
        max_drawdowns.append(drawdown.min())
    
    return np.mean(max_drawdowns)


def calculate_var_cvar(returns: np.ndarray, confidence_level: float = 0.05) -> Tuple[float, float]:
    """
    Calculate Value at Risk (VaR) and Conditional Value at Risk (CVaR).
    
    Args:
        returns: Array of returns
        confidence_level: Confidence level (e.g., 0.05 for 95% VaR)
        
    Returns:
        Tuple of (VaR, CVaR)
    """
    sorted_returns = np.sort(returns)
    index = int(confidence_level * len(sorted_returns))
    
    var = sorted_returns[index]
    cvar = sorted_returns[:index].mean()
    
    return var, cvar


def plot_monte_carlo_results(simulation_results: pd.DataFrame, summary_stats: Dict,
                           initial_price: float, save_path: str = None) -> None:
    """
    Create comprehensive plots for Monte Carlo simulation results.
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Price path evolution
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
    
    # Final price distribution
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
    
    # Returns distribution
    returns = simulation_results.pct_change().dropna()
    all_returns = returns.values.flatten()
    all_returns = all_returns[~np.isnan(all_returns)]
    
    axes[1, 0].hist(all_returns, bins=50, alpha=0.7, density=True, color='lightgreen')
    axes[1, 0].set_title('Returns Distribution')
    axes[1, 0].set_xlabel('Returns')
    axes[1, 0].set_ylabel('Density')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Add normal distribution overlay
    mu, sigma = stats.norm.fit(all_returns)
    x = np.linspace(all_returns.min(), all_returns.max(), 100)
    axes[1, 0].plot(x, stats.norm.pdf(x, mu, sigma), 'r-', linewidth=2, label='Normal Fit')
    axes[1, 0].legend()
    
    # Summary statistics table
    axes[1, 1].axis('tight')
    axes[1, 1].axis('off')
    
    stats_data = [
        ['Metric', 'Value'],
        ['Initial Price', f'${initial_price:,.2f}'],
        ['Mean Final Price', f'${summary_stats["mean_final_price"]:,.2f}'],
        ['Std Final Price', f'${summary_stats["std_final_price"]:,.2f}'],
        ['Min Final Price', f'${summary_stats["min_final_price"]:,.2f}'],
        ['Max Final Price', f'${summary_stats["max_final_price"]:,.2f}'],
        ['5th Percentile', f'${summary_stats["percentile_5"]:,.2f}'],
        ['95th Percentile', f'${summary_stats["percentile_95"]:,.2f}'],
        ['Probability of Profit', f'{summary_stats["probability_profit"]:.2%}'],
        ['Max Drawdown', f'{summary_stats["max_drawdown"]:.2%}'],
        ['Realized Volatility', f'{summary_stats["volatility_realized"]:.2%}']
    ]
    
    table = axes[1, 1].table(cellText=stats_data, cellLoc='left', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)
    
    # Style the header row
    for i in range(len(stats_data[0])):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    axes[1, 1].set_title('Simulation Summary')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Monte Carlo plots saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


def analyze_volatility_patterns(df: pd.DataFrame) -> Dict:
    """
    Analyze intraday and weekly volatility patterns in Bitcoin data.
    """
    df = df.copy()
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['return'] = df['close'].pct_change()
    df['hour'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    
    # Calculate hourly volatility patterns
    hourly_vol = df.groupby('hour')['return'].std().to_dict()
    
    # Calculate daily volatility patterns (0=Monday, 6=Sunday)
    daily_vol = df.groupby('day_of_week')['return'].std().to_dict()
    
    # US trading hours analysis
    us_hours = df[(df['hour'] >= 14) & (df['hour'] <= 21)]
    asian_hours = df[(df['hour'] >= 0) & (df['hour'] <= 8)]
    off_hours = df[~df.index.isin(us_hours.index) & ~df.index.isin(asian_hours.index)]
    
    patterns = {
        'hourly_volatility': hourly_vol,
        'daily_volatility': daily_vol,
        'us_hours_volatility': us_hours['return'].std() if len(us_hours) > 0 else 0,
        'asian_hours_volatility': asian_hours['return'].std() if len(asian_hours) > 0 else 0,
        'off_hours_volatility': off_hours['return'].std() if len(off_hours) > 0 else 0,
        'weekend_volatility': df[df['day_of_week'] >= 5]['return'].std() if len(df[df['day_of_week'] >= 5]) > 0 else 0,
        'weekday_volatility': df[df['day_of_week'] < 5]['return'].std() if len(df[df['day_of_week'] < 5]) > 0 else 0
    }
    
    return patterns


def create_feature_importance_plot(feature_names: List[str], importances: np.ndarray,
                                 save_path: str = None, top_n: int = 20) -> None:
    """
    Create feature importance plot.
    """
    # Sort features by importance
    indices = np.argsort(importances)[::-1][:top_n]
    
    plt.figure(figsize=(12, 8))
    plt.title(f'Top {top_n} Feature Importances')
    plt.barh(range(top_n), importances[indices])
    plt.yticks(range(top_n), [feature_names[i] for i in indices])
    plt.xlabel('Importance')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Feature importance plot saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


def format_prediction_output(prediction: Dict) -> str:
    """
    Format prediction results for pretty printing.
    """
    output = f"""
    ╭─────────────────────────────────────────────────────────────╮
    │                Bitcoin Volatility Prediction                │
    ╰─────────────────────────────────────────────────────────────╯
    
    📅 Timestamp: {prediction['timestamp']}
    💰 Current Price: ${prediction['current_price']:,.2f}
    
    📊 Predictions (24h horizon):
    ├─ Volatility: {prediction['predicted_volatility']:.4f}
    ├─ Skewness: {prediction['predicted_skewness']:.4f}
    └─ Kurtosis: {prediction['predicted_kurtosis']:.4f}
    
    📈 Risk Metrics:
    ├─ Annualized Volatility: {prediction['volatility_annualized']:.2%}
    ├─ Market Regime: {prediction['market_regime']}
    ├─ Risk Level: {prediction['risk_assessment']}
    └─ 95% Confidence Interval: ${prediction['confidence_interval_lower']:,.2f} - ${prediction['confidence_interval_upper']:,.2f}
    """
    
    return output


def save_prediction_to_csv(predictions: List[Dict], filepath: str) -> None:
    """
    Save predictions to CSV file.
    """
    df = pd.DataFrame(predictions)
    df.to_csv(filepath, index=False)
    print(f"Predictions saved to {filepath}")


def load_and_validate_data(csv_path: str) -> pd.DataFrame:
    """
    Load and validate Bitcoin price data from CSV.
    """
    try:
        df = pd.read_csv(csv_path)
        validation_results = validate_crypto_data(df)
        
        if not validation_results['is_valid']:
            raise ValueError(f"Data validation failed: {validation_results['errors']}")
        
        if validation_results['warnings']:
            print("Data validation warnings:")
            for warning in validation_results['warnings']:
                print(f"  - {warning}")
        
        print(f"Data loaded successfully: {validation_results['statistics']}")
        return df
        
    except Exception as e:
        raise Exception(f"Failed to load data: {str(e)}")


# Create directory structure
def create_project_directories():
    """Create necessary project directories."""
    import os
    
    directories = ['data', 'models', 'results', 'logs']
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"Created directory: {directory}")

def safe_torch_load(file_path: str, map_location=None):
    """
    Safely load PyTorch model checkpoints, handling PyTorch 2.6+ weights_only changes.
    
    Args:
        file_path: Path to the checkpoint file
        map_location: Device to load the model on
        
    Returns:
        Loaded checkpoint dictionary
    """
    import torch
    
    try:
        # First try with weights_only=False for trusted files
        return torch.load(file_path, map_location=map_location, weights_only=False)
    except Exception as e:
        if "weights_only" in str(e).lower() or "WeightsUnpickler" in str(e):
            try:
                # Fallback: Add safe globals for sklearn objects
                from sklearn.preprocessing import RobustScaler, StandardScaler
                torch.serialization.add_safe_globals([RobustScaler, StandardScaler])
                return torch.load(file_path, map_location=map_location)
            except ImportError:
                # If sklearn is not available, try loading weights only
                return torch.load(file_path, map_location=map_location, weights_only=True)
        else:
            raise e


def check_system_requirements():
    """
    Check system requirements and package versions.
    """
    print("🔍 Checking system requirements...")
    
    import sys
    print(f"Python version: {sys.version}")
    
    try:
        import torch
        print(f"PyTorch version: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA version: {torch.version.cuda}")
            print(f"GPU device: {torch.cuda.get_device_name(0)}")
    except ImportError:
        print("❌ PyTorch not installed")
    
    try:
        import pandas as pd
        print(f"Pandas version: {pd.__version__}")
    except ImportError:
        print("❌ Pandas not installed")
    
    try:
        import numpy as np
        print(f"NumPy version: {np.__version__}")
    except ImportError:
        print("❌ NumPy not installed")
    
    try:
        import sklearn
        print(f"Scikit-learn version: {sklearn.__version__}")
    except ImportError:
        print("❌ Scikit-learn not installed")
    
    try:
        import yfinance as yf
        print(f"yfinance available: ✅")
    except ImportError:
        print("❌ yfinance not installed")
    
    print("✅ System check completed!")


def analyze_training_data_quality(csv_path: str) -> Dict:
    """
    Analyze training data quality to identify potential issues with kurtosis and other targets.
    
    Args:
        csv_path: Path to the training data CSV file
        
    Returns:
        Dictionary with analysis results
    """
    print("🔍 Analyzing training data quality...")
    
    # Load data
    df = pd.read_csv(csv_path)
    
    # Check if target columns exist
    target_cols = ['target_volatility', 'target_skewness', 'target_kurtosis']
    missing_cols = [col for col in target_cols if col not in df.columns]
    
    if missing_cols:
        print(f"❌ Missing target columns: {missing_cols}")
        return {'error': f"Missing columns: {missing_cols}"}
    
    analysis = {}
    
    # Analyze each target
    for col in target_cols:
        values = df[col].dropna()
        
        analysis[col] = {
            'count': len(values),
            'mean': float(values.mean()),
            'std': float(values.std()),
            'min': float(values.min()),
            'max': float(values.max()),
            'q25': float(values.quantile(0.25)),
            'q50': float(values.quantile(0.50)),
            'q75': float(values.quantile(0.75)),
            'outliers_3std': len(values[abs(values - values.mean()) > 3 * values.std()]),
            'outliers_5std': len(values[abs(values - values.mean()) > 5 * values.std()])
        }
    
    # Print summary
    print("\n📊 Training Data Analysis Summary:")
    print("=" * 50)
    
    for col, stats in analysis.items():
        print(f"\n{col}:")
        print(f"  Count: {stats['count']:,}")
        print(f"  Range: {stats['min']:.4f} to {stats['max']:.4f}")
        print(f"  Mean: {stats['mean']:.4f} ± {stats['std']:.4f}")
        print(f"  Q25/Q50/Q75: {stats['q25']:.4f} / {stats['q50']:.4f} / {stats['q75']:.4f}")
        print(f"  Outliers (>3σ): {stats['outliers_3std']} ({stats['outliers_3std']/stats['count']*100:.1f}%)")
        print(f"  Outliers (>5σ): {stats['outliers_5std']} ({stats['outliers_5std']/stats['count']*100:.1f}%)")
    
    # Specific kurtosis analysis
    kurtosis_values = df['target_kurtosis'].dropna()
    print(f"\n🎯 Kurtosis Analysis:")
    print(f"  Excess kurtosis range: {kurtosis_values.min():.2f} to {kurtosis_values.max():.2f}")
    print(f"  Absolute kurtosis range: {kurtosis_values.min() + 3:.2f} to {kurtosis_values.max() + 3:.2f}")
    
    # Check for extreme values
    extreme_kurtosis = kurtosis_values[kurtosis_values > 20]
    if len(extreme_kurtosis) > 0:
        print(f"  ⚠️  Found {len(extreme_kurtosis)} extreme kurtosis values (>20)")
        print(f"     These will be capped during training")
    
    # Recommendations
    print(f"\n💡 Recommendations:")
    if analysis['target_kurtosis']['outliers_3std'] > analysis['target_kurtosis']['count'] * 0.01:
        print("  - High number of kurtosis outliers detected")
        print("  - The new bounds (3-30 absolute kurtosis) should help")
    
    if analysis['target_kurtosis']['std'] > 10:
        print("  - High kurtosis variance detected")
        print("  - Log transformation should stabilize training")
    
    return analysis


def validate_prediction_bounds(predictions: List[Dict]) -> Dict:
    """
    Validate that predictions are within reasonable bounds.
    
    Args:
        predictions: List of prediction dictionaries
        
    Returns:
        Dictionary with validation results
    """
    if not predictions:
        return {'error': 'No predictions provided'}
    
    validation = {
        'total_predictions': len(predictions),
        'volatility_valid': 0,
        'skewness_valid': 0,
        'kurtosis_valid': 0,
        'all_valid': 0,
        'issues': []
    }
    
    for i, pred in enumerate(predictions):
        vol = pred.get('predicted_volatility', 0)
        skew = pred.get('predicted_skewness', 0)
        kurt = pred.get('predicted_kurtosis', 0)
        
        # Check bounds
        vol_valid = 0.001 <= vol <= 0.1
        skew_valid = -2.0 <= skew <= 2.0
        kurt_valid = -1.0 <= kurt <= 10.0
        
        if vol_valid:
            validation['volatility_valid'] += 1
        if skew_valid:
            validation['skewness_valid'] += 1
        if kurt_valid:
            validation['kurtosis_valid'] += 1
        if vol_valid and skew_valid and kurt_valid:
            validation['all_valid'] += 1
        else:
            validation['issues'].append({
                'index': i,
                'volatility': vol,
                'skewness': skew,
                'kurtosis': kurt,
                'vol_valid': vol_valid,
                'skew_valid': skew_valid,
                'kurt_valid': kurt_valid
            })
    
    # Calculate percentages
    validation['volatility_valid_pct'] = validation['volatility_valid'] / validation['total_predictions'] * 100
    validation['skewness_valid_pct'] = validation['skewness_valid'] / validation['total_predictions'] * 100
    validation['kurtosis_valid_pct'] = validation['kurtosis_valid'] / validation['total_predictions'] * 100
    validation['all_valid_pct'] = validation['all_valid'] / validation['total_predictions'] * 100
    
    return validation


if __name__ == "__main__":
    # Create project structure
    create_project_directories()
    
    # Run system check
    print("\n" + "="*50)
    check_system_requirements() 
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

def validate_bitcoin_data(df: pd.DataFrame) -> Dict[str, any]:
    """
    Validate Bitcoin price data for completeness and quality.
    
    Args:
        df: DataFrame with Bitcoin price data
        
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
        validation_results = validate_bitcoin_data(df)
        
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


if __name__ == "__main__":
    # Create project structure
    create_project_directories()
    
    # Run system check
    print("\n" + "="*50)
    check_system_requirements() 
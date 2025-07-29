# Enhanced Monte Carlo Simulator

## Overview

The Enhanced Monte Carlo Simulator is a sophisticated tool designed to work with the enhanced model's database predictions to generate realistic cryptocurrency price path simulations. It supports multiple simulation methods and provides comprehensive risk analysis.

## Features

- **Database Integration**: Direct integration with MongoDB to load real-time predictions
- **Multiple Simulation Methods**: 
  - Cornish-Fisher expansion (for non-normal distributions)
  - Normal distribution (for comparison)
  - Student's t-distribution (for fat tails)
  - Mixed approach (adaptive method selection)
- **Time-Varying Parameters**: Uses 288 predictions (24 hours × 5-minute intervals)
- **Comprehensive Risk Metrics**: VaR, CVaR, maximum drawdown, volatility clustering
- **Advanced Visualization**: 9-panel dashboard with detailed analysis
- **Multiple Cryptocurrencies**: Supports BTC, ETH, XAU, SOL

## Installation

The simulator is part of the enhanced model package. Ensure you have the required dependencies:

```bash
pip install numpy pandas matplotlib seaborn scipy pymongo python-dotenv
```

## Quick Start

### Basic Usage with Database

```python
from enhanced_monte_carlo_simulator import EnhancedMonteCarloSimulator

# Initialize simulator
simulator = EnhancedMonteCarloSimulator(
    crypto_symbol="BTC",
    method="cornish_fisher"
)

# Run simulation with database data
simulation_results, summary_stats = simulator.run_simulation_from_database(
    initial_price=117760.88,
    num_simulations=1000,
    hours_back=24,
    use_time_varying=True,
    save_results=True,
    show_plots=True
)
```

### Usage with File Data

```python
# Load predictions from JSON file
predictions = simulator.load_predictions_from_file("predictions.json")

# Run simulation
simulation_results, summary_stats = simulator.simulate_time_varying(
    predictions, initial_price=117760.88, num_simulations=1000
)
```

## Database Data Format

The simulator expects predictions in the following format (as stored by the enhanced model):

```json
{
  "_id": {"$oid": "6889185eaab7b63552cf53fc"},
  "prediction_timestamp": {"$date": "2025-07-29T18:52:14.994Z"},
  "data_timestamp": {"$date": "2025-07-29T18:51:58.269Z"},
  "model_version": "BTC_model",
  "batch_id": "continuous_1753815134",
  "prediction_type": "continuous_batch",
  "current_price": 117760.87883316,
  "predictions_count": 288,
  "interval_minutes": 5,
  "prediction_horizon_hours": 24,
  "source": "Pyth Network",
  "crypto_symbol": "BTC",
  "predictions": [
    {
      "sequence_number": 1,
      "timestamp": {"$date": "2025-07-29T18:57:11.194Z"},
      "minutes_ahead": 0,
      "predicted_volatility": 0.0010698665864765644,
      "predicted_skewness": -0.1543153077363968,
      "predicted_kurtosis": 4.762501239776611,
      "volatility_annualized": 0.3468744406723271,
      "confidence": 0.8,
      "prediction_horizon_minutes": 5
    }
    // ... 287 more predictions
  ]
}
```

## Simulation Methods

### 1. Cornish-Fisher Expansion
- **Best for**: Non-normal distributions with skewness and kurtosis
- **Use case**: When you have reliable predictions of higher moments
- **Formula**: Transforms normal distribution using skewness and kurtosis

### 2. Normal Distribution
- **Best for**: Comparison and baseline analysis
- **Use case**: When you only have volatility predictions
- **Formula**: Standard normal distribution scaled by volatility

### 3. Student's t-Distribution
- **Best for**: Fat-tailed distributions
- **Use case**: When kurtosis is high (>3)
- **Formula**: t-distribution with degrees of freedom calculated from kurtosis

### 4. Mixed Approach
- **Best for**: Adaptive method selection
- **Use case**: Automatic selection based on distribution characteristics
- **Logic**: 
  - Cornish-Fisher for moderate non-normality
  - Student's t for high kurtosis
  - Normal for near-normal distributions

## Output Analysis

### Summary Statistics

The simulator provides comprehensive statistics:

```python
summary_stats = {
    'simulation_method': 'cornish_fisher',
    'num_simulations': 1000,
    'initial_price': 117760.88,
    'final_price_stats': {
        'mean': 118234.56,
        'median': 118201.23,
        'std': 2345.67,
        'ci_95_lower': 113456.78,
        'ci_95_upper': 123012.34
    },
    'return_stats': {
        'mean': 0.0040,  # 0.4%
        'std': 0.0199,   # 1.99%
        'skewness': -0.123,
        'kurtosis': 4.567,
        'var_95': -0.0321,  # -3.21%
        'cvar_95': -0.0456  # -4.56%
    },
    'risk_metrics': {
        'max_drawdown': 0.0891,  # 8.91%
        'volatility_of_volatility': 0.0023,
        'mean_volatility': 0.0198
    }
}
```

### Visualization

The simulator creates a comprehensive 9-panel dashboard:

1. **Price Paths**: Multiple simulation paths with confidence intervals
2. **Final Price Distribution**: Histogram of final prices
3. **Return Distribution**: Histogram of returns
4. **Risk Metrics**: VaR, CVaR, and maximum drawdown
5. **Prediction Parameters**: Volatility, skewness, kurtosis over time
6. **Q-Q Plot**: Normality test
7. **Summary Table**: Key statistics
8. **Volatility Clustering**: Rolling volatility analysis
9. **Metadata**: Simulation parameters

## Configuration

### Environment Variables

Set these in your `.env` file:

```bash
# Database settings
MONGODB_URI=mongodb://localhost:27017/
DATABASE_NAME=synth_prediction_enhanced
ENABLE_DATABASE=true

# Simulation settings
ENABLE_ONLINE_LEARNING=true
RETRAIN_INTERVAL_HOURS=24
```

### Supported Cryptocurrencies

```python
SUPPORTED_CRYPTOS = {
    'BTC': {'name': 'Bitcoin', 'pyth_symbol': 'Crypto.BTC/USD'},
    'ETH': {'name': 'Ethereum', 'pyth_symbol': 'Crypto.ETH/USD'},
    'XAU': {'name': 'Gold', 'pyth_symbol': 'Metal.XAU/USD'},
    'SOL': {'name': 'Solana', 'pyth_symbol': 'Crypto.SOL/USD'}
}
```

## Examples

### Example 1: Basic Simulation

```python
from enhanced_monte_carlo_simulator import EnhancedMonteCarloSimulator

# Initialize
simulator = EnhancedMonteCarloSimulator("BTC", "cornish_fisher")

# Run simulation
results, stats = simulator.run_simulation_from_database(
    initial_price=117760.88,
    num_simulations=1000
)

# Print results
print(f"Mean Final Price: ${stats['final_price_stats']['mean']:,.2f}")
print(f"VaR 95%: {stats['return_stats']['var_95']:.2%}")
```

### Example 2: Compare Methods

```python
methods = ["normal", "cornish_fisher", "student_t", "mixed"]
results = {}

for method in methods:
    simulator = EnhancedMonteCarloSimulator("BTC", method)
    _, stats = simulator.run_simulation_from_database(
        initial_price=117760.88,
        num_simulations=500,
        save_results=False,
        show_plots=False
    )
    results[method] = stats

# Compare VaR across methods
for method, stats in results.items():
    var_95 = stats['return_stats']['var_95']
    print(f"{method}: VaR 95% = {var_95:.2%}")
```

### Example 3: Custom Analysis

```python
# Load predictions manually
predictions = simulator.load_predictions_from_database(hours_back=48)

# Run time-varying simulation
results, stats = simulator.simulate_time_varying(
    predictions, initial_price=117760.88, num_simulations=2000
)

# Custom analysis
final_prices = results.iloc[:, -1]
returns = (final_prices - 117760.88) / 117760.88

# Calculate custom metrics
custom_var = np.percentile(returns, 1)  # 99% VaR
print(f"Custom 99% VaR: {custom_var:.2%}")
```

## File Structure

```
enhanced_model/
├── enhanced_monte_carlo_simulator.py  # Main simulator
├── example_monte_carlo_usage.py       # Usage examples
├── MONTE_CARLO_README.md             # This file
├── database_manager.py               # Database operations
├── config.py                         # Configuration
└── results/                          # Output directory
    ├── BTC_simulation_20250101_120000.json
    └── BTC_simulation_20250101_120000.png
```

## Error Handling

The simulator includes comprehensive error handling:

```python
try:
    simulator = EnhancedMonteCarloSimulator("BTC", "cornish_fisher")
    results, stats = simulator.run_simulation_from_database(
        initial_price=117760.88
    )
except ValueError as e:
    print(f"Configuration error: {e}")
except ConnectionError as e:
    print(f"Database connection failed: {e}")
except Exception as e:
    print(f"Unexpected error: {e}")
```

## Performance Considerations

- **Memory Usage**: Large simulations (10,000+ paths) may require significant memory
- **Computation Time**: Time-varying simulations are slower than constant parameter
- **Database Load**: Loading predictions from database adds latency
- **Visualization**: Plotting many paths can be slow; use `show_plots=False` for batch processing

## Troubleshooting

### Common Issues

1. **Database Connection Failed**
   - Check MongoDB is running
   - Verify connection string in `.env`
   - Ensure database exists

2. **No Predictions Found**
   - Check if enhanced model has generated predictions
   - Verify `hours_back` parameter
   - Check database collection exists

3. **Memory Error**
   - Reduce `num_simulations`
   - Use constant parameters instead of time-varying
   - Close other applications

4. **Plotting Issues**
   - Install required plotting libraries
   - Use `show_plots=False` for headless environments
   - Check matplotlib backend

### Debug Mode

Enable debug output:

```python
import logging
logging.basicConfig(level=logging.DEBUG)

simulator = EnhancedMonteCarloSimulator("BTC", "cornish_fisher")
```

## Contributing

To extend the simulator:

1. Add new simulation methods to the class
2. Update the `method` validation in `__init__`
3. Add corresponding `generate_returns_*` method
4. Update documentation and examples

## License

This simulator is part of the enhanced model package and follows the same license terms. 
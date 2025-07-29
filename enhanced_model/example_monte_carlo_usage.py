#!/usr/bin/env python3
"""
Example usage of the Enhanced Monte Carlo Simulator

This script demonstrates how to use the enhanced Monte Carlo simulator
with the enhanced model's database predictions.
"""

import os
import sys
from pathlib import Path

# Add the enhanced_model directory to the path
sys.path.append(str(Path(__file__).parent))

from enhanced_monte_carlo_simulator import EnhancedMonteCarloSimulator
from database_manager import DatabaseManager
from config import EnhancedConfig

def example_with_database():
    """
    Example using real database data.
    """
    print("🚀 Enhanced Monte Carlo Simulator Example")
    print("=" * 50)
    
    # Initialize simulator with database connection
    try:
        simulator = EnhancedMonteCarloSimulator(
            crypto_symbol="BTC",
            method="cornish_fisher"  # Can be: cornish_fisher, normal, student_t, mixed
        )
        
        # Example price from your data
        current_price = 117760.88
        
        print(f"📊 Running simulation for BTC at ${current_price:,.2f}")
        print(f"🎯 Method: {simulator.method.upper()}")
        
        # Run simulation with database data
        simulation_results, summary_stats = simulator.run_simulation_from_database(
            initial_price=current_price,
            num_simulations=1000,  # Number of simulation paths
            hours_back=24,         # Load predictions from last 24 hours
            use_time_varying=True, # Use time-varying parameters
            save_results=True,     # Save results to files
            show_plots=True        # Display plots
        )
        
        # Print summary
        print("\n📈 Simulation Results Summary:")
        print("-" * 30)
        print(f"Mean Final Price: ${summary_stats['final_price_stats']['mean']:,.2f}")
        print(f"95% CI: ${summary_stats['final_price_stats']['ci_95_lower']:,.2f} - ${summary_stats['final_price_stats']['ci_95_upper']:,.2f}")
        print(f"Mean Return: {summary_stats['return_stats']['mean']:.2%}")
        print(f"Volatility: {summary_stats['return_stats']['std']:.2%}")
        print(f"Skewness: {summary_stats['return_stats']['skewness']:.3f}")
        print(f"Kurtosis: {summary_stats['return_stats']['kurtosis']:.3f}")
        print(f"VaR 95%: {summary_stats['return_stats']['var_95']:.2%}")
        print(f"CVaR 95%: {summary_stats['return_stats']['cvar_95']:.2%}")
        print(f"Max Drawdown: {summary_stats['risk_metrics']['max_drawdown']:.2%}")
        
        return simulation_results, summary_stats
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return None, None

def example_with_file_data():
    """
    Example using data from a JSON file (for testing without database).
    """
    print("\n📁 Example with File Data")
    print("=" * 30)
    
    # Create sample data structure similar to your database format
    sample_data = {
        "prediction_timestamp": "2025-07-29T18:52:14.994Z",
        "data_timestamp": "2025-07-29T18:51:58.269Z",
        "model_version": "BTC_model",
        "batch_id": "continuous_1753815134",
        "prediction_type": "continuous_batch",
        "current_price": 117760.87883316,
        "predictions_count": 288,
        "interval_minutes": 5,
        "prediction_horizon_hours": 24,
        "source": "Pyth Network",
        "crypto_symbol": "BTC",
        "predictions": []
    }
    
    # Generate sample predictions (288 predictions for 24 hours)
    import numpy as np
    from datetime import datetime, timedelta
    
    base_time = datetime.now()
    for i in range(288):
        # Generate realistic sample data
        volatility = 0.001 + 0.0005 * np.sin(i * 2 * np.pi / 288)  # Varying volatility
        skewness = -0.1 + 0.05 * np.sin(i * 2 * np.pi / 144)       # Varying skewness
        kurtosis = 4.0 + 2.0 * np.sin(i * 2 * np.pi / 96)          # Varying kurtosis
        
        prediction = {
            "sequence_number": i + 1,
            "timestamp": (base_time + timedelta(minutes=5*i)).isoformat(),
            "minutes_ahead": i * 5,
            "predicted_volatility": volatility,
            "predicted_skewness": skewness,
            "predicted_kurtosis": kurtosis,
            "volatility_annualized": volatility * np.sqrt(288 * 12),  # Annualized
            "confidence": 0.8,
            "prediction_horizon_minutes": (i + 1) * 5
        }
        sample_data["predictions"].append(prediction)
    
    # Save sample data to file
    import json
    sample_file = "sample_predictions.json"
    with open(sample_file, 'w') as f:
        json.dump(sample_data, f, indent=2)
    
    # Initialize simulator without database
    simulator = EnhancedMonteCarloSimulator(
        crypto_symbol="BTC",
        method="mixed",  # Try mixed method
        db_manager=None  # No database connection
    )
    
    # Load predictions from file
    predictions = simulator.load_predictions_from_file(sample_file)
    
    # Run simulation
    current_price = 117760.88
    simulation_results, summary_stats = simulator.simulate_time_varying(
        predictions, current_price, num_simulations=500
    )
    
    # Create plots
    simulator.plot_enhanced_results(
        simulation_results, summary_stats, current_price, predictions,
        save_path="sample_simulation.png", show_plots=True
    )
    
    # Clean up
    os.remove(sample_file)
    
    print("✅ Sample simulation completed!")
    return simulation_results, summary_stats

def compare_simulation_methods():
    """
    Compare different simulation methods.
    """
    print("\n🔄 Comparing Simulation Methods")
    print("=" * 35)
    
    # Initialize simulator
    simulator = EnhancedMonteCarloSimulator(
        crypto_symbol="BTC",
        method="cornish_fisher"
    )
    
    # Create sample predictions
    import numpy as np
    predictions = []
    for i in range(288):
        prediction = {
            "predicted_volatility": 0.0015,
            "predicted_skewness": -0.1,
            "predicted_kurtosis": 4.5
        }
        predictions.append(prediction)
    
    current_price = 117760.88
    methods = ["normal", "cornish_fisher", "student_t", "mixed"]
    
    results = {}
    for method in methods:
        print(f"📊 Running {method.upper()} simulation...")
        simulator.method = method
        simulation_results, summary_stats = simulator.simulate_time_varying(
            predictions, current_price, num_simulations=500
        )
        results[method] = summary_stats
    
    # Compare results
    print("\n📈 Method Comparison:")
    print("-" * 50)
    print(f"{'Method':<15} {'Mean Return':<12} {'Volatility':<12} {'VaR 95%':<12} {'Max DD':<12}")
    print("-" * 50)
    
    for method, stats in results.items():
        mean_return = stats['return_stats']['mean']
        volatility = stats['return_stats']['std']
        var_95 = stats['return_stats']['var_95']
        max_dd = stats['risk_metrics']['max_drawdown']
        
        print(f"{method.upper():<15} {mean_return:>10.2%} {volatility:>10.2%} {var_95:>10.2%} {max_dd:>10.2%}")

def main():
    """
    Main function to run examples.
    """
    print("🎯 Enhanced Monte Carlo Simulator Examples")
    print("=" * 50)
    
    # Check if database is available
    try:
        db_manager = DatabaseManager(crypto_symbol="BTC")
        print("✅ Database connection available")
        
        # Run database example
        example_with_database()
        
    except Exception as e:
        print(f"⚠️ Database not available: {str(e)}")
        print("📁 Running file-based examples instead...")
        
        # Run file-based examples
        example_with_file_data()
        compare_simulation_methods()
    
    print("\n🎉 Examples completed!")

if __name__ == "__main__":
    main() 
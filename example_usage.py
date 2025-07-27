#!/usr/bin/env python3
"""
Multi-Crypto Volatility Prediction - Complete Usage Example

This script demonstrates the complete workflow for multiple cryptocurrencies:
1. Data preparation and validation
2. Model training
3. Real-time prediction
4. Monte Carlo simulation
5. Results visualization

Supports: BTC, ETH, XAU, SOL

Author: AI Assistant
Date: 2024
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import yfinance as yf

# Import project modules
from config import Config
from data_processor import CryptoDataProcessor
from trainer import CryptoVolatilityTrainer
from predictor import RealTimeVolatilityPredictor
from utils import (
    validate_crypto_data, 
    monte_carlo_simulation, 
    plot_monte_carlo_results,
    format_prediction_output,
    create_project_directories,
    analyze_volatility_patterns
)

def download_sample_data(crypto_symbol: str = 'BTC', days: int = 60) -> str:
    """
    Download sample cryptocurrency data using yfinance.
    
    Args:
        crypto_symbol: Cryptocurrency symbol (BTC, ETH, XAU, SOL)
        days: Number of days of historical data to download
        
    Returns:
        Path to the saved CSV file
    """
    # Map crypto symbols to yfinance symbols
    yf_symbols = {
        'BTC': 'BTC-USD',
        'ETH': 'ETH-USD',
        'XAU': 'GC=F',  # Gold futures
        'SOL': 'SOL-USD'
    }
    
    if crypto_symbol not in yf_symbols:
        raise ValueError(f"Unsupported crypto symbol: {crypto_symbol}")
    
    yf_symbol = yf_symbols[crypto_symbol]
    crypto_name = Config.SUPPORTED_CRYPTOS[crypto_symbol]['name']
    
    print(f"📥 Downloading sample {crypto_name} data...")
    
    try:
        # Download cryptocurrency data
        data = yf.download(
            yf_symbol, 
            interval="5m", 
            period=f"{days}d",
            progress=False
        )
        
        if data.empty:
            raise ValueError("No data downloaded")
        
        # Reset index and prepare columns
        data.reset_index(inplace=True)
        
        # Handle different yfinance column structures
        if len(data.columns) == 7:
            data.columns = ['timestamp', 'open', 'high', 'low', 'close', 'adj_close', 'volume']
        elif len(data.columns) == 6:
            data.columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        else:
            # Use original column names and find the required ones
            expected_cols = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
            available_cols = [col for col in expected_cols if col in data.columns]
            
            # Create mapping for standardized names
            col_mapping = {
                'Open': 'open',
                'High': 'high', 
                'Low': 'low',
                'Close': 'close',
                'Adj Close': 'adj_close',
                'Volume': 'volume'
            }
            
            # Rename columns
            data = data.rename(columns=col_mapping)
            
            # Add timestamp column name if datetime index was reset
            if 'Datetime' in data.columns:
                data = data.rename(columns={'Datetime': 'timestamp'})
            elif data.columns[0] not in ['timestamp', 'open', 'high', 'low', 'close']:
                # First column is likely the timestamp
                data.columns = ['timestamp'] + list(data.columns[1:])
        
        # Select required columns (ensure they exist)
        required_cols = ['timestamp', 'open', 'close', 'high', 'low']
        available_required = [col for col in required_cols if col in data.columns]
        
        if len(available_required) < 5:
            raise ValueError(f"Missing required columns. Available: {list(data.columns)}, Required: {required_cols}")
        
        data = data[required_cols].copy()
        
        # Remove any NaN values
        data.dropna(inplace=True)
        
        # Save to CSV
        csv_path = f'data/{crypto_symbol}_price_data.csv'
        os.makedirs('data', exist_ok=True)
        data.to_csv(csv_path, index=False)
        
        print(f"✅ Downloaded {len(data)} records")
        print(f"📊 Date range: {data['timestamp'].min()} to {data['timestamp'].max()}")
        print(f"💾 Saved to: {csv_path}")
        
        return csv_path
        
    except Exception as e:
        print(f"❌ Failed to download data: {str(e)}")
        print("💡 Please ensure you have internet connection and yfinance is installed")
        return None

def validate_data_quality(csv_path: str, crypto_symbol: str = 'BTC') -> bool:
    """
    Validate the quality of cryptocurrency price data.
    
    Args:
        csv_path: Path to CSV file
        crypto_symbol: Cryptocurrency symbol (BTC, ETH, XAU, SOL)
        
    Returns:
        True if data is valid, False otherwise
    """
    crypto_name = Config.SUPPORTED_CRYPTOS[crypto_symbol]['name']
    print(f"\n🔍 Validating {crypto_name} data quality...")
    
    try:
        df = pd.read_csv(csv_path)
        validation_results = validate_crypto_data(df)
        
        if validation_results['is_valid']:
            print("✅ Data validation passed!")
            
            # Print statistics
            stats = validation_results['statistics']
            print(f"📊 Total records: {stats['total_rows']:,}")
            print(f"📅 Date range: {stats['date_range_start']} to {stats['date_range_end']}")
            print(f"💰 Price range: ${stats['price_range_min']:,.2f} - ${stats['price_range_max']:,.2f}")
            print(f"📈 Average price: ${stats['average_price']:,.2f}")
            print(f"📊 Price volatility: {stats['price_volatility']:.4f}")
            
            if validation_results['warnings']:
                print("\n⚠️ Warnings:")
                for warning in validation_results['warnings']:
                    print(f"  - {warning}")
            
            return True
        else:
            print("❌ Data validation failed!")
            for error in validation_results['errors']:
                print(f"  - {error}")
            return False
            
    except Exception as e:
        print(f"❌ Error validating data: {str(e)}")
        return False

def train_model(csv_path: str, crypto_symbol: str = 'BTC') -> bool:
    """
    Train the volatility prediction model.
    
    Args:
        csv_path: Path to training data
        crypto_symbol: Cryptocurrency symbol (BTC, ETH, XAU, SOL)
        
    Returns:
        True if training successful, False otherwise
    """
    crypto_name = Config.SUPPORTED_CRYPTOS[crypto_symbol]['name']
    print(f"\n🚀 Starting {crypto_name} model training...")
    
    try:
        config = Config()
        trainer = CryptoVolatilityTrainer(config, crypto_symbol)
        
        # Train the model
        training_history = trainer.train(csv_path)
        
        print("✅ Training completed successfully!")
        print(f"📉 Best validation loss: {min(training_history['val_losses']):.6f}")
        print(f"💾 Model saved to: {config.MODEL_SAVE_PATH}")
        print(f"📊 Training plots saved to: {config.RESULTS_PATH}")
        
        return True
        
    except Exception as e:
        print(f"❌ Training failed: {str(e)}")
        return False

def make_predictions(csv_path: str, crypto_symbol: str = 'BTC') -> dict:
    """
    Make volatility predictions using the trained model.
    
    Args:
        csv_path: Path to data for prediction
        crypto_symbol: Cryptocurrency symbol (BTC, ETH, XAU, SOL)
        
    Returns:
        Dictionary with prediction results
    """
    crypto_name = Config.SUPPORTED_CRYPTOS[crypto_symbol]['name']
    print(f"\n🔮 Making {crypto_name} volatility predictions...")
    
    try:
        # Load the predictor
        predictor = RealTimeVolatilityPredictor(crypto_symbol=crypto_symbol)
        
        if predictor.model is None:
            print("❌ No trained model found. Please train the model first.")
            return None
        
        # Load data
        data = pd.read_csv(csv_path)
        
        # Make prediction
        prediction = predictor.predict_next_period(data)
        
        # Print formatted results
        print("✅ Prediction completed!")
        print(format_prediction_output(prediction))
        
        # Generate intraday pattern
        print("📈 Generating intraday volatility pattern...")
        intraday_pattern = predictor.predict_intraday_pattern(data, intervals=96)  # Next 8 hours
        print(f"Generated {len(intraday_pattern)} intraday predictions")
        
        return prediction
        
    except Exception as e:
        print(f"❌ Prediction failed: {str(e)}")
        return None

def run_monte_carlo_simulation(prediction: dict, num_simulations: int = 1000) -> bool:
    """
    Run Monte Carlo simulation using predicted volatility parameters.
    
    Args:
        prediction: Prediction results from the model
        num_simulations: Number of simulation paths
        
    Returns:
        True if simulation successful, False otherwise
    """
    print(f"\n🎲 Running Monte Carlo simulation with {num_simulations:,} paths...")
    
    try:
        # Extract prediction parameters
        volatility = prediction['predicted_volatility']
        skewness = prediction['predicted_skewness']
        kurtosis = prediction['predicted_kurtosis']
        initial_price = prediction['current_price']
        
        print(f"Parameters: σ={volatility:.4f}, skew={skewness:.4f}, kurt={kurtosis:.4f}")
        
        # Run simulation
        simulation_results, summary_stats = monte_carlo_simulation(
            volatility=volatility,
            skewness=skewness,
            kurtosis=kurtosis,
            initial_price=initial_price,
            intervals=288,  # 24 hours
            num_simulations=num_simulations
        )
        
        # Display results
        print("✅ Monte Carlo simulation completed!")
        print(f"\n📊 Simulation Results:")
        print(f"  Initial Price: ${initial_price:,.2f}")
        print(f"  Expected Final Price: ${summary_stats['mean_final_price']:,.2f}")
        print(f"  Price Standard Deviation: ${summary_stats['std_final_price']:,.2f}")
        print(f"  Minimum Price: ${summary_stats['min_final_price']:,.2f}")
        print(f"  Maximum Price: ${summary_stats['max_final_price']:,.2f}")
        print(f"  95% Confidence Interval: ${summary_stats['percentile_5']:,.2f} - ${summary_stats['percentile_95']:,.2f}")
        print(f"  Probability of Profit: {summary_stats['probability_profit']:.2%}")
        print(f"  Maximum Drawdown: {summary_stats['max_drawdown']:.2%}")
        print(f"  Realized Volatility: {summary_stats['volatility_realized']:.2%}")
        
        # Create visualization
        save_path = 'results/monte_carlo_simulation.png'
        plot_monte_carlo_results(
            simulation_results, 
            summary_stats, 
            initial_price,
            save_path=save_path
        )
        
        print(f"📊 Simulation plots saved to: {save_path}")
        
        return True
        
    except Exception as e:
        print(f"❌ Monte Carlo simulation failed: {str(e)}")
        return False

def analyze_market_patterns(csv_path: str):
    """
    Analyze volatility patterns in the Bitcoin data.
    
    Args:
        csv_path: Path to Bitcoin price data
    """
    print("\n📊 Analyzing market patterns...")
    
    try:
        df = pd.read_csv(csv_path)
        patterns = analyze_volatility_patterns(df)
        
        print("✅ Pattern analysis completed!")
        print(f"\n🕐 Trading Hours Analysis:")
        print(f"  US Hours Volatility: {patterns['us_hours_volatility']:.4f}")
        print(f"  Asian Hours Volatility: {patterns['asian_hours_volatility']:.4f}")
        print(f"  Off Hours Volatility: {patterns['off_hours_volatility']:.4f}")
        
        print(f"\n📅 Weekly Patterns:")
        print(f"  Weekday Volatility: {patterns['weekday_volatility']:.4f}")
        print(f"  Weekend Volatility: {patterns['weekend_volatility']:.4f}")
        
        print(f"\n🕓 Peak Volatility Hours:")
        hourly_vol = patterns['hourly_volatility']
        if hourly_vol:
            sorted_hours = sorted(hourly_vol.items(), key=lambda x: x[1], reverse=True)
            for hour, vol in sorted_hours[:5]:
                print(f"  {hour:02d}:00 UTC - {vol:.4f}")
        
    except Exception as e:
        print(f"❌ Pattern analysis failed: {str(e)}")

def main():
    """
    Main function demonstrating the complete workflow for multiple cryptocurrencies.
    """
    print("🚀 Multi-Crypto Volatility Prediction - Complete Example")
    print("=" * 60)
    
    # Create project directories
    print("📁 Setting up project structure...")
    create_project_directories()
    
    # Get crypto symbol from command line or use default
    crypto_symbol = 'BTC'
    if len(sys.argv) > 1:
        crypto_symbol = sys.argv[1].upper()
        if crypto_symbol not in Config.SUPPORTED_CRYPTOS:
            print(f"❌ Unsupported crypto symbol: {crypto_symbol}")
            print(f"Supported: {', '.join(Config.SUPPORTED_CRYPTOS.keys())}")
            return
    
    crypto_name = Config.SUPPORTED_CRYPTOS[crypto_symbol]['name']
    print(f"📊 Running example for {crypto_name} ({crypto_symbol})")
    
    # Step 1: Download and validate data
    csv_path = f'data/{crypto_symbol.lower()}_price_data.csv'
    
    if not os.path.exists(csv_path):
        csv_path = download_sample_data(crypto_symbol=crypto_symbol, days=60)
        if csv_path is None:
            print(f"❌ Cannot proceed without data. Please provide {crypto_name} price data.")
            return
    
    if not validate_data_quality(csv_path, crypto_symbol=crypto_symbol):
        print("❌ Data quality issues detected. Please fix data before proceeding.")
        return
    
    # Analyze patterns in the data
    analyze_market_patterns(csv_path)
    
    # Step 2: Train the model (skip if model already exists)
    model_path = f'models/{crypto_symbol}_best_model.pth'
    if not os.path.exists(model_path):
        if not train_model(csv_path, crypto_symbol=crypto_symbol):
            print("❌ Cannot proceed without trained model.")
            return
    else:
        print(f"\n🎯 Using existing trained model: {model_path}")
    
    # Step 3: Make predictions
    prediction = make_predictions(csv_path, crypto_symbol=crypto_symbol)
    if prediction is None:
        print("❌ Cannot proceed without predictions.")
        return
    
    # Step 4: Run Monte Carlo simulation
    if not run_monte_carlo_simulation(prediction, num_simulations=1000):
        print("❌ Monte Carlo simulation failed.")
        return
    
    # Final summary
    print("\n" + "=" * 60)
    print("🎉 Complete workflow finished successfully!")
    print("📁 Check the following directories for results:")
    print("  - models/: Trained model files")
    print("  - results/: Training plots and simulation results")
    print("  - data/: Cryptocurrency price data")
    
    print(f"\n💡 Next steps:")
    print(f"  1. Use predictor.py for real-time predictions")
    print(f"  2. Use multi_crypto_orchestrator.py for multi-crypto predictions")
    print(f"  3. Integrate Monte Carlo simulation into your trading system")
    print(f"  4. Retrain model with new data as needed")
    print(f"  5. Adjust config.py parameters for your specific use case")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⏹️ Process interrupted by user")
    except Exception as e:
        print(f"\n❌ Unexpected error: {str(e)}")
        import traceback
        traceback.print_exc() 
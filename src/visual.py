#!/usr/bin/env python3

import csv
import sys
import os

from model import (
    estimate_price,
    load_theta,
    calculate_r_squared,
    calculate_mae,
    calculate_rmse
)

try:
    import matplotlib.pyplot as plt
except ImportError:
    print("Error: matplotlib is required for visualization.")
    print("Install it with: pip install matplotlib")
    sys.exit(1)


def load_data(filename='data.csv'):
    mileages = []
    prices = []
    
    try:
        with open(filename, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                mileages.append(float(row['km']))
                prices.append(float(row['price']))
        return mileages, prices
    except FileNotFoundError:
        print(f"Error: File '{filename}' not found.")
        sys.exit(1)
    except Exception as e:
        print(f"Error loading data: {e}")
        sys.exit(1)


def plot_regression(mileages, prices, theta0=None, theta1=None):
    plt.figure(figsize=(10, 6))
    
    # Plot data points
    plt.scatter(mileages, prices, color='blue', alpha=0.6, s=50, label='Training data')
    
    # Plot regression line if theta values are available
    if theta0 is not None and theta1 is not None:
        # Generate points for the regression line
        min_mileage = min(mileages)
        max_mileage = max(mileages)
        line_x = [min_mileage, max_mileage]
        line_y = [estimate_price(x, theta0, theta1) for x in line_x]
        
        plt.plot(line_x, line_y, color='red', linewidth=2, label='Linear regression')
        
        # Add equation to plot
        equation = f'Price = {theta0:.2f} + ({theta1:.4f} * Mileage)'
        plt.text(0.05, 0.95, equation, transform=plt.gca().transAxes,
                fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.xlabel('Mileage (km)', fontsize=12)
    plt.ylabel('Price ($)', fontsize=12)
    plt.title('Car Price vs Mileage - Linear Regression', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Format axes
    plt.ticklabel_format(style='plain', axis='both')
    
    plt.tight_layout()
    
    # Save plot
    plt.savefig('regression_plot.png', dpi=150)
    print("Plot saved as 'regression_plot.png'")
    
    # Show plot
    plt.show()


def display_metrics(mileages, prices, theta0, theta1):
    # Calculate metrics using shared functions
    r_squared = calculate_r_squared(mileages, prices, theta0, theta1)
    mae = calculate_mae(mileages, prices, theta0, theta1)
    rmse = calculate_rmse(mileages, prices, theta0, theta1)
    
    # Display metrics
    print("\n" + "="*50)
    print("MODEL PERFORMANCE METRICS")
    print("="*50)
    print(f"Mean Absolute Error (MAE):  ${mae:,.2f}")
    print(f"Root Mean Squared Error:    ${rmse:,.2f}")
    print(f"R² Score:                   {r_squared:.4f} ({r_squared * 100:.2f}%)")
    print("="*50)
    
    # Interpretation
    print("\nInterpretation:")
    if r_squared > 0.9:
        print("  Excellent fit! The model explains the data very well.")
    elif r_squared > 0.7:
        print("  Good fit. The model captures most of the trend.")
    elif r_squared > 0.5:
        print("  Moderate fit. The model captures some of the trend.")
    else:
        print("  Poor fit. The model may not be appropriate for this data.")


def main():
    print("Loading data and model parameters...")
    
    # Load training data
    mileages, prices = load_data('data.csv')
    print(f"Loaded {len(mileages)} data points")
    
    # Load theta parameters
    theta0, theta1 = load_theta()
    
    if theta0 is None or theta1 is None:
        print("\nWarning: Model not trained yet.")
        print("Plotting data only. Run 'python src/learn.py' or 'make train' first.\n")
        plot_regression(mileages, prices)
    else:
        print(f"Model parameters: theta0 = {theta0:.6f}, theta1 = {theta1:.6f}\n")
        
        # Calculate and display metrics
        display_metrics(mileages, prices, theta0, theta1)
        
        # Plot data and regression line
        print("\nGenerating plot...")
        plot_regression(mileages, prices, theta0, theta1)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

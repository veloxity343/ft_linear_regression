#!/usr/bin/env python3

import csv
import sys
import os

from model import (
    estimate_price,
    save_theta,
    normalise_data,
    denormalise_theta,
    calculate_r_squared
)


def load_data(filename='data.csv'):
    mileages = []
    prices = []
    
    try:
        with open(filename, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                mileages.append(float(row['km']))
                prices.append(float(row['price']))
        
        if len(mileages) == 0:
            raise ValueError("No data found in file")
        
        return mileages, prices
    
    except FileNotFoundError:
        print(f"Error: File '{filename}' not found.")
        sys.exit(1)
    except KeyError as e:
        print(f"Error: CSV must have 'km' and 'price' columns. Missing: {e}")
        sys.exit(1)
    except ValueError as e:
        print(f"Error: Invalid data format - {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Error loading data: {e}")
        sys.exit(1)


def train(mileages, prices, learning_rate=0.1, iterations=1000, verbose=True):
    # Normalise data
    norm_mileages, min_mile, range_mile = normalise_data(mileages)
    norm_prices, min_price, range_price = normalise_data(prices)
    
    m = len(mileages)
    theta0 = 0.0
    theta1 = 0.0
    
    if verbose:
        print(f"Training on {m} examples...")
        print(f"Learning rate: {learning_rate}, Iterations: {iterations}\n")
    
    # Track cost
    prev_cost = float('inf')
    
    for iteration in range(iterations):
        # Calculate predictions and errors
        predictions = [estimate_price(x, theta0, theta1) for x in norm_mileages]
        errors = [pred - actual for pred, actual in zip(predictions, norm_prices)]
        
        # Gradient descent formulae
        tmp_theta0 = learning_rate * (1 / m) * sum(errors)
        tmp_theta1 = learning_rate * (1 / m) * sum(error * x for error, x in zip(errors, norm_mileages))
        
        # update theta0 and theta1
        theta0 -= tmp_theta0
        theta1 -= tmp_theta1
        
        # every 100 iterations
        if verbose and ((iteration + 1) % 100 == 0 or iteration == 0):
            cost = (1 / (2 * m)) * sum(e ** 2 for e in errors)
            print(f"Iteration {iteration + 1:4d}: Cost = {cost:.6f}")
            
            if abs(prev_cost - cost) < 1e-9:
                print(f"\nConverged at iteration {iteration + 1}")
                break
            prev_cost = cost
    
    # Denormalise theta values
    theta0_original, theta1_original = denormalise_theta(
        theta0, theta1, min_mile, range_mile, min_price, range_price
    )
    
    return theta0_original, theta1_original


def main():
    # Load training data
    print("Loading training data...")
    mileages, prices = load_data('data.csv')
    print(f"Loaded {len(mileages)} data points\n")
    
    # Train the model
    theta0, theta1 = train(mileages, prices, learning_rate=0.1, iterations=1000)
    
    # Save parameters
    save_theta(theta0, theta1)
    
    # Display results
    print(f"\nFinal model parameters:")
    print(f"  theta0 (intercept) = {theta0:.6f}")
    print(f"  theta1 (slope)     = {theta1:.6f}")
    
    # Calculate and display model accuracy
    r_squared = calculate_r_squared(mileages, prices, theta0, theta1)
    print(f"\nModel R² score: {r_squared:.4f} ({r_squared * 100:.2f}%)")
    
    print("\nTraining complete! You can now use 'python predict.py' to make predictions.")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3

import sys

from model import (
    load_data,
    estimate_price,
    save_theta,
    normalise_data,
    denormalise_theta,
    calculate_r_squared
)


def train(mileages, prices, learning_rate=0.1, iterations=1000, verbose=True):
    """
    @brief Train linear regression model with gradient descent.
    @details Normalises mileage and price data, iteratively updates theta values,
    and returns denormalised parameters in the original units.
    """
    norm_mileages, min_mile, range_mile = normalise_data(mileages)
    norm_prices, min_price, range_price = normalise_data(prices)
    
    m = len(mileages)
    theta0 = 0.0
    theta1 = 0.0
    
    if verbose:
        print(f"Training on {m} examples...")
        print(f"Learning rate: {learning_rate}, Iterations: {iterations}\n")
    
    # track cost
    prev_cost = float('inf')
    
    for iteration in range(iterations):
        # calculate predictions and errors
        predictions = [estimate_price(x, theta0, theta1) for x in norm_mileages]
        errors = [pred - actual for pred, actual in zip(predictions, norm_prices)]
        
        # gradient descent formulae
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
    
    # denormalise theta values
    theta0_original, theta1_original = denormalise_theta(
        theta0, theta1, min_mile, range_mile, min_price, range_price
    )
    
    return theta0_original, theta1_original


def main():
    """
    @brief Run the training workflow from the command line.
    @details Loads dataset values, trains the model, saves learned parameters,
    and prints final coefficients with the R-squared score.
    """
    # load training data
    print("Loading training data...")
    mileages, prices = load_data('data.csv')
    print(f"Loaded {len(mileages)} data points\n")
    
    # train the model
    theta0, theta1 = train(mileages, prices, learning_rate=0.1, iterations=1000)
    
    # save parameters
    save_theta(theta0, theta1)
    
    # display results
    print(f"\nFinal model parameters:")
    print(f"  theta0 (intercept) = {theta0:.6f}")
    print(f"  theta1 (slope)     = {theta1:.6f}")
    
    # calculate and display model accuracy
    r_squared = calculate_r_squared(mileages, prices, theta0, theta1)
    print(f"\nModel R² score: {r_squared:.4f} ({r_squared * 100:.2f}%)")
    
    print("\nTraining complete! You can now use 'python predict.py' to make predictions.")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

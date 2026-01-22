#!/usr/bin/env python3

import os


def estimate_price(mileage, theta0, theta1):
    return theta0 + (theta1 * mileage)


def load_theta(filename='theta.txt'):
    if not os.path.exists(filename):
        return None, None
    
    try:
        with open(filename, 'r') as f:
            lines = f.readlines()
            theta0 = float(lines[0].strip())
            theta1 = float(lines[1].strip())
            return theta0, theta1
    except (ValueError, IndexError) as e:
        print(f"Error: Invalid theta file format - {e}")
        return None, None
    except Exception as e:
        print(f"Error loading theta parameters: {e}")
        return None, None


def save_theta(theta0, theta1, filename='theta.txt'):
    try:
        with open(filename, 'w') as f:
            f.write(f"{theta0}\n")
            f.write(f"{theta1}\n")
        print(f"\nModel parameters saved to {filename}")
    except Exception as e:
        print(f"Error saving parameters: {e}")
        raise


def normalise_data(data):
    min_val = min(data)
    max_val = max(data)
    range_val = max_val - min_val
    
    # if all values same
    if range_val == 0:
        return [0.0] * len(data), min_val, 1.0
    
    normalised = [(x - min_val) / range_val for x in data]
    return normalised, min_val, range_val


def denormalise_theta(theta0, theta1, min_mile, range_mile, min_price, range_price):
    theta1_original = (theta1 * range_price) / range_mile
    theta0_original = theta0 * range_price + min_price - theta1_original * min_mile
    
    return theta0_original, theta1_original


def calculate_cost(mileages, prices, theta0, theta1):
    m = len(mileages)
    predictions = [estimate_price(x, theta0, theta1) for x in mileages]
    errors = [pred - actual for pred, actual in zip(predictions, prices)]
    cost = sum(e ** 2 for e in errors) / (2 * m)
    return cost


def calculate_r_squared(mileages, prices, theta0, theta1):
    predictions = [estimate_price(x, theta0, theta1) for x in mileages]
    
    # Calculate mean of actual prices
    mean_price = sum(prices) / len(prices)
    
    # Total sum of squares (total variance)
    ss_tot = sum((y - mean_price) ** 2 for y in prices)
    
    # Residual sum of squares (unexplained variance)
    ss_res = sum((y - pred) ** 2 for y, pred in zip(prices, predictions))
    
    # R² score
    r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
    
    return r_squared


def calculate_mae(mileages, prices, theta0, theta1):
    predictions = [estimate_price(x, theta0, theta1) for x in mileages]
    mae = sum(abs(y - pred) for y, pred in zip(prices, predictions)) / len(prices)
    return mae


def calculate_rmse(mileages, prices, theta0, theta1):
    predictions = [estimate_price(x, theta0, theta1) for x in mileages]
    mse = sum((y - pred) ** 2 for y, pred in zip(prices, predictions)) / len(prices)
    rmse = mse ** 0.5
    return rmse

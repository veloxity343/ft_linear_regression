#!/usr/bin/env python3

import sys

from model import estimate_price, load_theta


def main():
    # Load trained parameters
    theta0, theta1 = load_theta()
    
    if theta0 is None or theta1 is None:
        print("Warning: Model not trained yet. Using default parameters (theta0=0, theta1=0)")
        print("Run 'python src/learn.py' or 'make train' first to train the model.\n")
        theta0, theta1 = 0.0, 0.0
    
    # Get mileage
    try:
        mileage_input = input("Enter the mileage (in km): ")
        mileage = float(mileage_input)
        
        if mileage < 0:
            print("Error: Mileage cannot be negative.")
            return 1
        
        # Calculate prediction
        price = estimate_price(mileage, theta0, theta1)
        
        # Display result
        print(f"\nEstimated price for {mileage:,.0f} km: ${price:,.2f}")
        
        if theta0 != 0.0 or theta1 != 0.0:
            print(f"\nModel parameters: theta0 = {theta0:.6f}, theta1 = {theta1:.6f}")
        
        return 0
        
    except ValueError:
        print("Error: Please enter a valid number.")
        return 1
    except KeyboardInterrupt:
        print("\n\nProgram interrupted.")
        return 1
    except Exception as e:
        print(f"Error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())

*This project has been created as part of the 42 curriculum by [rcheong](https://profile.intra.42.fr/users/rcheong)*

# ft_linear_regression

An implementation of linear regression from scratch using gradient descent to predict car prices based on mileage.

## Project Overview

This project implements a machine learning algorithm that predicts car prices using a simple linear regression model trained with gradient descent. The implementation features a **modular architecture** with shared functions and automated workflows via Makefile.

## Table of Contents

1. [Project Structure](#project-structure)
2. [Installation](#installation)
3. [Usage](#usage)
4. [Algorithm Details](#algorithm-details)
5. [Model Performance](#model-performance)
6. [Dataset Format](#dataset-format)
7. [Technical Implementation](#technical-implementation)

## Project Structure

### Core Files

1. **data.csv** - Sample dataset
   - Contains car mileage and price pairs
   - Format: `km,price`

2. **src/model.py** - Shared functions module
   - `estimate_price()` - Hypothesis function
   - `load_theta()` / `save_theta()` - Parameter I/O
   - `normalise_data()` / `denormalise_theta()` - Feature scaling
   - `calculate_r_squared()`, `calculate_mae()`, `calculate_rmse()` - Metrics
   - **No code duplication** - all programmes import from here

### Mandatory Programmes

3. **src/learn.py** - Training programme
   - Reads dataset from `data.csv`
   - Performs linear regression using gradient descent
   - Saves trained parameters (theta0, theta1) to `theta.txt`
   - Implements the gradient descent formulas:
     - `tmpTheta0 = learningRate * (1/m) * Σ(estimatePrice(mileage[i]) - price[i])`
     - `tmpTheta1 = learningRate * (1/m) * Σ((estimatePrice(mileage[i]) - price[i]) * mileage[i])`

4. **src/predict.py** - Prediction programme
   - Prompts user for a mileage value
   - Returns estimated price using the trained model
   - Uses the hypothesis: `estimatePrice(mileage) = theta0 + (theta1 * mileage)`

### Bonus

5. **src/visual.py** - Visualisation and metrics programme
   - Plots training data as scatter plot
   - Displays regression line
   - Calculates and displays performance metrics (R², MAE, RMSE)
   - Saves plot as `regression_plot.png`
   - Requires matplotlib

## Installation

### Requirements

- Python 3.6+
- matplotlib (for visualisation only)

### Quick Setup

This project uses a [Makefile](./Makefile) as the main task runner.

```bash
# Clone or download the project
cd ft_linear_regression

# Create virtual environment and install dependencies
make venv

# Activate virtual environment
source .venv/bin/activate

# Run everything
make
```

### Manual Setup

```bash
# Without virtual environment
pip install matplotlib

# Or with virtual environment
python3 -m venv .venv
source .venv/bin/activate
pip install matplotlib
```

## Usage

### Makefile Commands

| Command | Description |
|---------|-------------|
| `make venv` | Create virtual environment + install deps |
| `make` or `make run` | Train model + generate visualisation |
| `make train` | Train the model only |
| `make predict` | Make predictions |
| `make visual` | Generate charting and metrics |
| `make workflow` | Train + test with sample mileages |
| `make test` | Complete test suite |
| `make clean` | Remove generated files |
| `make fclean` | Remove all files including venv and cache |
| `make re` | Clean and retrain from scratch |

### Workflow

#### 1. Train the Model

Train model using dataset:

```bash
python src/learn.py
```

This will:
- Load data from `data.csv`
- Train the linear regression model
- Display training progress and convergence
- Save model parameters to `theta.txt`
- Show the final R² score

Example output:
```
Loading training data...
Loaded 24 data points

Training on 24 examples...
Learning rate: 0.1, Iterations: 1000

Iteration    1: Cost = 0.205790
Iteration  100: Cost = 0.025324
...
Iteration 1000: Cost = 0.010352

Model parameters saved to theta.txt

Final model parameters:
  theta0 (intercept) = 8481.172797
  theta1 (slope)     = -0.021274

Model R² score: 0.7329 (73.29%)
```

#### 2. Make Predictions

Use prediction program:

```bash
python src/predict.py
```

Example:
```
Enter the mileage (in km): 50000

Estimated price for 50,000 km: $7,417.49

Model parameters: theta0 = 8481.172797, theta1 = -0.021274
```

#### 3. Visualise Results

View the regression line and model performance:

```bash
python src/visual.py
```

This will:
- Display performance metrics (MAE, RMSE, R²)
- Generate a plot showing data points and regression line
- Save the plot as `regression_plot.png`

Example output:
```
MODEL PERFORMANCE METRICS
==================================================
Mean Absolute Error (MAE):  $XXX.XX
Root Mean Squared Error:    $XXX.XX
R² Score:                   0.7329 (73.29%)
==================================================

Interpretation:
  Good fit. The model captures most of the trend.

Plot saved as 'regression_plot.png'
```

## Algorithm Details

### Linear Regression

The model learns a linear relationship between mileage (x) and price (y):

```
price = theta0 + theta1 × mileage
```

Where:
- **theta0** is the y-intercept
- **theta1** is the gradient

### Gradient Descent

The training algorithm uses gradient descent to minimise the cost function by iteratively updating the parameters:

1. Calculate predictions for all training examples
2. Compute the error between predictions and actual values
3. Update parameters using the gradient descent formulas (from project specification)
4. Simultaneously update both theta0 and theta1
5. Repeat until convergence

### Data Normalisation

The implementation uses feature scaling to normalise the data to [0, 1] range, which:
- Improves convergence speed significantly
- Prevents numerical instability
- Ensures gradients are well-balanced

The final parameters are denormalised back to the original scale for predictions.

## Model Performance

The model achieves an R² score of approximately 73-94% depending on the dataset, indicating:
- The model explains the majority of variance in car prices
- Strong correlation between mileage and price
- Good predictive performance

### Performance Metrics

- **R² Score**: Measures how well the model explains the data (0-1, higher is better)
- **MAE (Mean Absolute Error)**: Average prediction error in dollars
- **RMSE (Root Mean Squared Error)**: Like MAE but penalises large errors more

## Dataset Format

The dataset should be a CSV file with the following format:

```csv
km,price
240000,3650
139800,3800
150500,4400
...
```

Where:
- `km` is the mileage in standardised unit of distance
- `price` is the car price in standardised currency

## Technical Implementation

### Algorithm Parameters

- **Learning rate**: 0.1 (configurable)
- **Iterations**: 1000 (configurable)
- **Convergence threshold**: 1e-9 (difference in cost between iterations)

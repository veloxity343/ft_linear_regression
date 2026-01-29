*This project has been created as part of the 42 curriculum by [rcheong](https://profile.intra.42.fr/users/rcheong)*

# ft_linear_regression

An implementation of linear regression from scratch using gradient descent to predict car prices based on mileage.

## Project Overview

This project implements univariate linear regression using batch gradient descent on a sample dataset. The model learns parameters θ₀ (intercept) and θ₁ (slope) by iteratively minimising mean squared error. Feature normalisation is applied to prevent gradient instability from feature scale disparity.

## Table of Contents

1. [Project Structure](#project-structure)
2. [The Problem](#understanding-the-problem)
3. [Mathematical Foundation](#mathematical-foundation)
4. [Implementation Details](#implementation-details)
5. [Installation & Usage](#installation--usage)
6. [Algorithm Parameters](#algorithm-parameters)
7. [Model Evaluation](#model-evaluation)
8. [Technical Notes](#technical-notes)
9. [References & Further Reading](#references--further-reading)

## Project Structure

### Components

1. **data.csv** - Sample dataset
   - Contains car mileage and price pairs
   - Format: `km,price`

2. **src/model.py** - Shared functions module
   - `estimate_price()` - Hypothesis function
   - `load_theta()` / `save_theta()` - Parameter I/O
   - `normalise_data()` / `denormalise_theta()` - Feature scaling
   - `calculate_r_squared()`, `calculate_mae()`, `calculate_rmse()` - Metrics
   - **No code duplication** - all programmes import from here

### Mandatory Programs

3. **src/learn.py** - Training program
   - Reads dataset from `data.csv`
   - Performs linear regression using gradient descent
   - Saves trained parameters (theta0, theta1) to `theta.txt`
   - Implements the gradient descent formulas:
     - `tmpTheta0 = learningRate * (1/m) * Σ(estimatePrice(mileage[i]) - price[i])`
     - `tmpTheta1 = learningRate * (1/m) * Σ((estimatePrice(mileage[i]) - price[i]) * mileage[i])`

4. **src/predict.py** - Prediction program
   - Prompts user for a mileage value
   - Returns estimated price using the trained model
   - Uses the hypothesis: `estimatePrice(mileage) = theta0 + (theta1 * mileage)`

### Bonus

5. **src/visual.py** - Visualisation and metrics program
   - Plots training data as scatter plot
   - Displays regression line
   - Calculates and displays performance metrics (R², MAE, RMSE)
   - Saves plot as `regression_plot.png`
   - Requires matplotlib

## Understanding the Problem

### Problem Statement

Given a dataset of car mileages and their corresponding prices, train a model to predict the price of a car given its mileage.

**Dataset Characteristics:**
- Input feature: Mileage (km) - continuous variable
- Target variable: Price ($) - continuous variable
- Samples: [data.csv](./data.csv)

### Linear Regression Model

A linear regression model assumes a linear relationship between input and output:

```
price = θ₀ + θ₁ × mileage
```

Where:
- **θ₀** (theta0) = y-intercept (bias term)
- **θ₁** (theta1) = gradient/slope (weight)

This equation defines a straight line in 2d space. The training objective is to find optimal values of θ₀ and θ₁ that best fit the observed data.

## Mathematical Foundation

### Hypothesis Function

The hypothesis is the line we are trying to fit; every prediction comes from this expression:

```
h_θ(x) = θ₀ + θ₁x
```

For our specific problem:
```
estimatePrice(mileage) = θ₀ + θ₁ × mileage
```

### Cost Function (Mean Squared Error)

To measure model performance, we use the Mean Squared Error (MSE) cost function:

```
J(θ₀, θ₁) = (1/2m) × Σᵢ₌₁ᵐ (h_θ(xᵢ) - yᵢ)²
```

Where:
- m = number of training examples
- xᵢ = mileage of i-th car
- yᵢ = actual price of i-th car
- h_θ(xᵢ) = predicted price

**Why square the errors?**
1. Makes all errors positive (avoids cancellation)
2. Penalises larger errors more heavily (quadratic penalty)
3. Produces a convex optimisation landscape (single global minimum)
4. Mathematically differentiable (enables gradient computation)

**Why divide by 2m instead of just m?**
The factor of 2 cancels with the derivative of the squared term, simplifying the gradient calculation. This is a mathematical convenience that doesn't affect the optimisation result.

### Gradient Descent Algorithm

Gradient descent iteratively adjusts parameters in the direction that reduces the cost function:

**Update rules:**
```
θ₀ := θ₀ - α × (1/m) × Σᵢ₌₁ᵐ (h_θ(xᵢ) - yᵢ)
θ₁ := θ₁ - α × (1/m) × Σᵢ₌₁ᵐ ((h_θ(xᵢ) - yᵢ) × xᵢ)
```

Where:
- α = learning rate (step size)
- := denotes simultaneous update

**Derivatives of gradients:**

For θ₀:
```
∂J/∂θ₀ = (1/m) × Σ(h_θ(x) - y)
```

For θ₁:
```
∂J/∂θ₁ = (1/m) × Σ((h_θ(x) - y) × x)
```

**Intuition:**
- Gradient points in direction of steepest ascent
- Negative gradient points toward minimum
- Learning rate α controls step size
- Process repeats until convergence (gradient ≈ 0)

### Feature Normalisation

**Problem:** Raw mileage values (~10,000 - ~250,000) much larger than price values (~3,500 - ~8,500), causing:
- Instability in gradient computation
- Slower convergence due to elongated error surface
- Potential overflow errors

**Solution:** Min-max scaling transforms features to [0, 1] range:

**Normalisation:**
```
x_norm = (x - x_min) / (x_max - x_min)
y_norm = (y - y_min) / (y_max - y_min)
```

**Denormalisation (reverting parameters):**
```
θ₁_original = θ₁_norm × (y_range / x_range)
θ₀_original = θ₀_norm × y_range + y_min - θ₁_original × x_min
```

Where:
- x_range = x_max - x_min
- y_range = y_max - y_min

**Why this works:**
Normalisation preserves the underlying relationship whilst equalising feature magnitudes. After training on scaled data, denormalisation transforms parameters back to the original scale for prediction.

## Implementation Details

### Training Algorithm

```
1. Load dataset from data.csv
2. Normalise features to [0, 1] range
3. Initialise θ₀ = 0, θ₁ = 0
4. For iteration = 1 to MAX_ITERATIONS:
   a. Compute predictions: ŷᵢ = θ₀ + θ₁xᵢ for all samples
   b. Calculate errors: eᵢ = ŷᵢ - yᵢ
   c. Compute gradients:
      ∂θ₀ = (1/m) × Σeᵢ
      ∂θ₁ = (1/m) × Σ(eᵢ × xᵢ)
   d. Update parameters:
      θ₀ := θ₀ - α × ∂θ₀
      θ₁ := θ₁ - α × ∂θ₁
   e. Calculate cost J(θ₀, θ₁)
   f. Check convergence: if |J_prev - J_current| < ε, break
5. Denormalise final θ₀ and θ₁
6. Save parameters to theta.txt
```

### Prediction Algorithm

```
1. Load θ₀ and θ₁ from theta.txt
2. Prompt user for mileage input
3. Validate input (non-negative number)
4. Compute: price = θ₀ + θ₁ × mileage
5. Display predicted price
```

### Simultaneous Parameter Updates

**Critical implementation detail:** Both parameters must be updated simultaneously using the old values.

**Incorrect (sequential):**
```python
theta0 = theta0 - learning_rate * gradient_theta0  # Updates theta0
theta1 = theta1 - learning_rate * gradient_theta1  # Uses NEW theta0 ✗
```

**Correct (simultaneous):**
```python
tmp_theta0 = learning_rate * gradient_theta0
tmp_theta1 = learning_rate * gradient_theta1
theta0 = theta0 - tmp_theta0  # Both use OLD values ✓
theta1 = theta1 - tmp_theta1
```

This ensures gradients are computed at a single point.

## Installation & Usage

### Requirements

- Python 3.6+
- matplotlib (for visualisation)

### Quick Start

This project uses a [Makefile](./Makefile) as the main task runner.

```bash
# Setup
make venv                     # Create virtual environment + install dependencies
source .venv/bin/activate     # Activate environment

# Usage
make                          # Train model + generate visualisation
make predict                  # Make predictions
make workflow                 # Train + test with sample values
```

### Makefile Commands

| Command | Action |
|---------|--------|
| `make venv` | Create virtual environment + install matplotlib |
| `make` / `make run` | Train model + generate visualisation |
| `make train` | Train model only |
| `make predict` | Interactive prediction (auto-trains if needed) |
| `make visual` | Generate visualisation and metrics |
| `make workflow` | Train + automated tests |
| `make test` | Complete test suite |
| `make clean` | Remove generated files |
| `make fclean` | Remove all files including venv |
| `make re` | Clean + retrain |

### Manual Execution

```bash
# Training
python src/learn.py

# Prediction
python src/predict.py

# Visualisation
python src/visual.py
```

## Algorithm Parameters

### Parameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Learning rate (α) | 0.1 | Balanced convergence speed vs. stability |
| Max iterations | 1000 | Sufficient for convergence on small dataset |
| Convergence threshold (ε) | 1e-9 | Detects when cost function plateaus |
| Initial θ₀ | 0.0 | Standard practice for linear regression |
| Initial θ₁ | 0.0 | Allows gradient descent to find optimal values |

### Training Output

```
Loading training data...
Loaded data points

Training on examples...
Learning rate: 0.1, Iterations: 1000

Iteration    1: Cost = 0.205790
Iteration  100: Cost = 0.025324
Iteration  200: Cost = 0.015884
...
Iteration 1000: Cost = 0.010352

Model parameters saved to theta.txt

Final model parameters:
  theta0 (intercept) = 8481.172797
  theta1 (slope)     = -0.021274

Model R² score: 0.7329 (73.29%)
```

**Interpretation:**
- Cost decreases monotonically → gradient descent working correctly
- θ₁ = -0.021274 → car loses ~$21.27 per 1000 km driven
- θ₀ = 8481.17 → theoretical price at 0 km (extrapolation)

## Model Evaluation

### Performance Metrics

**1. R² Score (coefficient of determination)**

```
R² = 1 - (SS_res / SS_tot)
```

Where:
- SS_res = Σ(yᵢ - ŷᵢ)² (residual sum of squares)
- SS_tot = Σ(yᵢ - ȳ)² (total sum of squares)

**Interpretation:**
- R² = 1.0: Perfect predictions
- R² = 0.73: Model explains 73% of price variance
- R² = 0.0: Model no better than predicting the mean

**2. Mean absolute error (MAE)**

```
MAE = (1/m) × Σ|yᵢ - ŷᵢ|
```

Average absolute deviation between predictions and actual values. Provides interpretable error in original units (dollars).

**3. Root mean squared error (RMSE)**

```
RMSE = √[(1/m) × Σ(yᵢ - ŷᵢ)²]
```

Penalises large errors more heavily than MAE. Useful for detecting outliers.

### Model Performance on Sample Dataset

```
MODEL PERFORMANCE METRICS
==================================================
Mean Absolute Error (MAE):  $XXX.XX
Root Mean Squared Error:    $XXX.XX
R² Score:                   0.7329 (73.29%)
==================================================

Interpretation:
  Relatively good fit. The model captures most of the trend.
```

### Example Predictions

| Mileage (km) | Predicted Price | Interpretation |
|--------------|----------------|----------------|
| 50,000 | $7,417.49 | Average usage vehicle |
| 100,000 | $6,353.09 | High mileage |
| 150,000 | $5,288.69 | Very high mileage |
| 200,000 | $4,224.29 | Extreme mileage |

**Price depreciation rate:** $21.27 per 1,000 km

## Technical Notes

### Convexity of Loss Function

For linear regression with MSE, the cost function is convex (bowl-shaped). This guarantees:
- Single global minimum (no local minima)
- Gradient descent will converge from any starting point
- No risk of getting stuck in suboptimal solutions

### Learning Rate Selection

The learning rate α = 0.1 was chosen empirically:
- α too large (>0.5): Risk of overshooting minimum, divergence
- α too small (<0.01): Slow convergence, many iterations required
- α = 0.1: Good balance for normalised data in [0,1] range

### Computational Complexity

- Training: O(m × n × iterations) where m = samples, n = features
- For this dataset: O(24 × 1 × 1000) ≈ 24,000 operations
- Prediction: O(1) constant time

### Extrapolation Warnings

Model predictions outside training range [22,899 - 240,000 km] are extrapolations. Linear assumption may not hold at extremes (e.g., brand new cars, completely depreciated vehicles).

## References & Further Reading

**Mathematical foundations:**
- Gradient descent optimisation
- Convex optimisation theory
- Feature scaling techniques
- Statistical regression analysis

**Machine learning concepts:**
- Supervised learning paradigm
- Overfitting vs. underfitting
- Bias-variance tradeoff
- Cross-validation methods

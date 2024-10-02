import numpy as np
import copy

# Predict function: calculates prediction for a given input x, weights w, and bias b
def predict(x, w, b):
    """
    Predicts the output for given input data, weights, and bias.
    
    Args:
        x : numpy array : Input features
        w : numpy array : Weights of the model
        b : float       : Bias term

    Returns:
        float : Predicted value (linear combination of x and w plus b)
    """
    return np.dot(x, w) + b


# Cost function: calculates mean squared error cost
def cost_fun(x, y, w, b):
    """
    Computes the mean squared error cost function.
    
    Args:
        x : numpy array : Input features (m examples, n features)
        y : numpy array : Target values
        w : numpy array : Weights of the model
        b : float       : Bias term

    Returns:
        float : The cost (mean squared error)
    """
    m = len(x)  # Number of training examples
    cost = 0  # Initialize cost
    f_x = np.zeros(m)  # Array to store predictions
    
    # Compute predictions and accumulate the squared error
    for i in range(m):
        f_x[i] = np.dot(x[i], w) + b  # Compute prediction
        cost += (f_x[i] - y[i])**2  # Accumulate squared error
    
    cost = cost / (2 * m)  # Average the squared errors to get mean squared error
    return cost


# Gradient calculation (partial derivatives of the cost function)
def dj_part(x, y, w, b):
    """
    Computes the gradient of the cost function with respect to weights and bias.
    
    Args:
        x : numpy array : Input features
        y : numpy array : Target values
        w : numpy array : Weights of the model
        b : float       : Bias term

    Returns:
        numpy array : Gradient with respect to weights
        float       : Gradient with respect to bias
    """
    m = len(x)  # Number of training examples
    f_x = np.zeros(m)  # Store predictions
    tw = np.zeros_like(w)  # Initialize gradient for weights
    tb = 0  # Initialize gradient for bias
    
    # Loop over all examples to compute gradients
    for i in range(m):
        f_x[i] = np.dot(x[i], w) + b  # Compute prediction for example i
        err = f_x[i] - y[i]  # Compute error
        tw += err * x[i]  # Accumulate weight gradients
        tb += err  # Accumulate bias gradient
    
    tw = (1 / m) * tw  # Average the gradient over all examples (for weights)
    tb = (1 / m) * tb  # Average the gradient over all examples (for bias)
    
    return tw, tb


# Gradient Descent function
def gd(x, y, w_in, b_in, alpha, iterations, dj_part):
    """
    Performs gradient descent to optimize weights and bias.
    
    Args:
        x : numpy array : Input features
        y : numpy array : Target values
        w_in : numpy array : Initial weights
        b_in : float       : Initial bias
        alpha : float      : Learning rate
        iterations : int   : Number of iterations
        dj_part : function : Function to compute gradients

    Returns:
        numpy array : Optimized weights
        float       : Optimized bias
    """
    w = copy.deepcopy(w_in)  # Copy the initial weights
    b = b_in  # Initialize bias
    
    # Perform the gradient descent loop for the specified number of iterations
    for i in range(iterations):
        dw, db = dj_part(x, y, w, b)  # Compute the gradients
        w = w - alpha * dw  # Update weights using gradient descent
        b = b - alpha * db  # Update bias using gradient descent
    
    return w, b


# Example usage (this is where you train the model)
x_train = np.array([[2104, 5, 1, 45], [1416, 3, 2, 40], [852, 2, 1, 35]])  # Training features
y_train = np.array([460, 232, 178])  # Training target values

# Initial weights and bias
w_init = np.array([0.39133535, 18.75376741, -53.36032453, -26.42131618])
b_init = 785.1811367994083

# Set hyperparameters
alpha = 5.0e-7  # Learning rate
iterations = 10000  # Number of iterations

# Initialize weights and bias
w_in = np.zeros_like(w_init)
b_in = 0.0

# Perform gradient descent
w, b = gd(x_train, y_train, w_in, b_in, alpha, iterations, dj_part)

# Print the final optimized weights and bias
print("Optimized weights:", w)
print("Optimized bias:", b)

# Make predictions with the optimized weights and bias
predictions = predict(x_train, w, b)
print("Predictions:", predictions)

import numpy as np
from MLP import MLP

# Create an instance of the MLP class
# Assuming 2 input neurons, a hidden layer with 4 neurons, and 1 output neuron
mlp = MLP(input_dim=2, hidden_dim=3, output_dim=1, lr=0.1)

# XOR input and output
X = np.array([[0, 0, 1, 1], [0, 1, 0, 1]]) # 2x4 matrix for inputs
Y = np.array([0, 1, 1, 0]) # 1x4 matrix for outputs

# Train the MLP with the XOR data
iterations = 1000 # Number of iterations for training
mlp.gradient_descent(X, Y, iterations, 0.1)

# Test the MLP with the XOR data
predictions = mlp.test_predictions(X)

# Print out the predictions to see if they match the XOR truth table
print("Predictions:", predictions)
print("Expected: [0, 1, 1, 0]")

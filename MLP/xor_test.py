import numpy as np
from nn import MultilayerPerceptron

def train_xor(mlp, epochs, learning_rate):
    # XOR input and output
    X = np.array([[0, 0],
                  [0, 1],
                  [1, 0],
                  [1, 1]])
    Y = np.array([[0],
                  [1],
                  [1],
                  [0]])

    # Train the network
    mlp.train(X, Y, epochs, learning_rate)

def test_xor(mlp):
    # Test data (same as training data for XOR)
    X_test = np.array([[0, 0],
                       [0, 1],
                       [1, 0],
                       [1, 1]])
    # Predictions
    predictions = mlp.predict(X_test)
    # Convert predictions to binary values
    predictions = np.round(predictions)
    return predictions

# Initialize the MLP
input_size = 2
hidden_size = 3  # Can experiment with this size
output_size = 1
mlp = MultilayerPerceptron(input_size, hidden_size, output_size)

# Train the MLP
train_xor(mlp, epochs=10000, learning_rate=0.1)

# Test the MLP
predictions = test_xor(mlp)
print("Predictions:")
print(predictions)
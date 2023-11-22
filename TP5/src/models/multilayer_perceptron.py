import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

class MultilayerPerceptron:
    def __init__(self, input_size, hidden_size, output_size):
        # Initialize weights and biases
        self.weights_input_hidden = np.random.rand(input_size, hidden_size)
        self.bias_hidden = np.random.rand(hidden_size)
        self.weights_hidden_output = np.random.rand(hidden_size, output_size)
        self.bias_output = np.random.rand(output_size)

    def forward_propagation(self, inputs):
        # Forward pass through the network
        self.hidden_layer_input = np.dot(inputs, self.weights_input_hidden) + self.bias_hidden
        self.hidden_layer_output = sigmoid(self.hidden_layer_input)
        self.output_layer_input = np.dot(self.hidden_layer_output, self.weights_hidden_output) + self.bias_output
        self.output_layer_output = sigmoid(self.output_layer_input)
        return self.output_layer_output

    def train(self, inputs, targets, epochs, learning_rate):
        for epoch in range(epochs):
            # Forward propagation
            outputs = self.forward_propagation(inputs)

            # Backward propagation
            output_errors = targets - outputs
            output_deltas = output_errors * sigmoid_derivative(outputs)
            hidden_errors = np.dot(output_deltas, self.weights_hidden_output.T)
            hidden_deltas = hidden_errors * sigmoid_derivative(self.hidden_layer_output)

            # Update weights and biases
            self.weights_hidden_output += learning_rate * np.dot(self.hidden_layer_output.T, output_deltas)
            self.bias_output += learning_rate * np.sum(output_deltas, axis=0)  # Removed keepdims=True

            self.weights_input_hidden += learning_rate * np.dot(inputs.T, hidden_deltas)
            self.bias_hidden += learning_rate * np.sum(hidden_deltas, axis=0)  # Removed keepdims=True



    def predict(self, inputs):
        return self.forward_propagation(inputs)

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

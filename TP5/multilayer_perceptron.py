import numpy as np

class MLP:
    def __init__(self, input_size, hidden_layers, output_size):
        # Initialization of parameters
        self.input_size = input_size
        self.hidden_layers = hidden_layers
        self.output_size = output_size

        # Initialization of weights and biases for all layers
        self.weights = []
        self.biases = []

        # Create a list of layer sizes (input, hidden layers, output)
        layer_sizes = [input_size] + hidden_layers + [output_size]

        # Initialize weights and biases for each layer
        for i in range(len(layer_sizes) - 1):
            weight_matrix = np.random.randn(layer_sizes[i], layer_sizes[i + 1])
            bias_vector = np.zeros((1, layer_sizes[i + 1]))
            self.weights.append(weight_matrix)
            self.biases.append(bias_vector)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def forward(self, x):
        # Forward propagation
        self.layer_outputs = [x]
        for i in range(len(self.weights)):
            x = self.sigmoid(np.dot(x, self.weights[i]) + self.biases[i])
            self.layer_outputs.append(x)
        return x

    def backward(self, x, y, learning_rate):
        # Backpropagation of error
        error = y - self.layer_outputs[-1]
        for i in reversed(range(len(self.weights))):
            delta = error * self.sigmoid_derivative(self.layer_outputs[i + 1])

            if i == 0:
                # If it's the first layer, use the input instead of the previous layer's output
                layer_output_reshaped = x.reshape(1, -1)
            else:
                # Reshape previous layer's output for correct matrix multiplication
                layer_output_reshaped = self.layer_outputs[i].reshape(1, -1)

            # Calculate the gradient
            weight_gradient = np.dot(layer_output_reshaped.T, delta)
            bias_gradient = delta

            # Update error for the next layer
            error = np.dot(delta, self.weights[i].T)

            # Update weights and biases
            self.weights[i] += weight_gradient * learning_rate
            self.biases[i] += np.sum(bias_gradient, axis=0, keepdims=True) * learning_rate



    def train(self, X, y, epochs, learning_rate):
        # Training the network
        for epoch in range(epochs):
            for x, target in zip(X, y):
                self.forward(x)
                self.backward(x, target, learning_rate)

    def predict(self, X):
        # Making predictions
        return np.array([self.forward(x) for x in X])

# Example usage
if __name__ == "__main__":
    # Training data
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([[0], [1], [1], [0]])

    # Create and train an MLP with two hidden layers of 3 and 2 neurons
    mlp = MLP(input_size=2, hidden_layers=[3, 2], output_size=1)
    mlp.train(X, y, epochs=10000, learning_rate=0.1)

    # Make predictions
    predictions = mlp.predict(X)
    print("Predictions:")
    print(predictions)

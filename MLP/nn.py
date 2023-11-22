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

    def backward_propagation(self, inputs, outputs, targets, learning_rate):
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

            # returns delta for the hidden layer for use un the autoencoder to backpropagate through the encoder
            return hidden_deltas


    def train(self, inputs, targets, epochs, learning_rate):
        for epoch in range(epochs):
            self.backward_propagation(inputs, self.forward_propagation(inputs), targets, learning_rate)

    def predict(self, inputs):
        return self.forward_propagation(inputs)

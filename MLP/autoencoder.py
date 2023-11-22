import numpy as np
from nn import MultilayerPerceptron

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

class Autoencoder:
    def __init__(self, input_size, hidden_size, latent_size, learning_rate=0.1):
        self.hidden_size = hidden_size
        self.latent_size = latent_size
        self.learning_rate = learning_rate

        # Initialize encoder
        self.encoder = MultilayerPerceptron(input_size, hidden_size, latent_size)
        # Initialize decoder
        self.decoder = MultilayerPerceptron(latent_size, hidden_size, input_size)

    def forward_propagation(self, inputs):
        # Forward pass through the network
        self.latent_layer_output = self.encoder.forward_propagation(inputs)
        self.output_layer_output = self.decoder.forward_propagation(self.latent_layer_output)
        return  self.latent_layer_output, self.output_layer_output

    def train(self, inputs, epochs):
        for epoch in range(epochs):
            # Forward propagation
            latent_layer_output, outputs = self.forward_propagation(inputs)

            # Backward propagation through decoder
            decoder_hidden_delta = self.decoder.backward_propagation(latent_layer_output, outputs, inputs, self.learning_rate)

            # Backward propagation through encoder
            output_errors = np.dot(decoder_hidden_delta, latent_layer_output.T)
            output_deltas = output_errors * sigmoid_derivative(latent_layer_output)
            hidden_errors = np.dot(output_deltas, self.weights_hidden_output.T)
            hidden_deltas = hidden_errors * sigmoid_derivative(self.hidden_layer_output)

            # Update weights and biases
            self.encoder.weights_hidden_output += self.learning_rate * np.dot(self.hidden_layer_output.T, output_deltas)
            self.encoder.bias_output += self.learning_rate * np.sum(output_deltas, axis=0)  # Removed keepdims=True




    def reconstruct(self, inputs):
        _, outputs = self.forward_propagation(inputs)
        return outputs
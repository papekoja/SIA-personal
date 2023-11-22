import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))

class Neuron:
    def __init__(self, activation, activation_deriv, input_dim, hidden=False):
        self.hidden = hidden
        self.activation_func = activation
        self.activation_deriv = activation_deriv
        self.weights = np.random.uniform(-1, 1, input_dim)
        self.input = np.zeros(input_dim)

    def forward(self, input_arr):
        self.input = input_arr
        return self.activation_func(np.dot(input_arr, self.weights))

    def calculate_delta(self, output, target, next_layer_weights=None, next_layer_deltas=None):
        if self.hidden:
            delta = np.dot(next_layer_weights, next_layer_deltas) * self.activation_deriv(np.dot(self.input, self.weights))
        else:
            delta = (target - output) * self.activation_deriv(np.dot(self.input, self.weights))
        return delta

    def update_weights(self, delta, lr):
        delta_w = lr * delta * self.input
        self.weights += delta_w
        return delta_w

class NeuralNetwork:
    def __init__(self, architecture, input_dim):
        self.layers = []
        for i, num_neurons in enumerate(architecture):
            layer = []
            dim = input_dim if i == 0 else architecture[i - 1]
            for _ in range(num_neurons):
                hidden = i < len(architecture) - 1
                layer.append(Neuron(sigmoid, sigmoid_derivative, dim, hidden))
            self.layers.append(layer)

    def forward_propagation(self, input_data):
        activation = input_data
        for layer in self.layers:
            next_activation = []
            for neuron in layer:
                next_activation.append(neuron.forward(activation))
            activation = np.array(next_activation)
        return activation

    def back_propagation(self, target, lr):
        layer_deltas = []
        for i in reversed(range(len(self.layers))):
            layer = self.layers[i]
            layer_delta = []
            for j, neuron in enumerate(layer):
                if i == len(self.layers) - 1:
                    delta = neuron.calculate_delta(neuron.forward(neuron.input), target)
                else:
                    next_layer_weights = np.array([neuron.weights for neuron in self.layers[i + 1]])
                    next_layer_deltas = np.array(layer_deltas[0])
                    delta = neuron.calculate_delta(neuron.forward(neuron.input), None, next_layer_weights[:, j], next_layer_deltas)
                neuron.update_weights(delta, lr)
                layer_delta.append(delta)
            layer_deltas.insert(0, layer_delta)

    def train(self, input_data, target, lr):
        output = self.forward_propagation(input_data)
        self.back_propagation(target, lr)
        return output

    def calculate_error(self, output, target):
        return np.sum(np.square(target - output)) / 2
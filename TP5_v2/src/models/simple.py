import numpy as np


class SimplePerceptron:
    def __init__(self, activation_fn, derivative_activation_fn, input_dimension: int,
                 is_hidden: bool = False, perceptron_index: int = 0,
                 use_momentum: bool = False, momentum_coefficient: float = 0.9):
        self.index = perceptron_index
        self.is_hidden = is_hidden
        self.activation_fn = activation_fn
        self.derivative_activation_fn = derivative_activation_fn
        self.weights = np.zeros(input_dimension)
        self.input_data = np.zeros(input_dimension)
        self.previous_delta_weights = np.zeros(input_dimension)
        self.use_momentum = use_momentum
        self.momentum_coefficient = momentum_coefficient
        self.weight_accumulator = np.zeros(input_dimension)

    def backpropagate(self, target_output: np.ndarray, upper_layer_weights: np.ndarray,
                      upper_layer_deltas: np.ndarray, learning_rate: float) -> (np.ndarray, float):
        activation_derivative = self.derivative_activation_fn(np.dot(self.input_data, self.weights))
        delta = self.calculate_delta(target_output, upper_layer_weights, upper_layer_deltas, activation_derivative)
        delta_weights = learning_rate * delta * self.input_data
        self.weight_accumulator += delta_weights
        return self.weights, delta

    def activate(self, input_array: np.ndarray, is_training: bool = False):
        if is_training:
            self.input_data = input_array
        return self.activation_fn(np.dot(input_array, self.weights))

    def randomize_weights(self, reference_value: float, normalize_by_length: bool = False) -> None:
        if normalize_by_length:
            range_value = np.sqrt(1 / len(self.weights))
            self.weights = np.random.uniform(-range_value, range_value, len(self.weights))
        else:
            self.weights = np.random.uniform(-reference_value, reference_value, len(self.weights))

    def update_weights(self):
        self.weights += self.weight_accumulator
        if self.use_momentum:
            self.weights += self.momentum_coefficient * self.previous_delta_weights
            self.previous_delta_weights = self.weight_accumulator
        self.weight_accumulator = np.zeros_like(self.weights)

    def set_weights(self, new_weights):
        self.weights = new_weights

    def __str__(self) -> str:
        return f"SimplePerceptron(index={self.index}, weights={self.weights})"

    def __repr__(self) -> str:
        return self.__str__()

    def calculate_delta(self, target_output, upper_layer_weights, upper_layer_deltas, activation_derivative):
        if not self.is_hidden:
            return (target_output[self.index] - self.activate(self.input_data)) * activation_derivative
        else:
            return np.dot(upper_layer_deltas, upper_layer_weights[:, self.index]) * activation_derivative

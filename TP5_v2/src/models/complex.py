import numpy as np
from typing import List
import TP5_v2.src.models.simple as simple_perceptron_module

class ComplexPerceptron:
    def __init__(self, activation_function, derivative_activation_function,
                 network_layout: List[int], input_dimension: int,
                 fully_hidden: bool = False, enable_momentum: bool = False,
                 momentum_coefficient: float = 0.9):

        self.activation_fn = activation_function
        self.derivative_activation_fn = derivative_activation_function
        self.network = None
        self.input_dim = input_dimension
        self.initialize_network(network_layout, fully_hidden, enable_momentum, momentum_coefficient)

    def activate(self, input_data: np.ndarray, training_mode: bool = False) -> np.ndarray:
        activation_result = input_data
        for layer in self.network:
            activation_result = [perceptron.activate(activation_result, training_mode) for perceptron in layer]
            activation_result = np.transpose(np.array(activation_result))

        return activation_result

    def backpropagate(self, target_output: np.ndarray, learning_rate: float,
                      initial_upper_weights: np.ndarray, initial_upper_deltas: np.ndarray) -> (np.ndarray, np.ndarray):
        upper_weights = initial_upper_weights
        upper_deltas = initial_upper_deltas
        for layer in reversed(self.network):
            upper_weights, upper_deltas = zip(*[perceptron.backpropagate(target_output, upper_weights, upper_deltas, learning_rate)
                                                for perceptron in layer])
            upper_weights = np.array(upper_weights)
            upper_deltas = np.array(upper_deltas)

        return upper_weights, upper_deltas

    def randomize_weights(self, reference_value: float, normalize_by_length: bool = False) -> None:
        for layer in self.network:
            for perceptron in layer:
                perceptron.randomize_weights(reference_value, normalize_by_length)

    def update_weights(self) -> None:
        for layer in self.network:
            for perceptron in layer:
                perceptron.update_weights()

    def __str__(self) -> str:
        network_description = "ComplexPerceptron=("
        for layer_index, layer in enumerate(self.network):
            network_description += f"\nLayer {layer_index}: {layer}"
        return network_description + ")"

    def __repr__(self) -> str:
        return self.__str__()

    def initialize_network(self, network_layout: List[int],
                           fully_hidden: bool = False, enable_momentum: bool = False,
                           momentum_coefficient: float = 0.9) -> None:
        layout_array = np.array(network_layout, dtype=int)
        self.network = np.empty(len(layout_array), dtype=object)

        for layer_index, perceptron_count in enumerate(layout_array):
            self.network[layer_index] = np.empty(perceptron_count, dtype=object)
            next_dim = layout_array[layer_index - 1] if layer_index != 0 else self.input_dim
            is_hidden = fully_hidden or layer_index != (len(layout_array) - 1)

            for perceptron_index in range(perceptron_count):
                self.network[layer_index][perceptron_index] = simple_perceptron_module.SimplePerceptron(
                    self.activation_fn, self.derivative_activation_fn, next_dim,
                    is_hidden, perceptron_index, enable_momentum, momentum_coefficient)

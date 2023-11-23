import numpy as np
import TP5_v2.src.models.complex as cp
import TP5_v2.src.utils.functions as utils
from scipy import optimize


class AutoEncoder:
    def __init__(self, activation_fn, derived_activation_fn, layer_sizes, input_dim, latent_dim,
                 use_momentum=False, momentum_alpha=0.9):
        self.input_dim = input_dim
        self.latent_dim = latent_dim

        encoder_layers = layer_sizes.copy()
        encoder_layers.append(latent_dim)
        self.encoder = cp.ComplexPerceptron(activation_fn, derived_activation_fn, encoder_layers,
                                            input_dim, fully_hidden=True, enable_momentum=use_momentum,
                                            momentum_coefficient=momentum_alpha)

        decoder_layers = layer_sizes[::-1]
        decoder_layers.append(input_dim)
        self.decoder = cp.ComplexPerceptron(activation_fn, derived_activation_fn, decoder_layers,
                                            latent_dim, fully_hidden=False, enable_momentum=use_momentum,
                                            momentum_coefficient=momentum_alpha)

        self.optimization_errors = []

    def train(self, input_data, output_data, learning_rate):
        self.forward_pass(input_data, training=True)
        self.backward_pass(output_data, learning_rate)

    def forward_pass(self, input_data, training=False):
        encoder_output = self.encoder.activate(input_data, training_mode=training)
        return self.decoder.activate(encoder_output, training_mode=training)

    def encode(self, input_data):
        return self.encoder.activate(input_data, training_mode=False)

    def decode(self, latent_input):
        return self.decoder.activate(latent_input, training_mode=False)

    def backward_pass(self, expected_output, learning_rate):
        output_dimension = len(expected_output)
        sup_w, sup_delta = self.decoder.backpropagate(expected_output, learning_rate,
                                                      np.empty(output_dimension), np.empty(output_dimension))
        return self.encoder.backpropagate(expected_output, learning_rate, sup_w, sup_delta)

    def initialize_weights(self, reference, normalize_by_length=False):
        self.encoder.randomize_weights(reference, normalize_by_length)
        self.decoder.randomize_weights(reference, normalize_by_length)

    def update_weights(self):
        self.encoder.update_weights()
        self.decoder.update_weights()

    def compute_error(self, input_data, output_data, threshold, apply_threshold):
        activation = utils.discrete(self.forward_pass(input_data)[:, 1:], threshold, apply_threshold)
        target_output = output_data[:, 1:]
        return np.linalg.norm(target_output - activation) ** 2 / len(target_output)

    # Optimization Methods

    def flatten_weights(self):
        weights = []
        for layer in self.encoder.network:
            for perceptron in layer:
                weights.append(perceptron.weights)
        for layer in self.decoder.network:
            for perceptron in layer:
                weights.append(perceptron.weights)
        return np.hstack(np.array(weights, dtype=object))

    def reshape_weights(self, flat_weights):
        weight_index = 0
        for layer in self.encoder.network:
            for perceptron in layer:
                perceptron.set_weights(flat_weights[weight_index:weight_index + len(perceptron.weights)])
                weight_index += len(perceptron.weights)
        for layer in self.decoder.network:
            for perceptron in layer:
                perceptron.set_weights(flat_weights[weight_index:weight_index + len(perceptron.weights)])
                weight_index += len(perceptron.weights)

    def error_for_minimization(self, flat_weights, input_data, output_data, threshold, apply_threshold):
        self.reshape_weights(flat_weights)
        error = self.compute_error(input_data, output_data, threshold, apply_threshold)
        self.optimization_errors.append(error)
        return error

    def train_with_minimization(self, input_data, output_data, threshold, apply_threshold, optimization_method,
                                max_iterations, max_function_evals):
        flat_initial_weights = self.flatten_weights()
        optimization_result = optimize.minimize(self.error_for_minimization, flat_initial_weights,
                                                method=optimization_method,
                                                args=(input_data, output_data, threshold, apply_threshold),
                                                options={'maxiter': max_iterations, 'maxfev': max_function_evals,
                                                         'disp': True})
        self.reshape_weights(optimization_result.x)
        final_error = optimization_result.fun
        print(f'Final error: {final_error}')
        return final_error

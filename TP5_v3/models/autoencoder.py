import numpy as np
import TP5_v2.src.models.complex as cp


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

    def update_weights(self):
        self.encoder.update_weights()
        self.decoder.update_weights()

    def compute_error(self, input_data, output_data):
        activation = self.forward_pass(input_data)[:, 1:]
        target_output = output_data[:, 1:]
        return np.linalg.norm(target_output - activation) ** 2 / len(target_output)

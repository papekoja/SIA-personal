import numpy as np
import TP5_v2.src.utils.functions as f
import TP5_v2.src.models.complex as cp

from scipy import optimize


# uses 2 complex perceptron
class AutoEncoder(object):

    def __init__(self, activation_function, activation_function_derived,
                 layout: [int], data_dim: int, latent_dim: int,
                 momentum: bool = False, mom_alpha: float = 0.9):
        self.data_dim: int = data_dim
        self.latent_dim: int = latent_dim

        encoder_layout: [] = layout.copy()
        encoder_layout.append(latent_dim)
        self.encoder = cp.ComplexPerceptron(activation_function, activation_function_derived, encoder_layout,
                                            data_dim, full_hidden=True, momentum=momentum, mom_alpha=mom_alpha)

        decoder_layout: [] = layout[::-1]
        decoder_layout.append(data_dim)
        self.decoder = cp.ComplexPerceptron(activation_function, activation_function_derived, decoder_layout,
                                            latent_dim, full_hidden=False, momentum=momentum, mom_alpha=mom_alpha)

        # optimizer variables
        self.opt_err = []

    # performs the training on the auto-encoder
    def train(self, data_in: np.ndarray, data_out: np.ndarray, eta: float) -> None:
        self.activation(data_in, training=True)
        self.retro(data_out, eta)

    # propagates input along the encoder and decoder
    # returns always the output
    def activation(self, init_input: np.ndarray, training: bool = False) -> np.ndarray:
        encoder_out: np.ndarray = self.encoder.activation(init_input, training=training)
        return self.decoder.activation(encoder_out, training=training)

    # returns the activation out from the latent space
    def activation_to_latent_space(self, init_input: np.ndarray) -> np.ndarray:
        return self.encoder.activation(init_input, training=False)

    # returns the activation value of the decoder from the latent space (generate things)
    def activation_from_latent_space(self, init_input: np.ndarray) -> np.ndarray:
        return self.decoder.activation(init_input, training=False)

    # retro-propagates the difference with the expected out through the auto encoder
    # returns the input on retro-propagation
    def retro(self, expected_out: np.ndarray, eta: float) -> (np.ndarray, np.ndarray):
        out_dim: int = len(expected_out)
        sup_w, sup_delta = self.decoder.retro(expected_out, eta, np.empty(out_dim), np.empty(out_dim))
        return self.encoder.retro(expected_out, eta, sup_w, sup_delta)

    # initially the weights (w) start with 0, initialize/change them
    def randomize_w(self, ref: float, by_len: bool = False) -> None:
        self.encoder.randomize_w(ref, by_len)
        self.decoder.randomize_w(ref, by_len)

    # for epoch training updates each perceptron its weights
    def update_w(self) -> None:
        self.encoder.update_w()
        self.decoder.update_w()

    # calculates the error of the auto-encoder
    def error(self, data_in: np.ndarray, data_out: np.ndarray, trust: float, use_trust: bool) -> float:
        act: np.ndarray = f.discrete(self.activation(data_in)[:, 1:], trust, use_trust)
        out: np.ndarray = data_out[:, 1:]

        return (np.linalg.norm(out - act) ** 2) / len(out)

    # OPTIMIZATION METHODS

    # flatten weights to 1D
    def flatten_weights(self):
        w_matrix: [] = []
        # append encoder weights
        for layer in self.encoder.network:
            for s_p in layer:
                w_matrix.append(s_p.w)
        # apend decoder weights
        for layer in self.decoder.network:
            for s_p in layer:
                w_matrix.append(s_p.w)
        # flatten weights to become array
        return np.hstack(np.array(w_matrix, dtype=object))

    # unflatten weights to network
    def unflatten_weights(self, flat_w: np.ndarray):
        w_index: int = 0
        # append encoder weights
        for layer in self.encoder.network:
            for s_p in layer:
                s_p.set_w(flat_w[w_index:w_index + len(s_p.w)])
                w_index += len(s_p.w)
        # apend decoder weights
        for layer in self.decoder.network:
            for s_p in layer:
                s_p.set_w(flat_w[w_index:w_index + len(s_p.w)])
                w_index += len(s_p.w)

    # calculate error for minimizer
    def error_minimizer(self, flat_w: np.ndarray, data_in: np.ndarray, data_out: np.ndarray, trust: float,
                        use_trust: bool):
        # unflatten weights
        self.unflatten_weights(flat_w)
        # calculate error
        err = self.error(data_in, data_out, trust, use_trust)
        self.opt_err.append(err)
        # print(f'Optimizer error: {err}')
        return err

    # train autoencoder with minimizer
    def train_minimizer(self, data_in: np.ndarray, data_out: np.ndarray, trust: float, use_trust: bool, method: str,
                        max_iter: int, max_fev: int):
        # flatten weights
        flat_w = self.flatten_weights()
        # optimize error
        res = optimize.minimize(self.error_minimizer, flat_w, method=method, args=(data_in, data_out, trust, use_trust),
                                options={'maxiter': max_iter, 'maxfev': max_fev, 'disp': True})
        # unflatten weights
        self.unflatten_weights(res.x)
        # Error of the cost function
        final_err = res.fun
        print(f'Final error is {final_err}')
        return final_err

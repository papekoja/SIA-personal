import numpy as np
import TP5_v2.src.models.simple as sp


class ComplexPerceptron(object):

    def __init__(self, activation_function, activation_function_derived,
                 layout: [int], input_dim: int, full_hidden: bool = False,
                 momentum: bool = False, mom_alpha: float = 0.9):

        self.act_func = activation_function
        self.act_func_der = activation_function_derived
        self.network = None
        self.in_dim: int = input_dim
        self.__init_network(layout, full_hidden, momentum, mom_alpha)

    # propagates input along the entire network
    # in case of training, saves  the input for later computation on retro propagation
    # returns the final activation value
    def activation(self, init_input: np.ndarray, training: bool = False) -> np.ndarray:
        activation_values = init_input
        for layer in self.network:
            activation_values = list(map(lambda s_p: s_p.activation(activation_values, training=training), layer))
            activation_values = np.transpose(np.asarray(activation_values))

        return activation_values

    # retro-propagates the error of the network given the true input
    # takes the given suo_w and sup_delta as initial values
    def retro(self, expected_out: np.ndarray, eta: float,
              init_sup_w: np.ndarray, init_sup_delta: np.ndarray) -> (np.ndarray, np.ndarray):
        sup_w: np.ndarray = init_sup_w
        sup_delta: np.ndarray = init_sup_delta
        for layer in reversed(self.network):
            sup_w, sup_delta = zip(*list(map(lambda s_p: s_p.retro(expected_out, sup_w, sup_delta, eta), layer)))
            # convert tuples to lists (used in the next layer)
            sup_w = np.asarray(sup_w)
            sup_delta = np.asarray(sup_delta)

        return sup_w, sup_delta

    # resets the w to a randomize range if desired for the entire network
    # if randomize is false, then does nothing
    def randomize_w(self, ref: float, by_len: bool = False) -> None:
        for layer in self.network:
            list(map(lambda s_p: s_p.randomize_w(ref, by_len), layer))

    # for epoch training updates each w with its accum
    def update_w(self) -> None:
        for layer in self.network:
            list(map(lambda s_p: s_p.update_w(), layer))

    def __str__(self) -> str:
        out: str = "CPerceptron=("
        for i, layer in enumerate(self.network):
            out += f"\nlayer {i}=" + str(layer)
        return out + ")"

    def __repr__(self) -> str:
        out: str = "CPerceptron=("
        for i, layer in enumerate(self.network):
            out += f"\nlayer {i}=" + str(layer)
        return out + ")"

    # private methods

    # initializes the entire network of perceptron given a layout
    def __init_network(self, hidden_layout: [int], full_hidden: bool = False,
                       momentum: bool = False, mom_alpha: float = 0.9) -> None:
        # the final amount of perceptron depends on expected output dimension
        layout: np.ndarray = np.array(hidden_layout, dtype=int)

        # initialize the length of the network
        self.network = np.empty(shape=len(layout), dtype=np.ndarray)

        # for each level, get its count of perceptron
        for level in range(len(layout)):

            # initialize (empty) level with its amount of perceptron
            self.network[level] = np.empty(shape=layout[level], dtype=sp.SimplePerceptron)

            # the dimension of the next level is set from the previous or the input data
            dim: int = layout[level - 1] if level != 0 else self.in_dim

            # if its a full hidden network all layers are hidden
            # if not check if it is the last layer
            hidden: bool = full_hidden or level != (len(layout) - 1)

            # create the corresponding amount of perceptron
            for index in range(layout[level]):
                # for each index and level, create the corresponding perceptron
                self.network[level][index] = \
                    sp.SimplePerceptron(self.act_func, self.act_func_der, dim, hidden, index, momentum, mom_alpha)

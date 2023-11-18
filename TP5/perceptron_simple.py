import numpy as np


class PerceptronSimple(object):

    def __init__(self, activation_function, activation_function_derived,
                 dimension: int, hidden: bool = False, index: int = 0,
                 momentum: bool = False, mom_alpha: float = 0.9):
        self.index = index
        self.hidden: bool = hidden
        self.act_func = activation_function
        self.act_func_der = activation_function_derived
        self.w: np.ndarray = np.zeros(dimension)
        self.input: np.ndarray = np.zeros(dimension)

        """Se crea un vector de ceros prev_delta_w de longitud dimension. Este vector se utiliza para almacenar los cambios acumulativos de pesos en la iteración anterior. 
      asimismo, se usa un booleano para referirse a si se usa o no el momento. Finalmente, mom_alpha es el coeficiente de momento"""
        
        self.prev_delta_w = np.zeros(dimension)
        self.momentum: bool = momentum
        self.mom_alpha: float = mom_alpha

        # for epoch training
        self.accu_w = np.zeros(dimension)

        # out es un 1D array que es usado unicamente en la capa de salida
    # sup_w es una 2D matrix con todos los vectores W de la capa de salida---capas ocultas
    # sup_delta es un 1D array de todos los valores de los deltas de la capa de calida ---- capas ocultas

    def retropropagation(self, out: np.ndarray, sup_w: np.ndarray, sup_delta: np.ndarray, eta: float) \
            -> (np.ndarray, float):
        # activation for this neuron
        activation_derived = self.act_func_der(np.dot(self.input, self.w))

        # delta sub i using the activation values
        if not self.hidden:
            delta = (out[self.index] - self.activation(self.input)) * activation_derived
        else:
            delta = np.dot(sup_delta, sup_w[:, self.index]) * activation_derived

        # calculate the delta w
        delta_w = (eta * delta * self.input)

        # epoch training accumulation
        self.accu_w += delta_w

        return self.w, delta
    
    def activation(self, input_arr: np.ndarray, training: bool = False):
        if training:
            self.input = input_arr

        # activation for this neuron, could be int or float, or an array in case is the full dataset
        return self.act_func(np.dot(input_arr, self.w))

    # resets the w to a randomize range
    def randomize_w(self, ref: float, by_len: bool = False) -> None:
        if by_len:
            self.w = np.random.uniform(-np.sqrt(1 / len(self.w)), np.sqrt(1 / len(self.w)), len(self.w))
        else:
            self.w = np.random.uniform(-ref, ref, len(self.w))

    # for epoch training delta is the accum value
    def update_w(self):
        self.w += self.accu_w

        # in case of momentum, calculate delta w and update values
        if self.momentum:
            self.w += self.mom_alpha * self.prev_delta_w
            self.prev_delta_w = self.accu_w

    def set_w(self, w):
        self.w = w

   
import numpy as np
import math


def get_activation_functions(name: str, beta: float):
    act_funcs_dict = {
        "linear": lambda x: x,
        "tanh": lambda x: np.tanh(x * beta),
        "exp": lambda x: 1 / (1 + math.e ** (-2 * beta * x)),
    }

    # the derived activation func, if it cant be, then return always 1
    act_funcs_der_dict = {
        "linear": lambda x: 1,
        "tanh": lambda x: beta * (1 - (act_funcs_dict["tanh"](x) ** 2)),
        "exp": lambda x: 2 * beta * act_funcs_dict["exp"](x) * (1 - act_funcs_dict["exp"](x))
    }

    return act_funcs_dict[name], act_funcs_der_dict[name]
